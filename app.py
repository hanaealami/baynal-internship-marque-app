import gc
import os
import sqlite3
import uuid
from functools import wraps

from flask import (
    Flask, render_template, request, redirect, url_for,
    session, flash, g
)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
DB_PATH = os.path.join(BASE_DIR, "app.db")
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")
ALLOWED = {"png", "jpg", "jpeg", "bmp", "webp"}

os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "change-me-in-production")
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB cap

# Brand-detection model stays resident (small, used every request).
model = YOLO(MODEL_PATH)

DAMAGE_MODEL_NAME = "beingamit99/car_damage_detection"
# This model always picks one of its 6 damage classes, even on undamaged
# cars (it has no "no damage" class). Below this confidence, treat the
# pick as noise rather than a real finding.
DAMAGE_CONF_THRESHOLD = 55.0


def detect_damage(image_path):
    # Loaded and freed per request (not kept resident like the YOLO model)
    # to keep idle memory low.
    processor = AutoImageProcessor.from_pretrained(DAMAGE_MODEL_NAME, local_files_only=True)
    damage_model = AutoModelForImageClassification.from_pretrained(
        DAMAGE_MODEL_NAME, local_files_only=True
    )
    damage_model.eval()

    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        logits = damage_model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    top_id = int(torch.argmax(probs))
    conf = round(float(probs[top_id]) * 100, 1)
    result = {
        "label": damage_model.config.id2label[top_id] if conf >= DAMAGE_CONF_THRESHOLD else None,
        "conf": conf,
    }

    del processor, damage_model, inputs, logits, probs
    gc.collect()
    return result


# ---------- database ----------
def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(exc):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    db = sqlite3.connect(DB_PATH)
    db.execute(
        """CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        )"""
    )
    db.commit()
    db.close()


# ---------- auth helpers ----------
def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return view(*args, **kwargs)
    return wrapped


def allowed_file(name):
    return "." in name and name.rsplit(".", 1)[1].lower() in ALLOWED


# ---------- routes ----------
@app.route("/")
def index():
    if "user_id" in session:
        return redirect(url_for("detect"))
    return redirect(url_for("login"))


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        if not username or not password:
            flash("Enter a username and password.")
        elif len(password) < 4:
            flash("Password must be at least 4 characters.")
        else:
            db = get_db()
            try:
                db.execute(
                    "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                    (username, generate_password_hash(password)),
                )
                db.commit()
                return redirect(url_for("login"))
            except sqlite3.IntegrityError:
                flash("That username is taken.")
    return render_template("signup.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        db = get_db()
        user = db.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ).fetchone()
        if user and check_password_hash(user["password_hash"], password):
            session.clear()
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            return redirect(url_for("detect"))
        flash("Wrong username or password.")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/detect", methods=["GET", "POST"])
@login_required
def detect():
    if request.method == "POST":
        file = request.files.get("image")
        if not file or file.filename == "":
            flash("Choose an image first.")
            return render_template("detect.html")
        if not allowed_file(file.filename):
            flash("That file type isn't supported. Use JPG, PNG, BMP, or WEBP.")
            return render_template("detect.html")

        stem = uuid.uuid4().hex
        ext = secure_filename(file.filename).rsplit(".", 1)[1].lower()
        in_name = f"{stem}.{ext}"
        in_path = os.path.join(UPLOAD_DIR, in_name)
        file.save(in_path)

        # run inference; save annotated image
        results = model(in_path)
        r = results[0]
        out_name = f"{stem}_out.jpg"
        out_path = os.path.join(UPLOAD_DIR, out_name)
        r.save(filename=out_path)

        detections = []
        for box in r.boxes:
            cls_id = int(box.cls[0])
            detections.append({
                "brand": model.names[cls_id],
                "conf": round(float(box.conf[0]) * 100, 1),
            })
        detections.sort(key=lambda d: d["conf"], reverse=True)
        damage = detect_damage(in_path)

        return render_template(
            "detect.html",
            result_image=url_for("static", filename=f"uploads/{out_name}"),
            damage=damage,
            detections=detections,
        )

    return render_template("detect.html")


if __name__ == "__main__":
    init_db()
    app.run(host="0.0.0.0", port=5000)
