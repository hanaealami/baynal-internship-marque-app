# Marque — Car Marque Detection

Flask web app with signup/login. A logged-in user uploads a car photo; a
YOLOv8 model detects the car's marque (brand) and shows the annotated image
with confidence scores.

## Status

**Working**
- Signup / login / logout (SQLite, hashed passwords, session-protected routes)
- Brand detection with a self-trained YOLOv8n model (19 brands; precision 0.918, recall 0.823, mAP50 0.915, mAP50-95 0.901 on the validation split)
- Damage classification using a pretrained Hugging Face model (`beingamit99/car_damage_detection`), loaded per request to keep idle memory low
- Docker image and a working deployment on an AWS EC2 instance

**Limits, stated plainly**
- The training data is a small Roboflow dataset (1,708 train / 190 validation images, about 10 validation instances per class), so per-class metrics are noisy. BMW and Tesla are the weakest classes.
- The damage model is pretrained, not trained here. It has no "no damage" class, so results under 55% confidence are reported as no visible damage.
- This version loads two models and is heavier than a 1 GB instance can comfortably run. The lean deployed variant used a single model.
- There is no automated test suite yet.

**Companion projects (separate repos)**
- A Flutter mobile prototype. It currently runs on sample data and is not connected to this backend.

## Possible improvements
- More training images for the weakest classes, plus a held-out test split
- A JSON API (token auth) so the mobile app can call the real models
- Automated tests for the auth flow and upload validation
- Email notifications for registered users through the mail setup

## What's in here

```
app.py              Flask app: auth (SQLite) + YOLO inference
templates/          login, signup, detect pages
static/uploads/     annotated results are written here at runtime
Dockerfile          CPU-only image
requirements.txt    dependencies
best.pt             the trained model (already included, see below)
```

## About the model

`best.pt` is a YOLOv8n model fine-tuned on a 19-brand car dataset
(Alpha Romeo, Audi, Bentley, BMW, Ferrari, Ford, Lamborghini, Porsche,
Tesla, Toyota, and more — see `TRAINING_RESULTS.md` for the full list
and metrics). It's already committed to this repo, so the app runs
as-is with no extra setup.

Want to swap in a different model instead? A few real pretrained
options if you don't want to train your own:

- [Brand Logo recognition — YOLOv8](https://universe.roboflow.com/myawesomeworkspace/brand-logo-recognition-yolov8-uichv) (Roboflow Universe) — detects brand logos rather than whole-car brand classification, closer to a logo-spotting use case.
- [Vehicle Detect — YOLOv8](https://universe.roboflow.com/yolov8-m2va2/vehicle-detect-qxprz) (Roboflow Universe) — general vehicle detection, good as a base to fine-tune on your own brand-labeled data.
- [Roboflow Universe YOLOv8 model search](https://universe.roboflow.com/search?q=model%3Ayolov8) — browse other pretrained YOLOv8 models and datasets, including logo/brand-specific ones, that can be exported and dropped in as `best.pt`.

Any YOLOv8 `.pt` file works as a drop-in replacement, as long as
`model.names` in `app.py` matches the classes it was trained on.

## Run locally (no Docker)

```bash
pip install -r requirements.txt
python app.py
# open http://localhost:5000
```

## Run with Docker (local test)

```bash
docker build -t marque .
docker run -p 5000:5000 marque
# open http://localhost:5000
```

Create an account, log in, upload a car image, see the detected marque.

---

## Deploy to AWS EC2

### 1. Launch an instance
- EC2 → Launch instance → Ubuntu 24.04
- Type: **t3.small** or bigger (t2.micro's 1 GB RAM is too little for torch)
- Key pair: create/download one for SSH
- Security group: allow **SSH (22)** and **HTTP (80)** from anywhere

### 2. Install Docker on the instance
```bash
ssh -i your-key.pem ubuntu@<EC2_PUBLIC_IP>
sudo apt update && sudo apt install -y docker.io
sudo usermod -aG docker ubuntu && exit   # re-login to apply
```

### 3. Get the app onto the instance
From your laptop, copy the whole folder (model included):
```bash
scp -i your-key.pem -r car-detect-app ubuntu@<EC2_PUBLIC_IP>:~/
```

### 4. Build and run
```bash
ssh -i your-key.pem ubuntu@<EC2_PUBLIC_IP>
cd car-detect-app
docker build -t marque .
docker run -d --restart unless-stopped -p 80:5000 \
  -e SECRET_KEY="pick-a-long-random-string" marque
```

Open `http://<EC2_PUBLIC_IP>` in a browser.

### Notes
- `-p 80:5000` maps the instance's port 80 to the app so no port is needed in the URL.
- `--restart unless-stopped` brings the app back after a reboot.
- Set a real `SECRET_KEY` in production (it signs login sessions).
- The SQLite DB and uploaded images live inside the container. To keep them
  across rebuilds, mount volumes:
  `-v ~/marque-db:/app -v ~/marque-uploads:/app/static/uploads` (optional).
