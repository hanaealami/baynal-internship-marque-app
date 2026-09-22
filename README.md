# Marque — Car Marque Detection

Flask web app with signup/login. A logged-in user uploads a car photo; a
YOLOv8 model detects the car's marque (brand) and shows the annotated image
with confidence scores.

## What's in here

```
app.py              Flask app: auth (SQLite) + YOLO inference
templates/          login, signup, detect pages
static/uploads/     annotated results are written here at runtime
Dockerfile          CPU-only image
requirements.txt    dependencies
best.pt             <-- YOU ADD THIS: your trained model
```

## Step 0 — Add your model

Copy your trained `best.pt` into this folder, next to `app.py`.
Without it the app won't start.

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
