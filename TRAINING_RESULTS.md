# Training Results — Car Brand Detection Model

Trained in Google Colab (Tesla T4 GPU).

## Setup
- Base model: `yolov8n.pt` (YOLOv8 nano)
- Dataset: Roboflow `Car-Brand-Detection-1` — 1708 train images, 190 val images
- Classes: 19 car brands
- Epochs: 50, image size 640, batch 16, optimizer AdamW (auto lr=0.000435)
- Training time: ~0.48 hours

## Overall metrics (final, on validation set)

| Metric | Value |
|---|---|
| Precision | 0.915 |
| Recall | 0.823 |
| mAP50 | 0.915 |
| mAP50-95 | 0.896 |

## Per-class results

| Class | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|
| alpha-romeo | 10 | 10 | 1.000 | 0.942 | 0.995 | 0.995 |
| audi | 9 | 9 | 1.000 | 0.702 | 0.975 | 0.975 |
| bentley | 10 | 11 | 0.985 | 1.000 | 0.995 | 0.995 |
| benz | 9 | 9 | 0.973 | 0.889 | 0.906 | 0.904 |
| bmw | 10 | 11 | 0.910 | 0.818 | 0.815 | 0.805 |
| cadillac | 9 | 9 | 0.875 | 0.781 | 0.973 | 0.962 |
| dodge | 10 | 10 | 0.835 | 0.700 | 0.904 | 0.904 |
| ferrari | 10 | 10 | 0.972 | 0.700 | 0.876 | 0.841 |
| ford | 10 | 10 | 0.951 | 0.900 | 0.986 | 0.955 |
| ford-mustang | 10 | 10 | 0.937 | 1.000 | 0.995 | 0.975 |
| hyundai | 10 | 10 | 0.904 | 0.700 | 0.835 | 0.835 |
| kia | 10 | 10 | 0.692 | 0.700 | 0.779 | 0.779 |
| lamborghini | 10 | 10 | 0.939 | 1.000 | 0.995 | 0.977 |
| lexus | 10 | 10 | 0.872 | 0.900 | 0.962 | 0.950 |
| maserati | 9 | 11 | 1.000 | 0.848 | 0.981 | 0.958 |
| porsche | 10 | 10 | 1.000 | 0.782 | 0.933 | 0.928 |
| rolls-royce | 10 | 10 | 0.909 | 1.000 | 0.986 | 0.820 |
| tesla | 9 | 9 | 0.815 | 0.494 | 0.674 | 0.649 |
| toyota | 9 | 9 | 0.822 | 0.778 | 0.823 | 0.820 |

## Notes
- Weakest classes: **tesla** (mAP50-95 0.649, recall 0.494) and **kia** (0.779). Worth more training images for these if accuracy needs improving.
- Speed: 0.3ms preprocess, 2.3ms inference, 4.8ms postprocess per image (on T4 GPU; CPU inference in the deployed Flask app will be slower but fine for single-image requests).
- Full plots (confusion matrix, PR curves, results.png, labels.jpg) are saved in Colab at `/content/drive/MyDrive/car_brand_model/train/` — pull those into this repo too if you want the images archived alongside this summary, not just the numbers.
