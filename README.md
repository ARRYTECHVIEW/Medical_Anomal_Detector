# 🏥 Medical Image Anomaly Detector

> AI-powered second-opinion system for medical image analysis. Detects anomalies in X-rays, scans, and pathology images using deep learning with Grad-CAM visual explanations.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange?style=flat-square)
![Flask](https://img.shields.io/badge/Flask-3.0-lightgrey?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

> ⚠️ **Disclaimer:** For research and educational purposes only. Not a substitute for professional medical diagnosis.

---

## ✨ Key Features

| Feature | Detail |
|---|---|
| 🧠 Backbone | EfficientNet-B0 / ResNet-50 / DenseNet-121 (configurable) |
| 🔥 Grad-CAM | Visual heatmaps showing WHICH regions triggered prediction |
| 📊 Metrics | Accuracy, F1, ROC-AUC, confusion matrix |
| 🌐 Dashboard | Drag-and-drop web UI with instant results |
| 🔌 REST API | `/api/predict` endpoint for integration |
| ⚡ Training | AMP mixed precision, early stopping, cosine LR |
| 🎯 Transfer learning | 2-phase: freeze backbone → fine-tune |

---

## 🏗️ Architecture

```
Input Image (X-ray / scan / pathology)
        │
        ▼
┌─────────────────────────┐
│  Preprocessing          │  Resize 224×224, ImageNet normalize
│  + Augmentation         │  Flip, rotate, jitter (train only)
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  CNN Backbone           │  EfficientNet-B0 (default)
│  (pretrained ImageNet)  │  ResNet-50 / DenseNet-121 optional
└───────────┬─────────────┘
            │  feature vector
            ▼
┌─────────────────────────┐
│  Classifier Head        │  Dropout → FC(256) → ReLU → FC(N)
└───────────┬─────────────┘
            │  class logits
            ▼
┌─────────────────────────┐   ┌──────────────────────┐
│  Softmax Probabilities  │──▶│  Grad-CAM Heatmap     │
│  + Predicted Label      │   │  (explainability)     │
└─────────────────────────┘   └──────────────────────┘
```

---

## 📁 Project Structure

```
medical-anomaly-detector/
├── main.py                     # Entry point (train/predict/api/demo)
│
├── model/
│   ├── architecture.py         # EfficientNet/ResNet/DenseNet + classifier head
│   ├── dataset.py              # MedicalImageDataset + transforms + DataLoader
│   ├── trainer.py              # Full training loop (AMP, early stopping, checkpoints)
│   └── predictor.py            # Inference + Grad-CAM heatmap generation
│
├── api/
│   ├── server.py               # Flask app + REST API + dashboard routes
│   └── templates/
│       └── index.html          # Web dashboard (drag-drop upload + live results)
│
├── utils/
│   ├── demo.py                 # Synthetic image generator + demo runner
│   └── metrics.py              # Accuracy, F1, ROC-AUC, confusion matrix
│
├── tests/
│   └── test_core.py            # 10 pytest unit tests (no GPU needed)
│
├── data/                       # Auto-created at runtime
│   ├── dataset/                # Your training data goes here
│   ├── uploads/                # API uploaded images
│   └── results/                # Grad-CAM heatmap outputs
│
├── models/                     # Saved model checkpoints
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install

```bash
git clone https://github.com/YOUR_USERNAME/medical-anomaly-detector.git
cd medical-anomaly-detector

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

---

### 2. Run demo (no dataset needed)

```bash
python main.py demo
```

Generates synthetic medical images and runs predictions with Grad-CAM. Works even without a trained model.

---

### 3. Start the web dashboard

```bash
python main.py dashboard
```

Open **http://127.0.0.1:5000** — drag and drop any medical image to analyse it.

---

### 4. Train on your own dataset

Organise your dataset like this:

```
data/dataset/
├── train/
│   ├── normal/     ← normal images
│   └── anomaly/    ← anomalous images
├── val/
│   ├── normal/
│   └── anomaly/
└── test/           (optional)
    ├── normal/
    └── anomaly/
```

Then train:

```bash
python main.py train \
  --data-dir data/dataset \
  --backbone efficientnet_b0 \
  --epochs 30 \
  --batch-size 32 \
  --lr 0.0001
```

---

### 5. Predict a single image

```bash
python main.py predict \
  --image path/to/scan.jpg \
  --model models/best_model.pth
```

---

### 6. Evaluate on test set

```bash
python main.py evaluate \
  --data-dir data/dataset \
  --model models/best_model.pth
```

---

## 🌐 REST API

### Predict

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -F "image=@path/to/scan.jpg"
```

**Response:**
```json
{
  "label": "anomaly",
  "confidence": 91.3,
  "scores": { "normal": 8.7, "anomaly": 91.3 },
  "heatmap_url": "/results/gradcam_a1b2c3d4.jpg",
  "prediction_id": "a1b2c3d4"
}
```

### Health check

```bash
curl http://127.0.0.1:5000/api/health
```

---

## ⚙️ Configuration

| Argument | Default | Description |
|---|---|---|
| `--backbone` | `efficientnet_b0` | `efficientnet_b0` / `resnet50` / `densenet121` |
| `--epochs` | `30` | Training epochs |
| `--batch-size` | `32` | Batch size |
| `--lr` | `0.0001` | Learning rate |
| `--classes` | `normal anomaly` | Custom class names |

Multi-class example:
```bash
python main.py train --classes normal pneumonia tuberculosis --data-dir data/chest_xray
```

---

## 🧪 Tests

```bash
pytest tests/ -v
```

10 tests covering: dataset loading, model forward pass, Grad-CAM, metrics, demo generation.

---

## 📦 Public Datasets to Use

| Dataset | Classes | Size | Link |
|---|---|---|---|
| Chest X-Ray (Kaggle) | Normal / Pneumonia | 5,856 images | [kaggle.com](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) |
| HAM10000 | 7 skin lesion types | 10,015 images | [kaggle.com](https://www.kaggle.com/kmader/skin-lesion-analysis-toward-melanoma-detection) |
| Brain MRI | Normal / Tumor | 3,264 images | [kaggle.com](https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection) |
| COVID-19 X-Ray | Normal / COVID | 3,616 images | [kaggle.com](https://www.kaggle.com/tawsifurrahman/covid19-radiography-database) |

---

## 🛠️ Tech Stack

- **PyTorch** — model training, inference, AMP
- **TorchVision** — pretrained backbones + transforms
- **OpenCV** — Grad-CAM heatmap overlay
- **Flask** — web dashboard + REST API
- **Pillow / NumPy** — image processing + synthetic data

---

## 🔮 Roadmap

- [ ] DICOM (.dcm) file support
- [ ] Multi-label classification
- [ ] TensorRT / ONNX export for production deployment
- [ ] Docker container
- [ ] Batch prediction endpoint
- [ ] Training progress live chart in dashboard

---

## 👤 Author

**Your Name**  
[LinkedIn](https://linkedin.com/in/yourprofile) · [GitHub](https://github.com/yourusername)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
