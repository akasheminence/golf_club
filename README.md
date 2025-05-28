# 🏌️‍♂️ GolfAnalytics

![Python](https://img.shields.io/badge/python-%3E=3.10-blue)
![PyTorch](https://img.shields.io/badge/torch-%3E=2.3.1-orange)
![Torchvision](https://img.shields.io/badge/torchvision-%3E=0.18.2-yellow)
![CUDA](https://img.shields.io/badge/CUDA-12.1-green)
![License](https://img.shields.io/github/license/affanrasheed/GolfAnalytics)

An advanced golf swing analysis pipeline using **Grounding DINO**, **Segment Anything v2 (SAM2)**, and **MediaPipe**. Upload your swing videos and receive an in-depth analysis with visual overlays and reports.

---

## ✅ Minimum Requirements

- OS: Ubuntu or **WSL 2** (Windows)
- Python: `>=3.10`
- PyTorch: `>=2.3.1`
- Torchvision: `>=0.18.2`
- CUDA: `12.1`

---

## 🚀 Installation Guide

### 1. Clone the Repository

```bash
git clone https://github.com/affanrasheed/GolfAnalytics.git
cd GolfAnalytics
```

---

### 2. Download Pretrained Weights

#### 🔹 Segment Anything v2 (SAM2)

```bash
cd checkpoints
bash download_ckpts.sh
```

#### 🔹 Grounding DINO

```bash
cd ../gdino_checkpoints
bash download_ckpts.sh
```

---

### 3. Set Up Environment

#### 🔸 Install Anaconda

[Download Anaconda](https://www.anaconda.com/products/distribution)

#### 🔸 Create Conda Environment

```bash
conda create -n env_golf_analytics python=3.10
conda activate env_golf_analytics
```

#### 🔸 Install PyTorch with CUDA 12.1

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

---

### 4. Install Project Dependencies

#### 🔹 Segment Anything v2

```bash
cd GolfAnalytics
pip install -e ".[notebooks]"
```

#### 🔹 Grounding DINO and Additional Packages

```bash
pip install scipy transformers addict yapf pycocotools timm mediapipe firebase-admin runpod
```

---

## ⚙️ Run the API Server

```bash
python api_app.py
```

Server will start at:

```
http://0.0.0.0:8001
```

---

## 📡 API Usage

### 🔸 Endpoint

```
POST http://0.0.0.0:8001/upload/
```

### 🔸 Headers

```
Accept: application/json
```

### 🔸 Form Data

| Key     | Type   | Description        |
|---------|--------|--------------------|
| `file1` | File   | Front view video (optional) |
| `file2` | File   | Side view video (optional)  |

---

## 🧠 System Behavior

| Scenario              | Behavior                               |
|-----------------------|----------------------------------------|
| Both `file1` & `file2`| Runs `golf_swing_analysis`             |
| Only `file1`          | Runs `golf_swing_analys_front`         |
| Only `file2`          | Runs `golf_swing_analys_side`          |
| Invalid landmarks     | Returns: `"Landmarks are not visible"` |
| Valid input           | Returns `.zip` with results & report   |

---

## 📥 Sample cURL Request

```bash
curl --location 'http://0.0.0.0:8001/upload/' --header 'accept: application/json' --form 'file1=@"/path/to/front_video.MOV"' --form 'file2=@"/path/to/side_video.MOV"'
```

---

## 📦 Output

Returns a `.zip` file containing:

- 🎥 Annotated result videos  
- 🖼️ Extracted frames  
- 📊 Swing analysis report  

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---
