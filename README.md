# GolfAnalytics
## Description

## Minimum Requirements
1. Ubuntu or WSL 2 if on Windows
2. python>=3.10
3. torch>=2.3.1 and torchvision>=0.18.2
4. Cuda 12.1

## Installation
1. Clone the repo
```
git clone https://github.com/affanrasheed/GolfAnalytics.git
cd GolfAnalytics
```

2. Download `SAM2` Pretrained Model Weights
```
cd checkpoints
bash download_ckpts.sh
```

3. Download `Grounding DINO` Pretrained Model Weights
```
cd ../gdino_checkpoints
bash download_ckpts.sh
```
4. Install anaconda from [here](https://docs.anaconda.com/anaconda/install/)

5. Create Conda Envirnment with `python>=3.10`
```
conda create -n env_golf_analytics python=3.10
conda activate env_golf_analytics
```

6. Install `CUDA 12.1` from [here](https://developer.nvidia.com/cuda-downloads)

7. Install Pytorch Envirnment. Minimum requirment `torch >= 2.3.1` and `torchvision>=0.18.1`
```
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

8. Install `Segment Anything 2`
```
cd GolfAnalytics
pip install -e ".[notebooks]"
```

9. Install `Grounding DINO`
```
DataLoader
```

10. Additional packages 
```
pip install scipy transformers addict yapf pycocotools timm mediapipe firebase-admin runpod
```
Run the Fast API: 
 
 bash 
python api_app.py 

 This will start the server on http://0.0.0.0:8001 (or as configured). 
 

 

📡 API Endpoint 

URL: 

http://0.0.0.0:8001/upload/ 

 

Method: POST 

Headers: 

Accept: application/json 

 

Form Data: 

file1=@"/path/to/front_video.MOV"  # Optional: Front view 

file2=@"/path/to/side_video.MOV"   # Optional: Side view 

 

Shape 

🧠 System Behavior 

Scenario 

Behavior 

Both file1 and file2 

Runs full golf_swing_analysis 

Only file1 

Runs golf_swing_analys_front 

Only file2 

Runs golf_swing_analys_side 

Blurry or bad landmarks 

Returns error: "Landmarks are not visible" 

Valid input 

Returns link to download .zip file with: 

 

→ Result videos, extracted frames, and swing analysis report. 

Shape 

📥 Sample cURL Request 

curl --location 'http://0.0.0.0:8001/upload/' \ 

--header 'accept: application/json' \ 

--form 'file1=@"/home/webexpert/Downloads/may_swing_front_2.MOV"' \ 

--form 'file2=@"/home/webexpert/Downloads/may_swing_side_2.MOV"' 

 

Shape 

 

 
```


