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
## Input Request
1. Parameters
    1. `input_video` [It should be a firebase storage url for input video, required parameter]
    2. `shot_type` [It can be back or side represent shot type, required parameter]
    3. `firestore_collection` [Firestore collection name where processed output flags along with processed video url to be stored, optional parameter]
    4. `firebase_cred` [Firebase service account json file path, optional parameter] 
    5. `firebase_bucket` [Firebase storage database address, optional parameter]
    6. `firebase_storage_location` [Firebase storage database location where video will be stored, optional parameter]
    7. `firestore_doc_id` [Firestore location where flag and other url will be stored, optional parameter]


2. Sample Request
curl -X POST http://api/runsync 
    -H "Content-Type: application/json" 
    -d '{"input": 
                {"input_video": "gs://",
                "shot_type": "back",
                "firestore_collection":"videos2",
                "firebase_cred","cred.json",
                "firebase_bucket",".appspot.com",
                "firebase_storage_location":"user1/v1",
                "firestore_doc_id":"user_v1"}}'

## Testing
1. Testing endpoint
Create test_input.json file as provided. Change the inputs accordingly and run the command below
```
python main.py
```


