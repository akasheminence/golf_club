import os
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
import torch
torch.cuda.empty_cache()


import sys
import cv2
import torch
import numpy as np
import supervision as sv
from torchvision.ops import box_convert
from pathlib import Path
from tqdm import tqdm
import json
import mediapipe as mp
import shutil
import zipfile
import logging
import uuid
import base64
# Firebase imports
import time
from datetime import datetime

# Import your helper functions and models
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
from utils.track_utils import sample_points_from_masks
from utils.video_utils import create_video_from_images
from utils.helper import parse_gs_url, filter_detections_by_label, zip_folder, convert_ndarray_to_list
from utils.helper import (
    player_pose1,
    get_rotated_bounding_box,
    split_oriented_bounding_box,
    fit_line_to_segmentation,
    find_closest_point_to_hand,
    draw_reference_axis_and_calculate_angle_for_back_shot,
    draw_reference_axis_and_calculate_angle_for_side_shot,
    get_centroid_from_mask,
    frame_skipping_on_angle_anomaly
)
import json
import numpy as np

import cv2
import torch
import time
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed





import copy
import numpy as np
import cv2
import os
import argparse

os.environ["DISPLAY"] = ""

from OpenPosePyTorch.src import model
from OpenPosePyTorch.src import util
from OpenPosePyTorch.src.body import Body
from OpenPosePyTorch.src.hand import Hand

model_type = 'body25'
model_path = 'OpenPosePyTorch/model/pose_iter_584000.caffemodel.pt'
body_estimation = Body(model_path, model_type)
hand_estimation = Hand('OpenPosePyTorch/model/hand_pose_model.pth')

# Helper function to convert all numpy types to Python types recursively
def convert_to_python_types(obj):
    if isinstance(obj, dict):
        return {k: convert_to_python_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_python_types(i) for i in obj]
    elif isinstance(obj, np.generic):  # np.float32, np.int32, etc.
        return obj.item()
    else:
        return obj





def mediapipe_style_pose(frame):
    POSE_LANDMARK_NAMES = [
        "NOSE",
        "LEFT_EYE_INNER", "LEFT_EYE", "LEFT_EYE_OUTER",
        "RIGHT_EYE_INNER", "RIGHT_EYE", "RIGHT_EYE_OUTER",
        "LEFT_EAR", "RIGHT_EAR",
        "MOUTH_LEFT", "MOUTH_RIGHT",
        "LEFT_SHOULDER", "RIGHT_SHOULDER",
        "LEFT_ELBOW", "RIGHT_ELBOW",
        "LEFT_WRIST", "RIGHT_WRIST",
        "LEFT_PINKY", "RIGHT_PINKY",
        "LEFT_INDEX", "RIGHT_INDEX",
        "LEFT_THUMB", "RIGHT_THUMB",
        "LEFT_HIP", "RIGHT_HIP",
        "LEFT_KNEE", "RIGHT_KNEE",
        "LEFT_ANKLE", "RIGHT_ANKLE",
        "LEFT_HEEL", "RIGHT_HEEL",
        "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX"
    ]
    BODY25_TO_MEDIAPIPE = [
        0,    # NOSE
        -1,   # LEFT_EYE_INNER
        16,   # LEFT_EYE
        -1,   # LEFT_EYE_OUTER
        -1,   # RIGHT_EYE_INNER
        15,   # RIGHT_EYE
        -1,   # RIGHT_EYE_OUTER
        18,   # LEFT_EAR
        17,   # RIGHT_EAR
        -1,   # MOUTH_LEFT
        -1,   # MOUTH_RIGHT
        5,    # LEFT_SHOULDER
        2,    # RIGHT_SHOULDER
        6,    # LEFT_ELBOW
        3,    # RIGHT_ELBOW
        7,    # LEFT_WRIST
        4,    # RIGHT_WRIST
        -1,   # LEFT_PINKY (from hand)
        -1,   # RIGHT_PINKY (from hand)
        -1,   # LEFT_INDEX (from hand)
        -1,   # RIGHT_INDEX (from hand)
        -1,   # LEFT_THUMB (from hand)
        -1,   # RIGHT_THUMB (from hand)
        12,   # LEFT_HIP
        9,    # RIGHT_HIP
        13,   # LEFT_KNEE
        10,   # RIGHT_KNEE
        14,   # LEFT_ANKLE
        11,   # RIGHT_ANKLE
        21,   # LEFT_HEEL
        24,   # RIGHT_HEEL
        19,   # LEFT_FOOT_INDEX (big toe)
        22    # RIGHT_FOOT_INDEX (big toe)
    ]
    hand_indices = {
        "LEFT_PINKY": 20,
        "RIGHT_PINKY": 20,
        "LEFT_INDEX": 8,
        "RIGHT_INDEX": 8,
        "LEFT_THUMB": 4,
        "RIGHT_THUMB": 4
    }

    H, W = frame.shape[:2]
    H =1
    W = 1
    candidate, subset = body_estimation(frame)

    # --- Detect hands and assign to persons ---
    all_hand_peaks = [{} for _ in range(len(subset))]
    if len(subset) > 0:
        hands_list = util.handDetect(candidate, subset, frame)
        for hand_idx, (x, y, w, is_left) in enumerate(hands_list):
            peaks = hand_estimation(frame[y:y+w, x:x+w, :])
            peaks[:, 0] = np.where(peaks[:, 0]==0, peaks[:, 0], peaks[:, 0]+x)
            peaks[:, 1] = np.where(peaks[:, 1]==0, peaks[:, 1], peaks[:, 1]+y)
            # Assign hand to closest person (by wrist)
            min_dist = float('inf')
            assigned_person = None
            wrist_kpt_idx = 7 if is_left else 4  # BODY25: 7=L_WRIST, 4=R_WRIST
            for person_idx, person in enumerate(subset):
                idx = int(person[wrist_kpt_idx])
                if idx == -1 or idx >= candidate.shape[0]:
                    continue
                wx, wy, _, _ = candidate[idx]
                hand_cx, hand_cy = x + w/2, y + w/2
                dist = np.hypot(wx - hand_cx, wy - hand_cy)
                if dist < min_dist:
                    min_dist = dist
                    assigned_person = person_idx
            if assigned_person is not None:
                side = "left" if is_left else "right"
                all_hand_peaks[assigned_person][side] = peaks

    # --- Build MediaPipe-style output ---
    results = {}
    for person_idx, person in enumerate(subset):
        person_dict = {}
        for i, name in enumerate(POSE_LANDMARK_NAMES):
            idx = BODY25_TO_MEDIAPIPE[i]
            if idx != -1 and int(person[idx]) != -1 and int(person[idx]) < candidate.shape[0]:
                x, y, conf, _ = candidate[int(person[idx])]

                person_dict[name] = {
                    "x": float(x) / W,
                    "y": float(y) / H,
                    "confidence": float(conf)
                }

            else:
                # Try to fill hand points if available
                filled = False
                if name.startswith("LEFT_") and all_hand_peaks[person_idx].get("left") is not None:
                    hand_peaks = all_hand_peaks[person_idx]["left"]
                    if name in hand_indices:
                        hi = hand_indices[name]
                        if hi < len(hand_peaks):
                            if len(hand_peaks[hi]) == 3:
                                x, y, conf = hand_peaks[hi]
                            elif len(hand_peaks[hi]) == 2:
                                x, y = hand_peaks[hi]
                                conf = 0.0
                            else:
                                x, y, conf = 0.0, 0.0, 0.0
                        else:
                            x, y, conf = 0.0, 0.0, 0.0
                        person_dict[name] = {
                            "x": float(x) / W,
                            "y": float(y) / H,
                            "confidence": float(conf)
                        }
                        filled = True
                if not filled and name.startswith("RIGHT_") and all_hand_peaks[person_idx].get("right") is not None:
                    hand_peaks = all_hand_peaks[person_idx]["right"]
                    if name in hand_indices:
                        hi = hand_indices[name]
                        if hi < len(hand_peaks):
                            if len(hand_peaks[hi]) == 3:
                                x, y, conf = hand_peaks[hi]
                            elif len(hand_peaks[hi]) == 2:
                                x, y = hand_peaks[hi]
                                conf = 0.0
                            else:
                                x, y, conf = 0.0, 0.0, 0.0
                        else:
                            x, y, conf = 0.0, 0.0, 0.0
                        person_dict[name] = {
                            "x": float(x) / W,
                            "y": float(y) / H,
                            "confidence": float(conf)
                        }
                        filled = True
                if not filled:
                    person_dict[name] = {"x": 0.0, "y": 0.0, "confidence": 0.0}

        results[person_idx] = person_dict
    canvas = frame.copy()
    if len(candidate) > 0 and len(subset) > 0:
        canvas = util.draw_bodypose(canvas, candidate, subset, model_type)
    # Prepare hand peaks for drawing
    hand_peaks_list = []
    for hand_dict in all_hand_peaks:
        for side in hand_dict:
            hand_peaks_list.append(hand_dict[side])
    #if hand_peaks_list:
        #canvas = util.draw_handpose(canvas, hand_peaks_list)    

    #return person_dict,canvas
    return results[0] if results else {}, canvas



def process_frame(frame_id, frame, mediapipe_style_pose, width, height):
    start = time.time()
    poses,canvas = mediapipe_style_pose(frame)

    elapsed = time.time() - start
    return {
        "frame": frame_id,
        "poses": poses,
        "drawn_frame": canvas,
        "time": elapsed
    }



def get_pose_list_from_video(video_path, max_frames=None, threads=12, output_json=None):
    start_wall = datetime.now()
    start_time = time.time()
    print(f"Process started at: {start_wall.strftime('%Y-%m-%d %H:%M:%S')}")

    # CPU optimization
    torch.set_num_threads(threads)
    cv2.setNumThreads(threads)
    cv2.useOptimized()

    cap = cv2.VideoCapture(video_path)

    #for adding layer of filter


    pose_results = []
    frame_id = 0
    futures = []

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if not cap.isOpened():
        print("Error: Could not open video.")
        return [], []

    executor = ThreadPoolExecutor(max_workers=threads)
    print("[INFO] Processing video...")

    while True:
        ret, frame = cap.read()
        if not ret or (max_frames and frame_id >= max_frames):
            break
        futures.append(executor.submit(process_frame, frame_id, frame, mediapipe_style_pose, width, height))
        frame_id += 1

    pose_list = []
    result_coordinates = []

    for future in as_completed(futures):
        result = future.result()
        print(f"[INFO] Processed frame {result['frame']} in {result['time']:.3f}s")
        pose_list.append(result['drawn_frame'])
        pose_results.append({
            "frame": result["frame"],
            "poses": result["poses"]
        })
        result_coordinates.append(result["poses"])

    executor.shutdown(wait=True)
    cap.release()

    # Optionally save to JSON
    if output_json:
        with open(output_json, "w") as f:
            json.dump(sorted(pose_results, key=lambda x: x["frame"]), f, indent=2)
        print(f"[DONE] Pose data saved to: {output_json}")

    end_wall = datetime.now()
    end_time = time.time()
    print(f"Process completed at: {end_wall.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Elapsed time: {end_time - start_time:.2f} seconds")
    return pose_list, result_coordinates



import math


# --------------- RunPod Serverless Imports ---------------
# ---------------- Logging Configuration ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


# ---------------- Model Initialization ----------------
GROUNDING_DINO_CHECKPOINT = 'groundingdino_swint_ogc.pth'
GROUNDING_DINO_CONFIG = "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
try:
    grounding_model = load_model(
        model_config_path=GROUNDING_DINO_CONFIG,
        model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
        device=DEVICE
    )
    logger.info("Grounding DINO model loaded successfully.")
except Exception as e:
    logger.error("Failed to load Grounding DINO model: %s", str(e))
    raise

try:
    sam2_checkpoint = "sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, vos_optimized=False)
    sam2_image_model = build_sam2(model_cfg, sam2_checkpoint)
    image_predictor = SAM2ImagePredictor(sam2_image_model)
    logger.info("SAM2 models loaded successfully.")
except Exception as e:
    logger.error("Failed to load SAM2 models: %s", str(e))
    raise

# ---------------- Mediapipe Setup ----------------
try:
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose_detector = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    logger.info("Mediapipe pose detector initialized.")
except Exception as e:
    logger.error("Failed to initialize Mediapipe: %s", str(e))
    raise


schema = {
    "input_video": {
        "type": str,
        "required": True,
    },
    "shot_type":{
        "type": str,
        "required": True,
    },
    "skip_frames":{
        "type": bool,
        "required": False,
        "default": True,
    },
    "firestore_collection":{
        "type": str,
        "required": False,
        "default": "videos",
    },
    "firebase_cred":{
        "type": str,
        "required": False,
        "default": "",
    },
    "firebase_bucket":{
        "type": str,
        "required": False,
        "default": "",
    },
    "firestore_doc_id":{
        "type": str,
        "required": False,
        "default": "",
    },
    "firebase_storage_location":{
        "type": str,
        "required": False,
        "default": "",
    }
}

def process_video(input_video: str,skip_frames: True,shot_type: str,test = False,firestore_collection: str = "videos",firebase_cred: str  = "",firebase_bucket: str="",firestore_doc_id: str="",firebase_storage_location: str=""):
    """
    Main logic that processes the input video, runs GroundingDINO + SAM2,
    and optionally integrates with Firebase if `input_video` is a gs:// URL
    or if Firebase environment variables are set.
    """

    # ---------------- Unique Request Directory ----------------
    request_id = str(uuid.uuid4())
    temp_dir = f"runpod_{request_id}"
    frames_dir = os.path.join(temp_dir, "custom_video_frames")
    tracking_dir = os.path.join(temp_dir, "tracking_results")
    video_results = os.path.join(temp_dir,"results")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(tracking_dir, exist_ok=True)
    os.makedirs(video_results, exist_ok=True)


    # ---------------- Firebase Initialization ----------------
    # Determine if Firebase integration should be used.


        # If the input is a Firebase Storage URL, download it locally.

    local_video_path = input_video
    get_pose_list,result_coordinates = get_pose_list_from_video(local_video_path)
    print(len(get_pose_list))
    import csv




    # ---------------- Validate Local Video ----------------
    if not os.path.exists(local_video_path):
        raise FileNotFoundError(f"Input video file not found: {local_video_path}")
    

    # ---------------- Hyperparameters and File Paths ----------------
    TEXT_PROMPT = 'golf ball. golf club. golf player. human hand fingers.'
    BOX_THRESHOLD = 0.35
    TEXT_THRESHOLD = 0.25

    SKIP = skip_frames
    SHOT_TYPE = shot_type

    

    CLUB_REF_AXIS = "-y"
    HEAD_REF_AXIS = "-y"

    SAVE_TRACKING_RESULTS_DIR = "./tracking_results"
    
    OUTPUT_VIDEO_PATH = f"{video_results}/output.mp4"
    JSON_OUTPUT_PATH = f"{video_results}/results.json"

    PROMPT_TYPE_FOR_VIDEO = "box"  # choose from ["point", "box", "mask"]

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------------- Read and Save Video Frames ----------------
    try:
        video_info = sv.VideoInfo.from_video_path(local_video_path)
        logger.info(f"Video Info: {video_info}")
        frame_generator = sv.get_video_frames_generator(local_video_path, stride=1, start=0, end=None)
    except Exception as e:
        logger.error("Error reading video frames: %s", str(e))
        raise


    try:
        with sv.ImageSink(
            target_dir_path=frames_dir,
            overwrite=True,
            image_name_pattern="{:05d}.jpg"
        ) as sink:
            for frame in tqdm(frame_generator, desc="Saving Video Frames"):
                sink.save_image(frame)
        logger.info("Video frames saved to %s", frames_dir)
    except Exception as e:
        logger.error("Error saving video frames: %s", str(e))
        raise

    # List the saved frame files.
    frame_names = sorted([
        p for p in os.listdir(frames_dir)
        if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]
    ], key=lambda p: int(os.path.splitext(p)[0]))

    if not frame_names:
        raise RuntimeError("No video frames found after processing.")

    # ---------------- Initialize SAM2 Video Predictor ----------------
    try:
        inference_state = video_predictor.init_state(video_path=frames_dir)
    except Exception as e:
        logger.error("Failed to initialize video predictor state: %s", str(e))
        raise

    ann_frame_idx = 0  # the frame index we interact with

    # ---------------- Step 2: Prompt Grounding DINO on a Reference Frame ----------------
    try:
        img_path = os.path.join(frames_dir, frame_names[ann_frame_idx])
        image_source, image = load_image(img_path)
        boxes, confidences, labels = predict(
            model=grounding_model,
            image=image,
            caption=TEXT_PROMPT,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
        )
        boxes, confidences, labels = filter_detections_by_label(boxes, confidences, labels)
        h, w, _ = image_source.shape
        boxes = boxes * torch.Tensor([w, h, w, h])
        input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        logger.info("Grounding DINO predictions complete.")
    except Exception as e:
        logger.error("Error during Grounding DINO prediction: %s", str(e))
        raise

    try:
        image_predictor.set_image(image_source)
        OBJECTS = labels

        # Optional: Mixed precision if a suitable GPU is available
        #torch.autocast(device_type=DEVICE, dtype=torch.bfloat16).__enter__()
        torch.autocast(device_type=DEVICE, dtype=torch.float16).__enter__()
        if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        masks, scores, logits = image_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )
        if masks.ndim == 4:
            masks = masks.squeeze(1)
    except Exception as e:
        logger.error("Error during SAM2 image prediction: %s", str(e))
        raise

    # ---------------- Step 3: Register object prompts in SAM2 video predictor ----------------
    try:
        assert PROMPT_TYPE_FOR_VIDEO in ["point", "box", "mask"], "Unsupported prompt type."

        if PROMPT_TYPE_FOR_VIDEO == "point":
            # Sample points from each mask and register them
            all_sample_points = sample_points_from_masks(masks=masks, num_points=10)
            for object_id, (label, points) in enumerate(zip(OBJECTS, all_sample_points), start=1):
                labels_arr = np.ones((points.shape[0]), dtype=np.int32)
                video_predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=object_id,
                    points=points,
                    labels=labels_arr,
                )

        elif PROMPT_TYPE_FOR_VIDEO == "box":
            # Use bounding boxes as prompts
            for object_id, (label, box) in enumerate(zip(OBJECTS, input_boxes), start=1):
                video_predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=object_id,
                    box=box,
                )

        elif PROMPT_TYPE_FOR_VIDEO == "mask":
            # Use binary masks as prompts
            for object_id, (label, mask) in enumerate(zip(OBJECTS, masks), start=1):
                video_predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=object_id,
                    mask=mask
                )

        logger.info("Object registration with video predictor complete.")
    except Exception as e:
        logger.error("Error during object registration: %s", str(e))
        raise

    # ---------------- Step 4: Propagate segmentation results through the video ----------------
    video_segments = {}
    try:
        with torch.no_grad():
            for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state):
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }
            logger.info("Video segmentation propagation complete.")
    except Exception as e:
        logger.error("Error during video segmentation propagation: %s", str(e))
        raise

    # ---------------- Step 5: Visualize, compute angles, and save frames ----------------

    

    ID_TO_OBJECTS = {i: obj for i, obj in enumerate(OBJECTS, start=1)}
    angles_club = []
    angles_head = []
    frame_indices = []
    annotated_frames = {}
    results = []  # for storing data per frame
    analysis = {}
    for frame_idx, segments in video_segments.items():
        try:
            img_path = os.path.join(frames_dir,frame_names[frame_idx])
            img = cv2.imread(img_path)
            



            object_ids = list(segments.keys())
            masks_list = list(segments.values())
            # Combine all masks for detection bounding boxes
            masks_concat = np.concatenate(masks_list, axis=0)

            detections = sv.Detections(
                xyxy=sv.mask_to_xyxy(masks_concat),
                mask=masks_concat,
                class_id=np.array(object_ids, dtype=np.int32),
            )
            #annotated_frame = img.copy()
            #annotated_frame, pose_points = player_pose1(annotated_frame, mp_drawing, pose_detector, mp_pose)

            annotated_frame = get_pose_list[int(frame_idx)]
      
            obb_centers = {item: None for item in OBJECTS}

            for obj_id, mask in zip(object_ids, masks_concat):
                mask_binary = mask.astype(np.uint8)
                obb_box, rect = get_rotated_bounding_box(mask_binary)
                if rect is not None:
                    label = ID_TO_OBJECTS[obj_id]
                    if label in obb_centers:
                        center = rect[0]
                        obb_centers[label] = (center, obb_box, mask_binary)

            golf_club_head_box = None
            golf_club_shaft_box = None
            player_center = None
            golf_club_head_center = None
            angle_club = None
            angle_head = None

            # If we have all relevant objects
            if (
                obb_centers.get("golf club")
                and obb_centers.get("human hand fingers")
                and obb_centers.get("golf player")
            ):
                hand_center = obb_centers["human hand fingers"][0]
                golf_club_data = obb_centers["golf club"]
                golf_club_center = golf_club_data[0]
                golf_club_box = golf_club_data[1]
                golf_club_mask = golf_club_data[2]

                # Identify the "head" of the club as the farthest point from the hand
                golf_head_point = max(
                    golf_club_box,
                    key=lambda point: np.linalg.norm(np.array(point) - np.array(hand_center))
                )

                (golf_club_head_box,
                 golf_club_shaft_box,
                 golf_club_head_center,
                 golf_club_shaft_center,
                 obb_angle) = split_oriented_bounding_box(golf_club_box, golf_head_point, 0.1)

                cv2.drawContours(annotated_frame, [golf_club_head_box], 0, (0, 0, 255), 2)
                cv2.drawContours(annotated_frame, [golf_club_shaft_box], 0, (0, 255, 0), 2)

                # Fit a line along the shaft region
                golf_club_shaft_p1, golf_club_shaft_p2 = fit_line_to_segmentation(
                    golf_club_mask, golf_club_shaft_box
                )

                if golf_club_shaft_p1 and golf_club_shaft_p2:
                    closest_club_point, farthest_club_point = find_closest_point_to_hand(
                        (golf_club_shaft_p1, golf_club_shaft_p2), hand_center
                    )
                    cv2.circle(annotated_frame, (int(closest_club_point[0]), int(closest_club_point[1])), 5, (0, 0, 255), -1)
                    cv2.circle(annotated_frame, (int(farthest_club_point[0]), int(farthest_club_point[1])), 5, (0, 0, 255), -1)

                    if SHOT_TYPE == 'back':
                        annotated_frame, ref, angle_club = draw_reference_axis_and_calculate_angle_for_back_shot(
                            annotated_frame, farthest_club_point, closest_club_point
                        )
                    else:
                        annotated_frame, ref, angle_club = draw_reference_axis_and_calculate_angle_for_side_shot(
                            annotated_frame,
                            farthest_club_point,
                            closest_club_point,
                            axis=CLUB_REF_AXIS,
                            angle_type="club_angle"
                        )
                    angles_club.append(angle_club)

                if SHOT_TYPE != 'back':
                    # Also measure the angle between the player's center and the club head
                    player_mask = obb_centers["golf player"][2]
                    player_center = get_centroid_from_mask(player_mask)

                    annotated_frame, ref, angle_head = draw_reference_axis_and_calculate_angle_for_side_shot(
                        annotated_frame,
                        golf_club_head_center,
                        player_center,
                        axis=HEAD_REF_AXIS,
                        angle_type="head_angle"
                    )
                    angles_head.append(angle_head)

                frame_indices.append(frame_idx)

            annotated_frames[frame_idx] = annotated_frame

            # Store results for JSON
            results.append({
                "frame": frame_idx,
                "shot_type": SHOT_TYPE,
                "club_angle": angle_club,
                "head_angle": angle_head,
                "golf_club_shaft_obb": golf_club_shaft_box,
                "golf_club_head_obb": golf_club_head_box,
                "pose_points": result_coordinates[int(frame_idx)]
                
            })
            analysis[frame_idx] = result_coordinates[int(frame_idx)]


        except Exception as e:
            logger.error("Error processing frame %s: %s", frame_idx, str(e))
            continue
    

    # ---------------- Remove Anomalous Frames Based on Angles ----------------
    # try:
    #     anomalous_frames_indices, _ = frame_skipping_on_angle_anomaly(frame_indices, angles_club, skip=SKIP)

    #     if SHOT_TYPE != 'back':
    #         anomalous_frames_indices2, _ = frame_skipping_on_angle_anomaly(frame_indices, angles_head, skip=SKIP)
    #         anomalous_frames_indices = anomalous_frames_indices.union(anomalous_frames_indices2)

    # except Exception as e:
    #     logger.error("Error during frame skipping: %s", str(e))
    #     raise

    try:
        if len(angles_club) >= 2:
            anomalous_frames_indices, _ = frame_skipping_on_angle_anomaly(
                frame_indices, angles_club, skip=SKIP
            )
        else:
            # Club not detected — consider notifying the backend via DB flag
            anomalous_frames_indices = frame_indices

        if SHOT_TYPE != 'back':
            if len(angles_head) >= 2:
                anomalous_frames_indices2, _ = frame_skipping_on_angle_anomaly(
                    frame_indices, angles_head, skip=SKIP
                )
                anomalous_frames_indices = anomalous_frames_indices.union(anomalous_frames_indices2)

    except Exception as e:
        logger.error("Error during frame skipping: %s", str(e))
        raise


    # Save annotated frames (skip anomalous ones)
    try:
        for frame_idx in sorted(annotated_frames.keys()):
            if frame_idx in anomalous_frames_indices:
                continue
            annotated_frame = annotated_frames[frame_idx]
            cv2.putText(
                annotated_frame,
                f"Frame: {frame_idx}",
                (annotated_frame.shape[1] - 150, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2
            )
            cv2.imwrite(
                os.path.join(tracking_dir, f"annotated_frame_{frame_idx:05d}.jpg"),
                annotated_frame
            )
    except Exception as e:
        logger.error("Error saving annotated frames: %s", str(e))
        raise

    # Write final JSON
    try:
        filtered_results = [
            convert_ndarray_to_list(entry)
            for entry in results
            if entry["frame"] not in anomalous_frames_indices
        ]
        with open(JSON_OUTPUT_PATH, 'w') as json_file:
            json.dump(filtered_results, json_file, indent=4)
        logger.info("JSON output saved to %s", JSON_OUTPUT_PATH)


    except Exception as e:
        logger.error("Error writing JSON output: %s", str(e))
        raise

    # Create annotated video
    try:
        create_video_from_images(tracking_dir, OUTPUT_VIDEO_PATH)
        logger.info("Annotated video created at %s", OUTPUT_VIDEO_PATH)
    except Exception as e:
        logger.error("Error creating output video: %s", str(e))
        raise
    if test:
        return filtered_results
    # ---------------- Firebase: Upload Processed Files and Update Firestore ----------------
    uploaded_files = {}
    use_firebase = False
    if use_firebase and output_folder:
        try:
            # Upload JSON output.
            json_dest = f"{output_folder}/results.json"
            json_blob = bucket.blob(json_dest)
            json_blob.upload_from_filename(JSON_OUTPUT_PATH)
            uploaded_files["json_url"] = json_blob.public_url

            # Upload annotated video.
            video_dest = f"{output_folder}/annotated_video.mp4"
            video_blob = bucket.blob(video_dest)
            video_blob.upload_from_filename(OUTPUT_VIDEO_PATH)
            uploaded_files["video_url"] = video_blob.public_url

            # Zip the annotated frames directory.
            frames_zip_local = f"{video_results}/annotated_frames.zip"
            zip_folder(tracking_dir, frames_zip_local)
            frames_dest = f"{output_folder}/annotated_frames.zip"
            frames_blob = bucket.blob(frames_dest)
            frames_blob.upload_from_filename(frames_zip_local)
            uploaded_files["frames_zip_url"] = frames_blob.public_url

            # Update Firestore
            if firestore_doc_id:
                doc_id = firestore_doc_id
            else:
                doc_id = os.path.splitext(os.path.basename(output_folder))[0]
            doc_ref = db.collection(firestore_collection).document(doc_id)
            doc_ref.set({
                "processed": True,
                "results_json": uploaded_files["json_url"],
                "annotated_video": uploaded_files["video_url"],
                "annotated_frames_zip": uploaded_files["frames_zip_url"],
                "original_video": input_video
            }, merge=True)

            logger.info("Firestore document '%s' updated.", doc_id)
            uploaded_files["processed"] = True

        except Exception as e:
            logger.error("Error during Firebase upload or Firestore update: %s", str(e))
            uploaded_files["processed"] = False
            raise
#    clean(folder_list=[temp_dir])        
    # Return local or uploaded file paths for convenience
    return OUTPUT_VIDEO_PATH,JSON_OUTPUT_PATH,analysis,tracking_dir
    # return {
    #     "uploaded_files": uploaded_files
    # }



# ------------------ RUNPOD HANDLER ------------------
def handler(event):
    """
    The handler function that RunPod will invoke. It receives an event (JSON input),
    parses required fields, and returns a JSON response.
    """
    # Example of required/optional inputs from the event. Adjust as needed.
    try:
        validated_input = validate(event["input"], schema)
        
        if "errors" in validated_input:
            return {"error": validated_input["errors"]}
        # Required fields (you can expand or add defaults as needed)
        input = validated_input['validated_input']
        input_video = input.get("input_video")  # Local path or gs://
        shot_type = input.get("shot_type", "side")  # 'side' or 'back'
        # Optional or default fields
        skip_frames = input.get("skip_frames", True)
        firestore_collection = input.get("firestore_collection", "videos")
        firebase_cred = input.get("firebase_cred", "")
        firebase_bucket = input.get("firebase_bucket", "")
        firestore_doc_id = input.get("firestore_doc_id", "")
        firebase_storage_loc = input.get("firebase_storage_location", "")


        if not input_video:
            return {
                "error": "Missing 'input_video'. Provide a local path or a gs:// URL."
            }
        try:
            result = process_video(
                input_video=input_video,
                skip_frames=skip_frames,
                shot_type=shot_type,
                firestore_collection=firestore_collection,
                firebase_cred=firebase_cred,
                firebase_bucket=firebase_bucket,
                firestore_doc_id=firestore_doc_id,
                firebase_storage_location=firebase_storage_loc
            )

            return {
                "status": "success",
                "detail": result
            }

        except Exception as e:
            logger.exception(f"Processing failed: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    except Exception as e:
        return {"error": str(e)}


# ------------------ RUNPOD SERVERLESS ENTRYPOINT ------------------
# if __name__ == '__main__':
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--input_video', required=True)
#     parser.add_argument('--shot_type', required=True)
#     parser.add_argument('--skip_frames', action='store_true')
#     args = parser.parse_args()
#     process_video(
#         input_video=args.input_video,
#         skip_frames=args.skip_frames,
#         shot_type=args.shot_type
#     )

# if __name__ == '__main__':
#     # Change these as needed
#     input_video = "test_1.mp4"
#     shot_type = "side"
#     skip_frames = True  # or False, as needed

#     process_video(
#         input_video=input_video,
#         skip_frames=skip_frames,
#         shot_type=shot_type
#     )

import time
from datetime import datetime

import numpy as np







if __name__ == '__main__':
    # Record the start time
    start_wall = datetime.now()
    start_time = time.time()
    print(f"Process started at: {start_wall.strftime('%Y-%m-%d %H:%M:%S')}")

    # Change these as needed
    input_video = "may_swing_side_2.MOV"
    shot_type = "side"
    skip_frames = True  # or False, as needed

    OUTPUT_VIDEO_PATH,JSON_OUTPUT_PATH,side_body_analysis,tracking_dir =process_video(
        input_video=input_video,
        skip_frames=skip_frames,
        shot_type=shot_type
    )

    # Record the completion time
    end_wall = datetime.now()
    end_time = time.time()
    print(f"Process completed at: {end_wall.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Elapsed time: {end_time - start_time:.2f} seconds")
    start_wall = datetime.now()
    start_time = time.time()
    print(f"Process started at: {start_wall.strftime('%Y-%m-%d %H:%M:%S')}")

    # Change these as needed
    input_video = "may_swing_front_2.MOV"
    shot_type = "front"
    skip_frames = True  # or False, as needed

    OUTPUT_VIDEO_PATH,JSON_OUTPUT_PATH,front_body_analysis,tracking_dir =process_video(
        input_video=input_video,
        skip_frames=skip_frames,
        shot_type=shot_type,  
    )

    # Record the completion time
    end_wall = datetime.now()
    end_time = time.time()
    print(f"Process completed at: {end_wall.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Elapsed time: {end_time - start_time:.2f} seconds")


    import json
    with open("body_landmarks_side.json", "w") as f:
        json.dump(convert_to_python_types(side_body_analysis), f, indent=2)

    with open("body_landmarks_front.json", "w") as f:
        json.dump(convert_to_python_types(front_body_analysis), f, indent=2) 


#print(analyze_golf_swing(side_body_analysis,front_body_analysis))