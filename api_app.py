from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
import os
import shutil
import uuid
import numpy as np
# Import process_video from main.py (ensure main.py is in the same directory or PYTHONPATH)
from main import process_video

app = FastAPI()

UPLOAD_DIR = "video_results"
os.makedirs(UPLOAD_DIR, exist_ok=True)

import math
from collections import defaultdict



def average_confidence_by_keypoint_rounded(frames):
    keypoint_sums = defaultdict(float)
    keypoint_counts = defaultdict(int)

    for frame in frames:
        for keypoint, data in frame["pose_points"].items():
            keypoint_sums[keypoint] += data["confidence"]
            keypoint_counts[keypoint] += 1

    keypoint_avg_conf = {
        keypoint: round(keypoint_sums[keypoint] / keypoint_counts[keypoint], 4)
        for keypoint in keypoint_sums
    }

    # Apply ground floor to nearest 0.1
    keypoint_avg_conf_grounded = {
        k: math.floor(v * 10) / 10 for k, v in keypoint_avg_conf.items()
    }

    return keypoint_avg_conf_grounded

def print_keypoint_confidences(avg_conf):
    print(f"{'Keypoint':<20} {'Avg Confidence'}")
    for keypoint in sorted(avg_conf.keys()):
        print(f"{keypoint:<20} {avg_conf[keypoint]:.1f}")

def is_all_points_greater_or_equal(front_dict: dict, side_dict: dict) -> bool:
    front_dictionary = {

 'LEFT_SHOULDER': 0.1,
 'RIGHT_SHOULDER': 0.1,
 'LEFT_ELBOW': 0.1,
 'RIGHT_ELBOW': 0.1,
 'LEFT_WRIST': 0.1,
 'RIGHT_WRIST': 0.1,
 'LEFT_PINKY': 0.0,
 'RIGHT_PINKY': 0.0,
 'LEFT_INDEX': 0.0,
 'RIGHT_INDEX': 0.0,
 'LEFT_THUMB': 0.0,
 'RIGHT_THUMB': 0.0,
 'LEFT_HIP': 0.1,
 'RIGHT_HIP': 0.1,
 'LEFT_KNEE': 0.1,
 'RIGHT_KNEE': 0.1,
 'LEFT_ANKLE': 0.1,
 'RIGHT_ANKLE': 0.1,
 'LEFT_HEEL': 0.1,
 'RIGHT_HEEL': 0.1,
 'LEFT_FOOT_INDEX': 0.1,
 'RIGHT_FOOT_INDEX': 0.1} 

    side_dictionary = {
 'LEFT_SHOULDER': 0.1,
 'RIGHT_SHOULDER': 0.1,
 'LEFT_ELBOW': 0.1,
 'RIGHT_ELBOW': 0.1,
 'LEFT_WRIST': 0.1,
 'RIGHT_WRIST': 0.1,
 'LEFT_PINKY': 0.0,
 'RIGHT_PINKY': 0.0,
 'LEFT_INDEX': 0.0,
 'RIGHT_INDEX': 0.0,
 'LEFT_THUMB': 0.0,
 'RIGHT_THUMB': 0.0,
 'LEFT_HIP': 0.1,
 'RIGHT_HIP': 0.1,
 'LEFT_KNEE': 0.1,
 'RIGHT_KNEE': 0.1,
 'LEFT_ANKLE': 0.1,
 'RIGHT_ANKLE': 0.1,
 'LEFT_HEEL': 0.1,
 'RIGHT_HEEL': 0.1,
 'LEFT_FOOT_INDEX': 0.1,
 'RIGHT_FOOT_INDEX': 0.1}
    if front_dict:

        for key in front_dictionary:
            ref_val = front_dictionary.get(key, 0)
            input_val = front_dict.get(key, 0)
            if input_val < ref_val:
                print('front',key,input_val,ref_val)
                print(input_val,ref_val)
                return False
    if side_dict:
        for key in side_dictionary:
            ref_val = side_dictionary.get(key, 0)
            input_val = side_dict.get(key, 0)
            if input_val < ref_val:

                print('side',key,input_val,ref_val)
                return False

    return True
# Example usage:
def verify_frame(front,side):
    avg_conf_ground_front = None
    avg_conf_ground_side = None

    if front:
        avg_conf_ground_front = average_confidence_by_keypoint_rounded(front)

    if side:
        avg_conf_ground_side = average_confidence_by_keypoint_rounded(side)


  # print_keypoint_confidences(avg_conf_ground)
    respons = is_all_points_greater_or_equal(avg_conf_ground_front,avg_conf_ground_side)
    return respons
   









import cv2
import os
import shutil
import uuid
import zipfile
import json

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse

from main import process_video  # Now returns 3 values
from analysis_golf import analyze_golf_swing,analyze_golf_swing_front,analyze_golf_swing_side

app = FastAPI()

UPLOAD_DIR = "video_results"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def get_frame_count(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count
    
def add_directory_to_zip(zipf, folder_path, zip_folder_name):
    """
    Adds an entire folder to a zip file, preserving subdirectory structure.

    :param zipf: ZipFile object
    :param folder_path: Local path to the folder to add
    :param zip_folder_name: Folder name inside the zip archive
    """
    for root, _, files in os.walk(folder_path):
        for file in files:
            abs_path = os.path.join(root, file)
            # Calculate relative path for zip
            rel_path = os.path.relpath(abs_path, folder_path)
            arcname = os.path.join(zip_folder_name, rel_path)
            zipf.write(abs_path, arcname=arcname)



def test_video_quality(video_path):
    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Generate output path by adding _test before extension
    base, ext = os.path.splitext(video_path)
    output_path = f"{base}_test{ext}"

    # Calculate 10 evenly spaced frame indices
    num_samples = 20
    frame_indices = np.linspace(0, total_frames - 1, num=num_samples, dtype=int)

    print("Frames to capture:", frame_indices)

    frame_id = 0
    saved_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id in frame_indices:
            saved_frames.append(frame)
            print(f"Captured frame {frame_id}")
            if len(saved_frames) == num_samples:
                break

        frame_id += 1

    cap.release()

    # Save the sampled frames as a video
    if saved_frames:
        out = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*'mp4v'),  # Codec
            fps,
            (width, height)
        )

        for frame in saved_frames:
            out.write(frame)

        out.release()
        print(f"Saved sampled video to {output_path}")
        return output_path
    else:
        print("No frames were captured.")
        return None



                    

def process_and_zip(video1_path, video2_path, user_dir):
    # Fixed shot types
    shot_type1 = "front"
    shot_type2 = "side"

    # analysis1_test = process_video(test_video_quality(video1_path), skip_frames=True, shot_type=shot_type1,test = True)
    # analysis2_test = process_video(test_video_quality(video2_path), skip_frames=True, shot_type=shot_type2,test = True)
    
    # if not verify_frame(analysis1_test, analysis2_test):
    #     raise HTTPException(status_code=400, detail="Landmark confidence too low. Please upload clearer videos.")
   
    
     
    # Process both videos
    if video1_path and video2_path:
        output1, json1, analysis1, annoted_frames_dir_1 = process_video(video1_path, skip_frames=True, shot_type=shot_type1,test = False)
        output2, json2, analysis2, annoted_frames_dir_2 = process_video(video2_path, skip_frames=True, shot_type=shot_type2,test = False)




        # Run swing analysis combining both
        result_analysis = analyze_golf_swing(analysis2, analysis1)


        result_analysis_path = os.path.join(user_dir, "result_analysis.json")
        with open(result_analysis_path, "w") as f:
            json.dump(result_analysis, f, indent=2)

        # Create ZIP with all outputs
        zip_path = os.path.join(user_dir, "results.zip")
        with zipfile.ZipFile(zip_path, "w") as zipf:
            zipf.write(output1, arcname="output_front.mp4")
            zipf.write(json1, arcname="results_front.json")
            zipf.write(output2, arcname="output_side.mp4")
            zipf.write(json2, arcname="results_side.json")
            zipf.write(result_analysis_path, arcname="result_analysis.json")
            add_directory_to_zip(zipf, annoted_frames_dir_1, "annotated_frames_front")
            add_directory_to_zip(zipf, annoted_frames_dir_2, "annotated_frames_side")
    
    if video1_path and video2_path == None:
        output1, json1, analysis1, annoted_frames_dir_1 = process_video(video1_path, skip_frames=True, shot_type=shot_type1,test = False)



        # Run swing analysis combining both
        result_analysis = analyze_golf_swing_front(analysis1)


        result_analysis_path = os.path.join(user_dir, "result_analysis.json")
        with open(result_analysis_path, "w") as f:
            json.dump(result_analysis, f, indent=2)

        # Create ZIP with all outputs
        zip_path = os.path.join(user_dir, "results.zip")
        with zipfile.ZipFile(zip_path, "w") as zipf:
            zipf.write(output1, arcname="output_front.mp4")
            zipf.write(json1, arcname="results_front.json")
            zipf.write(result_analysis_path, arcname="result_analysis.json")
            add_directory_to_zip(zipf, annoted_frames_dir_1, "annotated_frames_front")

    if video1_path == None and video2_path:
        output2, json2, analysis2, annoted_frames_dir_2 = process_video(video2_path, skip_frames=True, shot_type=shot_type2,test = False)



        # Run swing analysis combining both
        result_analysis = analyze_golf_swing_side(analysis2)


        result_analysis_path = os.path.join(user_dir, "result_analysis.json")
        with open(result_analysis_path, "w") as f:
            json.dump(result_analysis, f, indent=2)

        # Create ZIP with all outputs
        zip_path = os.path.join(user_dir, "results.zip")
        with zipfile.ZipFile(zip_path, "w") as zipf:

            zipf.write(output2, arcname="output_side.mp4")
            zipf.write(json2, arcname="results_side.json")
            zipf.write(result_analysis_path, arcname="result_analysis.json")
            add_directory_to_zip(zipf, annoted_frames_dir_2, "annotated_frames_side")









# @app.post("/upload/")
# async def upload_videos(
#     file1: UploadFile = File(...),  # front view
#     file2: UploadFile = File(...),  # side view
#     background_tasks: BackgroundTasks = None
# ):
#     request_id = str(uuid.uuid4())
#     user_dir = os.path.join(UPLOAD_DIR, request_id)
#     os.makedirs(user_dir, exist_ok=True)

#     input_video1_path = os.path.join(user_dir, f"front_{file1.filename}")
#     input_video2_path = os.path.join(user_dir, f"side_{file2.filename}")
#     zip_base_name = f"{file1.filename.split('.')[0]}_{file2.filename.split('.')[0]}"

#     with open(input_video1_path, "wb") as buffer1:
#         shutil.copyfileobj(file1.file, buffer1)
#     with open(input_video2_path, "wb") as buffer2:
#         shutil.copyfileobj(file2.file, buffer2)

#     # Step 1: Run test video quality + frame confidence check synchronously
#     test_video1_path = test_video_quality(input_video1_path)
#     test_video2_path = test_video_quality(input_video2_path)

#     analysis1_test = process_video(test_video1_path, skip_frames=True, shot_type="front", test=True)
#     analysis2_test = process_video(test_video2_path, skip_frames=True, shot_type="side", test=True)

#     if not verify_frame(analysis1_test, analysis2_test):
#         raise HTTPException(
#             status_code=400,
#             detail={
#                 "error": "Low landmark confidence",
#                 "message": "Landmark confidence too low. Please upload clearer videos with better lighting and visibility."
#             }
#         )

#     # Step 2: Start processing task in background
#     background_tasks.add_task(process_and_zip, input_video1_path, input_video2_path, user_dir)
#     total_frames = get_frame_count(input_video1_path) + get_frame_count(input_video2_path)

#     return {
#         "Time": f"Please check the download URL after {total_frames * 2.25:.1f} seconds",
#         "job_id": request_id,
#         "download_url": f"/download/{request_id}/{zip_base_name}.zip"
#     }




@app.post("/upload/")
async def upload_videos(
    background_tasks: BackgroundTasks,
    file1: UploadFile = File(None),  # front view (optional)
    file2: UploadFile = File(None),  # side view (optional)
):
    if not file1 and not file2:
        raise HTTPException(
            status_code=400,
            detail="At least one video file must be uploaded (file1 or file2)."
        )

    request_id = str(uuid.uuid4())
    user_dir = os.path.join(UPLOAD_DIR, request_id)
    os.makedirs(user_dir, exist_ok=True)

    # Prepare variables for response and background task
    input_video1_path = input_video2_path = None
    zip_base_name = []
    total_frames = 0
    # Handle file1 (front view)
    if file1 and file2 == None:
        input_video1_path = os.path.join(user_dir, f"front_{file1.filename}")
        zip_base_name.append(file1.filename.split('.')[0])
        with open(input_video1_path, "wb") as buffer1:
            shutil.copyfileobj(file1.file, buffer1)
        test_video1_path = test_video_quality(input_video1_path)
        analysis1_test = process_video(test_video1_path, skip_frames=True, shot_type="front", test=True)
        # If only file1 is uploaded, skip verify_frame
        if file2 is None and not verify_frame(front = analysis1_test,side =file2):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Low landmark confidence",
                    "message": "Front video landmark confidence too low. Please upload clearer videos."
                }
            )
        total_frames += get_frame_count(input_video1_path)

    # Handle file2 (side view)
    elif file2 and file1 == None:
        input_video2_path = os.path.join(user_dir, f"side_{file2.filename}")
        zip_base_name.append(file2.filename.split('.')[0])
        with open(input_video2_path, "wb") as buffer2:
            shutil.copyfileobj(file2.file, buffer2)
        test_video2_path = test_video_quality(input_video2_path)
        analysis2_test = process_video(test_video2_path, skip_frames=True, shot_type="side", test=True)
        # If only file2 is uploaded, skip verify_frame
        if file1 is None and not verify_frame(front = file1,side = analysis2_test):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Low landmark confidence",
                    "message": "Side video landmark confidence too low. Please upload clearer videos."
                }
            )
        total_frames += get_frame_count(input_video2_path)

    # If both files are uploaded, verify both frames together
    elif file1 and file2:

        input_video1_path = os.path.join(user_dir, f"front_{file1.filename}")
        input_video2_path = os.path.join(user_dir, f"side_{file2.filename}")
        zip_base_name = f"{file1.filename.split('.')[0]}_{file2.filename.split('.')[0]}"

        with open(input_video1_path, "wb") as buffer1:
            shutil.copyfileobj(file1.file, buffer1)
        with open(input_video2_path, "wb") as buffer2:
            shutil.copyfileobj(file2.file, buffer2)

        # Step 1: Run test video quality + frame confidence check synchronously
        test_video1_path = test_video_quality(input_video1_path)
        test_video2_path = test_video_quality(input_video2_path)

        analysis1_test = process_video(test_video1_path, skip_frames=True, shot_type="front", test=True)
        analysis2_test = process_video(test_video2_path, skip_frames=True, shot_type="side", test=True)
        total_frames += get_frame_count(input_video1_path)
        total_frames += get_frame_count(input_video2_path)
        
        if not verify_frame(analysis1_test, analysis2_test):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Low landmark confidence",
                    "message": "Landmark confidence too low in one or both videos. Please upload clearer videos."
                }
            )

    # Start background processing for available files
    background_tasks.add_task(
        process_and_zip,
        input_video1_path,  # can be None if not uploaded
        input_video2_path,  # can be None if not uploaded
        user_dir)

    zip_base_name_str = "_".join(zip_base_name)
    return {
        "Time": f"Please check the download URL after {total_frames * 2.25:.1f} seconds",
        "job_id": request_id,
        "download_url": f"/download/{request_id}/{zip_base_name_str}.zip"
    }





@app.get("/download/{job_id}/{zipname}")
def download_zip(job_id: str, zipname: str):
    zip_path = os.path.join(UPLOAD_DIR, job_id, "results.zip")
    if not os.path.exists(zip_path):
        raise HTTPException(status_code=404, detail="Result not ready or not found")
    return FileResponse(zip_path, filename=zipname, media_type="application/zip")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)