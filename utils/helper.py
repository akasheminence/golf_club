import cv2
import numpy as np
import math
import torch

import torch
import zipfile
import os

def player_pose1(frame,mp_drawing,pose_detector,mp_pose):
    # Convert ROI to RGB as mediapipe expects.
    roi_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the ROI with mediapipe to get pose landmarks.
    results = pose_detector.process(roi_rgb)

    # If pose landmarks are found, draw them on the ROI.
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
        )

    return frame, results.pose_landmarks.landmark

def player_pose(frame,player_obb,mp_drawing,pose_detector,mp_pose):
    player_obb_box = np.array(player_obb)
    x1, y1 = np.min(player_obb_box, axis=0)
    x2, y2 = np.max(player_obb_box, axis=0)

    # Ensure the coordinates are integers
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
    # Ensure the bounding box is within frame boundaries.
    x1 = max(x1, 0)
    y1 = max(y1, 0)
    x2 = min(x2, frame.shape[1])
    y2 = min(y2, frame.shape[0])

    # Crop the region of interest (ROI) where the golf player is located.
    roi = frame[y1:y2, x1:x2]
    # Convert ROI to RGB as mediapipe expects.
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

    # Process the ROI with mediapipe to get pose landmarks.
    results = pose_detector.process(roi_rgb)

    # If pose landmarks are found, draw them on the ROI.
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            roi,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
        )
    frame[y1:y2, x1:x2] = roi

    return frame, results.pose_landmarks.landmark


def filter_detections_by_label(boxes, confidences, labels):
  
    # Create a dictionary to store the highest confidence detection for each label
    label_map = {}

    for box, confidence, label in zip(boxes, confidences, labels):
        if label not in label_map or confidence > label_map[label]['confidence']:
            label_map[label] = {'box': box, 'confidence': confidence}

    # Extract the filtered results
    filtered_boxes = [item['box'] for item in label_map.values()]
    filtered_confidences = [item['confidence'] for item in label_map.values()]
    filtered_labels = list(label_map.keys())

    # Convert filtered_boxes and filtered_confidences to single tensors
    filtered_boxes = torch.stack(filtered_boxes)
    filtered_confidences = torch.stack(filtered_confidences)

    return filtered_boxes, filtered_confidences, filtered_labels


def find_closest_point_to_hand(line_endpoints, hand_center):

    point1, point2 = line_endpoints
    distance1 = np.linalg.norm(np.array(point1) - np.array(hand_center))
    distance2 = np.linalg.norm(np.array(point2) - np.array(hand_center))

    if distance1 < distance2:
        return point1, point2
    return point2, point1

def fit_line_to_segmentation(mask, box):

    box = np.array(box)
    x_min, y_min = np.min(box, axis=0)
    x_max, y_max = np.max(box, axis=0)

    # Ensure the coordinates are integers
    x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])

    # Crop the mask to the region of interest
    mask_roi = mask[y_min:y_max, x_min:x_max]

    contours, _ = cv2.findContours(mask_roi.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest_contour)  # Get the oriented bounding box
        box_width, box_height = rect[1]

        # Fit a line to the largest contour
        [vx, vy, cx, cy] = cv2.fitLine(largest_contour, cv2.DIST_L2, 0, 0.01, 0.01)

        # Adjust the center of the line to the original image coordinates
        cx += x_min
        cy += y_min

        # Calculate the endpoints of the line based on the box size
        half_diagonal = np.sqrt((box_width / 2) ** 2 + (box_height / 2) ** 2)

        endpoint1_x = int(cx + vx * half_diagonal)
        endpoint1_y = int(cy + vy * half_diagonal)
        endpoint2_x = int(cx - vx * half_diagonal)
        endpoint2_y = int(cy - vy * half_diagonal)

        # Return the endpoints adjusted to the mask
        return (endpoint1_x, endpoint1_y), (endpoint2_x, endpoint2_y)

    return None, None

def split_oriented_bounding_box(obb_rect, corner_point, threshold):

    # Getting closes point from corner point
    # Exclude the corner point itself
    filtered_coords = obb_rect[~np.all(obb_rect == corner_point, axis=1)]
    # Calculate Euclidean distances to all box coordinates
    distances = np.linalg.norm(filtered_coords - corner_point, axis=1)

    # Find the index of the closest point
    closest_index = np.argmin(distances)

    # Retrieve the closest point
    corner_point2 = filtered_coords[closest_index]

    # center point
    center_point = ((corner_point[0]+corner_point2[0])/2,(corner_point[1]+corner_point2[1])/2)

    # getting angular obb
    obb = cv2.minAreaRect(obb_rect)
    center_x, center_y = obb[0]
    width, height = obb[1]
    angle = obb[2]
    
    # Calculate the vector from the center to the corner point
    corner_vector = np.array(center_point) - np.array([center_x, center_y])
    corner_vector_length = np.linalg.norm(corner_vector)
    
    # Calculate the new width/height for the split boxes
    if width > height:  # Split along the width
        new_width_head = width * threshold
        new_width_shaft = width * (1 - threshold)
        new_height_head = height
        new_height_shaft = height
        
        # Determine new centers
        direction_vector = corner_vector / corner_vector_length
        center_head = np.array([center_x, center_y]) + direction_vector * (width / 2 - new_width_head / 2)
        center_shaft = np.array([center_x, center_y]) + direction_vector * (new_width_shaft / 2 - width / 2)
    else:  # Split along the height
        new_width_head = width
        new_width_shaft = width
        new_height_head = height * threshold
        new_height_shaft = height * (1 - threshold)
        
        # Determine new centers
        direction_vector = corner_vector / corner_vector_length
        center_head = np.array([center_x, center_y]) + direction_vector * (height / 2 - new_height_head / 2)
        center_shaft = np.array([center_x, center_y]) + direction_vector * (new_height_shaft / 2 - height / 2)
    
    # Define the two new OBBs
    obb_head = (tuple(center_head),(new_width_head,new_height_head),angle)
    obb_shaft = (tuple(center_shaft),(new_width_shaft,new_height_shaft),angle)
    box_head = np.int32(cv2.boxPoints(obb_head))  # Convert to points
    box_shaft = np.int32(cv2.boxPoints(obb_shaft))  # Convert to points

    
    return box_head, box_shaft, center_head, center_shaft,angle

def draw_reference_axis_and_calculate_angle_for_back_shot(frame,point1, point2 , length=200):
    
    dx, dy = point2[0] - point1[0], point2[1] - point1[1]
    angle = math.degrees(math.atan2(abs(dy), abs(dx)))
    
    #Bottom-right
    if dy < 0 and dx <= 0: 
        axis = "+x"
     
    #Bottom Left
    elif dy <= 0 and dx > 0:
        axis= "-x"

    #Top-Left
    elif dy > 0 and dx >= 0:
        axis= "-x"

    #Top-Right
    elif dy >= 0 and dx < 0:
        axis= "+x"

    # Determine the reference axis endpoint using a dictionary-based approach
    axis_directions = {
        "+x": (length, 0),
        "-x": (-length, 0),
        "+y": (0, -length),
        "-y": (0, length)
    }
    #draw axis
    dx_ref, dy_ref = axis_directions.get(axis, (length, 0))
    ref_end = (int(point2[0] + dx_ref), int(point2[1] + dy_ref))

    # Draw the reference axis
    #cv2.line(frame, (int(point2[0]), int(point2[1])), ref_end, (0, 255, 0), 2)
    

    # Draw the line between points and annotate the angle
    cv2.line(frame, (int(point1[0]), int(point1[1])), (int(point2[0]), int(point2[1])), (255, 0, 0), 2)
    disp_point = (int(point2[0] + 10), int(point2[1] + 20))
    cv2.putText(frame, f"Angle: {angle:.2f} deg", disp_point, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    return frame, ref_end, angle
def draw_reference_axis_and_calculate_angle_for_side_shot(frame,point1, point2 , length=200, axis="+x",angle_type="club_angle"):
    # Determine the reference axis endpoint using a dictionary-based approach
    axis_directions = {
        "+x": (length, 0),
        "-x": (-length, 0),
        "+y": (0, -length),
        "-y": (0, length)
    }
    

    # Calculate the basic angle in degrees
    dx, dy = point2[0] - point1[0], point2[1] - point1[1]
    base_angle = math.degrees(math.atan2(abs(dy), abs(dx)))

   # Define angle transformations for each axis
    axis_adjustments = {
        "+x": lambda dx, dy, angle: (
            360 - angle if dy < 0 and dx <= 0 else
            180 + angle if dy <= 0 and dx > 0 else
            180 - angle if dy > 0 and dx >= 0 else
            angle
        ),
        "-x": lambda dx, dy, angle: (
            180 - angle if dy < 0 and dx <= 0 else
            angle if dy <= 0 and dx > 0 else
            360 - angle if dy > 0 and dx >= 0 else
            180 + angle
        ),
        "-y": lambda dx, dy, angle: (
            90 - angle if dy < 0 and dx <= 0 else
            270 + angle if dy <= 0 and dx > 0 else
            270 - angle if dy > 0 and dx >= 0 else
            90 + angle
        ),
        "+y": lambda dx, dy, angle: (
            270 - angle if dy < 0 and dx <= 0 else
            90 + angle if dy <= 0 and dx > 0 else
            90 - angle if dy > 0 and dx >= 0 else
            270 + angle
        )
    }

    # Get the adjusted angle based on the axis
    angle = axis_adjustments.get(axis, axis_adjustments[axis])(dx, dy, base_angle)
    if angle_type == "head_angle":
        if angle > 0 and angle <180:
            angle = -angle
        else:
            angle = 360-angle
        disp_point = (int(point1[0] + 10), int(point1[1] + 20))
        cv2.putText(frame, f"Angle: {angle:.2f} deg", disp_point, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    else:   
        # Draw the line between points and annotate the angle
        cv2.line(frame, (int(point1[0]), int(point1[1])), (int(point2[0]), int(point2[1])), (255, 0, 0), 2)
        disp_point = (int(point2[0] + 10), int(point2[1] + 20))
        cv2.putText(frame, f"Angle: {angle:.2f} deg", disp_point, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


    #draw axis
    dx_ref, dy_ref = axis_directions.get(axis, (length, 0))
    ref_end = (int(point2[0] + dx_ref), int(point2[1] + dy_ref))

    # Draw the reference axis
    #cv2.line(frame, (int(point2[0]), int(point2[1])), ref_end, (0, 255, 0), 2)
    return frame, ref_end, angle


def get_rotated_bounding_box(binary_mask):
    if binary_mask.dtype != np.uint8:
        binary_mask = (binary_mask > 0).astype(np.uint8) * 255
    # Find contours in the binary mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None, None  # No contours found

    # Get the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Get the rotated bounding box
    rotated_box = cv2.minAreaRect(largest_contour)
    
    box = cv2.boxPoints(rotated_box)
    box = np.int32(box)  # Convert to integer
    # Calculate the centroid of the points
    center = np.mean(box, axis=0)
    
    # Sort the points based on the angle with respect to the centroid
    sorted_points = np.array(sorted(box, key=lambda p: np.arctan2(p[1] - center[1], p[0] - center[0])))
    sorted_rotated_box = cv2.minAreaRect(sorted_points)
    return  sorted_points,sorted_rotated_box

def get_centroid_from_mask(binary_mask):
    moments = cv2.moments(binary_mask)
    if moments["m00"] != 0:
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        player_center = (cx, cy)
        return player_center
    return None

def frame_skipping_on_angle_anomaly(frame_indices,angles,threshold=20,skip=False):
    anomalous_frames = set()  # Use a set to store indices of frames with anomalies
    direction = {}


    for i in range(1, len(angles) - 1):
        if angles[i]>355 and angles[i]<360 and angles[i]>0 and angles[i]<5:
            continue
        else:
            prev_diff = (angles[i] - angles[i - 1])
            next_diff = (angles[i + 1] - angles[i])
        
            if abs(prev_diff) > threshold and abs(next_diff) > threshold:
                anomalous_frames.add(frame_indices[i+1])
            elif prev_diff>0 and next_diff>0:
                direction[frame_indices[i+1]]  = 1   #clockwise
            elif prev_diff<0 and next_diff<0:
                direction[frame_indices[i+1]] = 0   #counterclockwise
            elif skip:
                anomalous_frames.add(frame_indices[i+1])


    # Add boundary checks for the first and last angles
    if abs(angles[1] - angles[0]) > threshold:
        anomalous_frames.add(frame_indices[0])
    if abs(angles[-1] - angles[-2]) > threshold:
        anomalous_frames.add(frame_indices[-1])
    return anomalous_frames, direction

def convert_ndarray_to_list(obj):
    """
    Recursively convert all numpy.ndarray objects in the data structure to lists.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [convert_ndarray_to_list(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_ndarray_to_list(value) for key, value in obj.items()}
    else:
        return obj
    
# ---------------- Helper Functions for Firebase ----------------
def parse_gs_url(gs_url):
    """Parses a gs:// URL into (bucket_name, blob_name)."""
    assert gs_url.startswith("gs://")
    without_prefix = gs_url[5:]
    parts = without_prefix.split('/', 1)
    bucket_name = parts[0]
    blob_name = parts[1] if len(parts) > 1 else ""
    return bucket_name, blob_name

def zip_folder(folder_path, zip_path):
    """Zips the contents of a folder."""
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                full_path = os.path.join(root, file)
                # Save only the file name in the archive.
                zipf.write(full_path, arcname=file)