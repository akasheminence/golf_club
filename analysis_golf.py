import numpy as np
def center_of_bounding_box(x1, y1, x2, y2):
    """
    Goal: Calculate the center coordinates of a bounding box.
    """
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return center_x, center_y

def get_last_frame_number(dictionary_name):
    """
    Goal: to get the last frame number in the dictionary
    - dictionary_name: dictionary containing body keypoints coordinates or any other data
    """
    if not dictionary_name:
        return None  # Return None if the dictionary is empty

    return max(dictionary_name.keys())

def get_halfway_frame(dictionary_name):
    """
    Goal: Get the halfway frame number in the dictionary
    - dictionary_name: dictionary containing body keypoints coordinates or any other data
    """
    if not dictionary_name:
        return None  # Return None if the dictionary is empty

    first_frame = min(dictionary_name.keys())
    last_frame = max(dictionary_name.keys())
    
    halfway_frame = (int(first_frame )+ int(last_frame)) // 2  # Ensure integer division

    return int(halfway_frame)


def get_first_landmark_no_predecessors(dictionary_name, larger_joint_name, smaller_joint_name):

    '''
    Goal: Finds the first landmark where there are no dictionary_name before it, typically, this means finding
    the first frame of the second sequence
    Variables:
    - larger_joint_name: the joint that is currently larger than the smaller joint in x or y
    - smaller_joint_name: the joint that is samller but will be larger than the larger joint in x or y
    '''

    frames_meeting_condition = []


    for frame, dictionary_name in dictionary_name.items():
        larger_joint = dictionary_name.get(larger_joint_name, {}).get("y", None)
        smaller_joint = dictionary_name.get(smaller_joint_name, {}).get("y", None)

        if larger_joint is not None and smaller_joint is not None:
            if larger_joint > smaller_joint:
                frames_meeting_condition.append(frame)

    frames_without_predecessor = [frame for frame in frames_meeting_condition
                                   if (frame - 1) not in frames_meeting_condition]

    # Find the smallest frame number other than 0, if possible
    if frames_without_predecessor:
        non_zero_frames = [frame for frame in frames_without_predecessor if frame > 0]
        if non_zero_frames:
            smallest_frame = min(non_zero_frames)
        else:
            smallest_frame = min(frames_without_predecessor)
    else:
        smallest_frame = None

    return smallest_frame



def max_hand_height(dictionary_name, frame_threshold, P4_or_P9, handedness):
    max_dist_y = 0
    target_frame = None  # Initialize target_frame

    # Determine the relevant wrist and toe keypoints based on handedness
    if handedness == "RIGHT":
        wrist_keypoint = 'LEFT_WRIST'
        toe_keypoint = 'RIGHT_FOOT_INDEX'
    else:
        wrist_keypoint = 'RIGHT_WRIST'
        toe_keypoint = 'LEFT_FOOT_INDEX'

    # Filter frames based on P4_or_P9
    if P4_or_P9 == 'P9':
        relevant_frames = [k for k in dictionary_name.keys() if k >= frame_threshold]
    else:
        relevant_frames = [k for k in dictionary_name.keys() if k <= frame_threshold]

    # Iterate through the relevant frames
    for frame in relevant_frames:
        keypoints = dictionary_name[frame]  # Get keypoints for the current frame

        # Access the wrist and toe positions
        wrist = keypoints[wrist_keypoint]['y']
        toe = keypoints[toe_keypoint]['y']

        # Calculate the distance in the y direction
        dist_y = abs(wrist - toe)
        if dist_y > max_dist_y:
            max_dist_y = dist_y
            target_frame = frame

    return target_frame

def ensure_list(joint_names):
    """Utility function to ensure the input is a list."""
    return joint_names if isinstance(joint_names, list) else [joint_names]

def get_first_landmark(dictionary_name, larger_joint_names, smaller_joint_names, x_or_y_coordinates, frame_range):
    """
    Goal: Gets the first landmark frame within a specified range where any of the larger joints are greater than any of the smaller joints in the given coordinate.

    Parameters:
    - dictionary_name: A dictionary containing frame data.
    - larger_joint_names: a single joint name or a list of joint names that are considered larger
    - smaller_joint_names: a single joint name or a list of joint names that are considered smaller
    - x_or_y_coordinates: either 'x' or 'y' coordinates of the joints to be taken
    - frame_range: a tuple (start_frame, end_frame) defining the range of frames to check
    """
    # Ensure inputs are lists
    larger_joint_names = ensure_list(larger_joint_names)
    smaller_joint_names = ensure_list(smaller_joint_names)

    if x_or_y_coordinates not in ["x", "y"]:
        raise ValueError("x_or_y_coordinates must be either 'x' or 'y'")

    start_frame, end_frame = frame_range  # Unpack the frame range

    for frame in range(start_frame, end_frame + 1):  # Include end_frame
        if frame not in dictionary_name:
            continue  # Skip frames that are not in the dictionary

        landmarks = dictionary_name[frame]

        # Check each larger joint
        for larger_joint_name in larger_joint_names:
            # Safely get the coordinate for the larger joint
            larger_joint = landmarks.get(larger_joint_name, {}).get(x_or_y_coordinates, None)

            if larger_joint is None:
                continue  # Skip if larger joint's coordinate is None

            # Check each smaller joint
            for smaller_joint_name in smaller_joint_names:
                # Safely get the coordinate for the smaller joint
                smaller_joint = landmarks.get(smaller_joint_name, {}).get(x_or_y_coordinates, None)

                if smaller_joint is not None:
                    if larger_joint > smaller_joint:
                        return frame  # Return the first frame where the condition is met

    # Return None if no such frame is found
    return None

def get_center_of_object(dictionary_name, frame_number, object_name):
    """
    Goal: Calculate the center coordinates of a bounding box for a specified object in a specified frame.

    Variables:
    - dictionary_name: which dictionary_name dictionary to get
    - frame_number: the frame number to get the cordinates of
    - object_name: name of the object in question

    """


    if frame_number in dictionary_name:
        for bbox_id, bbox in dictionary_name[frame_number].items():
            if bbox['label'] == object_name:
                x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                return  center_x, center_y

    return None  # Return None if the object or frame is not found


def lowest_point_of_golf_club(dictionary_name, frame_threshold, P3_or_P5):# needs to be updated to json format
    """
    Goal: Find the frame number and the lowest point of the golf club shaft during the P3 or P5 phase.

    Parameters:
    - dictionary_name: Dictionary containing detection data for each frame.
    - frame_threshold: Frame number used as a reference point to not get setup frames.


    Returns:
    - lowest_frame: The frame number with the lowest point of the shaft.
    - lowest_point: The lowest point of the shaft in that frame.
    """

    frames_to_check = []


    frames_to_check = [frame for frame in dictionary_name.keys() if frame > frame_threshold]

    lowest_frame = None
    lowest_point = 0

    for frame in frames_to_check:
        for item in dictionary_name[frame]:
            if item['label'] == 'shaft':
                y2 = item['x2_y2']['coordinates'][1]
                if y2 > lowest_point:
                    lowest_point = y2
                    lowest_frame = frame

    return lowest_frame

    # return max(dictionary_name.keys())
def forearm_parallel_to_ground_frame(dictionary_name, upper_threshold, lower_threshold, handedness, position):
    """
    Goal: Find the frame where the forearm is parallel to the ground by finding the frame where the wrist and elbow have
    the most similar y value within the specified frame range based on the position provided.

    Parameters:
    - dictionary_name: dictionary containing body keypoints coordinates.
    - lower_threshold: the lower frame number/threshold.
    - upper_threshold: the upper frame number/threshold.
    - handedness: the dominant hand of the player.
    - position: 'P3', 'P5', or 'P8' to define frame selection criteria.

    Returns:
    - The frame number where the forearm is closest to parallel to the ground.
    """
    # Adjusted logic for side selection
    side = "LEFT" if handedness == "RIGHT" else "RIGHT"
    min_diff = float('inf')
    best_frame = None

    # Determine the frame range based on the position
    if position == 'P3':
        frame_range = [frame for frame in sorted(dictionary_name.keys()) if frame < upper_threshold]
    elif position == 'P5':
        frame_range = [frame for frame in sorted(dictionary_name.keys()) if lower_threshold < frame < upper_threshold]
    else:
        return "Invalid position. Please choose 'P3', or 'P5'."

    if not frame_range:
        return "No frames found within the specified range."

    for frame in frame_range:
        points = dictionary_name.get(frame, {})
        wrist_y = points.get(f'{side}_WRIST', {}).get('y')
        elbow_y = points.get(f'{side}_ELBOW', {}).get('y')


        if wrist_y is None or elbow_y is None:
            continue

        diff = abs(wrist_y - elbow_y)

        if diff < min_diff:
            min_diff = diff
            best_frame = frame

    return best_frame if best_frame is not None else "No suitable frame found."



def get_P2_side (handedness,body_landmarks_side):
    if handedness== "RIGHT":
        hip = "RIGHT_HIP"
        wrist = "RIGHT_WRIST"
    else:
        hip = "LEFT_HIP"
        elbow = "LEFT_WRIST"

    return get_first_landmark(body_landmarks_side,hip, wrist,"y", (1,P10_frame_side))


def get_P2_front (handedness,body_landmarks_front):
    side = "LEFT" if handedness == "RIGHT" else "RIGHT"
    if handedness== "RIGHT":
        hip ="RIGHT_HIP"
        hands =[ f'{handedness}_PINKY',f'{handedness}_INDEX',f'{handedness}_THUMB']
    else:
        hip = "LEFT_HIP"
        hands =[ f'{handedness}_PINKY',f'{handedness}_INDEX',f'{handedness}_THUMB']

    return get_first_landmark(body_landmarks_front,hip, hands,"y",(1,P10_frame_front) )


def get_P4_front(dictionary_name, max_frame):
    '''
    Goal: to determine the top of the swing/ end of backswing.
    Variables:
    - dictionary_name: list of dictionaries containing body keypoints coordinates
    - max_frame: the maximum frame number to consider so that it won't include the follow through
    '''

    max_head_angle = float('-inf')
    frame_number = None

    for entry in dictionary_name:
        if entry['frame'] > max_frame-1 and entry['head_angle'] < 0:
            if entry['head_angle'] > max_head_angle:
                max_head_angle = entry['head_angle']
                frame_number = entry['frame']

    return frame_number

def get_P3_frame_side (handedness,body_landmarks_side):
    handedness = "RIGHT" if handedness == "LEFT" else "LEFT"

    hands =[f'{handedness}_WRIST', f'{handedness}_PINKY',f'{handedness}_INDEX',f'{handedness}_THUMB']
    shoulder = f'{handedness}_SHOULDER'

    return get_first_landmark(body_landmarks_side,shoulder, hands ,"y",(1,P10_frame_side))

def get_P5_frame_side (handedness,body_landmarks_side):
    handedness = "RIGHT" if handedness == "LEFT" else "LEFT"

    hands =[f'{handedness}_WRIST', f'{handedness}_PINKY',f'{handedness}_INDEX',f'{handedness}_THUMB']
    shoulder = f'{handedness}_SHOULDER'

    return get_first_landmark(body_landmarks_side,hands,shoulder,"y",(halfway_frame_side,P10_frame_side))

def get_P8_frame_side (handedness,body_landmarks_side):

    hands =f'{handedness}_WRIST'
    hip = f'{handedness}_HIP'

    return get_first_landmark(body_landmarks_side, hip,hands,"y",(P7_frame_side,P10_frame_side))



def get_P8_front(handedness,body_landmarks_front):

    # Determine wrist and hip landmarks based on handedness
    wrist = 'RIGHT_WRIST' if handedness == "RIGHT" else 'LEFT_WRIST'
    hip = 'LEFT_HIP' if handedness == "RIGHT" else 'RIGHT_HIP'
    return get_first_landmark(body_landmarks_front, hip,wrist,"y",(P7_frame_front,P10_frame_front))

def get_P9_frame_side (handedness,body_landmarks_side):
    handedness = "RIGHT" if handedness == "LEFT" else "LEFT"

    hands =[f'{handedness}_WRIST', f'{handedness}_PINKY',f'{handedness}_INDEX',f'{handedness}_THUMB']
    shoulder = f'{handedness}_SHOULDER'

    return get_first_landmark(body_landmarks_side, shoulder,hands,"y",(P7_frame_side,P10_frame_side))


def get_P9_front(handedness,body_landmarks_front):
    handedness = "RIGHT" if handedness == "LEFT" else "LEFT"
    wrist =f'{handedness}_WRIST'
    shoulder = f'{handedness}_SHOULDER'


    return get_first_landmark(body_landmarks_front,shoulder,wrist, "y", (P7_frame_front,P10_frame_front))







#calculate angle
def calculate_angle_three_points(a, b, c):
    """
    Goal: Calculate the angle between three points a, b, and c where b is the center point.
    """
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle >180.0:
        angle = 360-angle

    return angle

def toe_distance_calc(dictionary_name,handedness):
    """
    Goal: get 1/4 of the distance of the heel and toe. Used as a relative distance to check for P1

    Variables:
    - dictionary_name: dictionary containing body keypoints coordinates
    """
    total_distance_fifth = 0
    frame_count = 0

    for frame, landmarks in dictionary_name.items():
        # Safely get the 'x' coordinate of RIGHT_HEEL and RIGHT_FOOT_INDEX, or set them to None if the landmark doesn't exist
        heel_x = landmarks.get(f'{handedness}_HEEL', {}).get('x', None)
        foot_index_x = landmarks.get(f'{handedness}_FOOT_INDEX', {}).get('x', None)

        # Check if both landmarks exist for the current frame
        if heel_x is not None and foot_index_x is not None:
            # Calculate the distance between the 'x' coordinates
            x_distance = abs(foot_index_x - heel_x)

            # Calculate 1/4 of the x_distance
            x_distance_fifth = x_distance /4

            total_distance_fifth += x_distance_fifth
            frame_count += 1

    if frame_count > 0:
        average_distance_fifth = total_distance_fifth / frame_count
        return average_distance_fifth
    else:
        return None



def hip_turn(dictionary_name, frame_number, toe_distance, handedness):
    """
    Goal: Check how much the hip has turned. Used as a relative distance to check for how much the hip is turning.

    Variables:
    - dictionary_name: Dictionary containing body keypoints coordinates.
    - frame_number: Frame number to check.
    - handedness: the dominant hand of the golfer
    """
    left_hip_x = dictionary_name[frame_number]["LEFT_HIP"]['x']
    right_hip_x = dictionary_name[frame_number]["RIGHT_HIP"]['x']
    heel = dictionary_name[frame_number][f'{handedness}_HEEL']['x']
    toe = dictionary_name[frame_number][f'{handedness}_FOOT_INDEX']['x']

    # Define conditions based on handedness
    if handedness == "RIGHT":
        did_not_turn_enough = right_hip_x > left_hip_x - (toe_distance / 2)
        turned_too_much = heel -(toe_distance / 5) > right_hip_x
    else:
        did_not_turn_enough = right_hip_x > left_hip_x + (toe_distance / 2)
        turned_too_much = left_hip_x < heel+(toe_distance / 5)

    # Check conditions based on the calculated values
    if did_not_turn_enough or turned_too_much:
        if did_not_turn_enough:
            message = "You didn't turn enough"
        elif turned_too_much:
            message = "You turned too much"
    else:
        message = "You turned correctly"

    return message  # Return message

#distance_from_ball
def distance_from_ball(dictionary_name,frame_number,handedness):
    """
    Goal: Checks how far the golfer is standing too far or close

    Variables:
    - dictionary_name: dictionary containing body keypoints coordinates
    - frame_number: the frame to calculate the angle at
    - handedness: the dominant hand of the golfer

    """

    if dictionary_name[frame_number][f'{handedness}_WRIST']['x']<dictionary_name[frame_number][f'{handedness}_ELBOW']['x']:
        message= "You are standing too close to the ball"
    elif dictionary_name[frame_number][f'{handedness}_WRIST']['x']>dictionary_name[frame_number][f'MOUTH_{handedness}']['x']:
        message= "You are standing too far away from the ball"
    else:
        message= "You are standing at a perfect distance"
    return message



#feet position
def stance_alignment(dictionary_name, toe_distance, frame_number, handedness):
    """
    Goal: check if your feet are aligned

    Variables:
    - dictionary_name: dictionary containing body keypoints coordinates
    - frame_number: the frame to calculate the angle at
    -

    """
    distance_between_feet= dictionary_name[frame_number]["RIGHT_FOOT_INDEX"]['x']-dictionary_name[frame_number]["LEFT_FOOT_INDEX"]['x']
    if distance_between_feet> toe_distance:
        message= ("Your right foot is too in front")
    elif distance_between_feet < - toe_distance:
        message=  ("Your left foot is too in front")
    else:
        message= ("Your feet are perfect")
    return message

def get_angles(position, golf_club):
    angles = {
        "driver": {
            "P1": (128, 143),
            "P2": (138, 153),
            "P3": (148, 163),
            "P4": (148, 163),
            "P5": (128, 143),
            "P7": (128, 143)
        },
        "iron": {
            "P1": (130, 140),
            "P2": (138, 153),
            "P3": (138, 153),
            "P4": (132, 153),
            "P5": (128, 143),
            "P7": (128, 143)
        }
    }

    # Get the angles for the specified position
    position_angles = angles.get(golf_club)

    # Return the angles for the specified position, or (None, None) if not found
    return position_angles.get(position, (None, None)) if position_angles else (None, None) #Back Angle
def calc_back_angle(dictionary_name, frame_number, handedness, club, position):


    """
    Goal: check if the back is too bent or straight

    Variables:
    - dictionary_name: dictionary containing body keypoints coordinates
    - angle_straight: the upper bound for the angle, if larger than this it means that they are standing too straight
    - angle_crooked:  the upper bound for the angle, if smaller than this it means that they are standing too bent
    - frame_number: the frame to calculate the angle at
    - handedness: the dominant hand of the golfer

    """

    hip = dictionary_name[frame_number][f'{handedness}_HIP']['x'],dictionary_name[frame_number][f'{handedness}_HIP']['y']
    knee = dictionary_name[frame_number][f'{handedness}_KNEE']['x'],dictionary_name[frame_number][f'{handedness}_KNEE']['y']
    shoulder = dictionary_name[frame_number][f'{handedness}_SHOULDER']['x'],dictionary_name[frame_number][f'{handedness}_SHOULDER']['y']



    angle_crooked, angle_straight = get_angles(position, club)
    # Calculate the angle.
    back_angle = calculate_angle_three_points(shoulder,hip,knee )

    if back_angle > angle_straight:
        message = "You are standing too upright"
    elif back_angle < angle_crooked:
        message = "You are standing too bent"
    else:
        message = "Your back is at the perfect angle"

    return message

# Elbow Angle
def elbow_angle(dictionary_name,frame_number,angle_crooked,angle_straight,handedness, trail_or_lead):
    """
    Goal: check if the left elbow is too bent or straight

    Variables:
    - dictionary_name: dictionary containing body keypoints coordinates
    - frame_number: the frame to calculate the angle at
    - angle_straight: the upper bound for the angle, if larger than this it means that the eblow is too straight too straight
    - angle_crooked:  the upper bound for the angle, if smaller than this it means that the eblow is too bent
    - handedness: the dominant hand of the golfer


    """
    if trail_or_lead == "lead":
        if handedness == "RIGHT":
            side = "LEFT"
        else:
            side = "RIGHT"
    elif trail_or_lead == "trail":
        if handedness == "RIGHT":
            side = "RIGHT"
        else:
            side = "LEFT"

    else:
        side = None

    wrist = dictionary_name[frame_number][f'{side}_WRIST']['x'],dictionary_name[frame_number][f'{side}_WRIST']['y']
    elbow = dictionary_name[frame_number][f'{side}_ELBOW']['x'],dictionary_name[frame_number][f'{side}_ELBOW']['y']
    shoulder = dictionary_name[frame_number][f'{side}_SHOULDER']['x'],dictionary_name[frame_number][f'{side}_SHOULDER']['y']

    # Calculate the angle.

    elbow_angle = calculate_angle_three_points(shoulder,elbow, wrist)
    elbow= side.lower()

    # Message based on elbow angle
    if elbow_angle < angle_crooked:
        message = f"Your {elbow} elbow is too crooked"
    elif elbow_angle > angle_straight:
        message = f"Your {elbow} elbow is too straight"
    else:
        message = f"Your {elbow} elbow is perfectly bent"
    return message

# knee angle

def knee_angle(dictionary_name,angle_crooked,angle_straight,frame_number, handedness):

    """
    Goal: check if the knee is too bent or straight

    Variables:
    - dictionary_name: dictionary containing body keypoints coordinates
    - angle_straight: the upper bound for the angle, if larger than this it means that then the knee too straight
    - angle_crooked:  the upper bound for the angle, if smaller than this it means that the knee is too bent
    - frame_number: the frame to calculate the angle at
    - handedness: the dominant hand of the golfer
    """

    hip = dictionary_name[frame_number][f'{handedness}_HIP']['x'],dictionary_name[frame_number][f'{handedness}_HIP']['y']
    knee = dictionary_name[frame_number][f'{handedness}_KNEE']['x'],dictionary_name[frame_number][f'{handedness}_KNEE']['y']
    ankle = dictionary_name[frame_number][f'{handedness}_ANKLE']['x'],dictionary_name[frame_number][f'{handedness}_ANKLE']['y']


    # Calculate the angle.
    knee_angle = calculate_angle_three_points(hip, knee, ankle)

    if knee_angle> angle_straight:
        message=  ("Your knee is too straight")
    elif knee_angle<angle_crooked:
        message=  ("Your knee is too croocked")
    else:
        message= "Your knee is bent correctly"
    return message

def shoulder_sway(dictionary_name, start_frame, final_frame, handedness):
    """
    Goal: check if the shoulder is moving too much

    Variables:
    - dictionary_name: dictionary containing body keypoints coordinates
    - start_frame: the frame number for the start
    - final_frame: the finishing frame where the check ends
    - handedness: the dominant hand of the golfer
    """

    # Get 'x' coordinates for the RIGHT or LEFT SHOULDER in frames

    shoulder_start = dictionary_name[start_frame][f'{handedness}_SHOULDER']['x']
    shoulder_final = dictionary_name[final_frame][f'{handedness}_SHOULDER']['x']

    # Calculate shoulder distance
    shoulder_distance = abs(shoulder_final - shoulder_start)

    # Calculate the distance from the left mouth to the right mouth at frame 0 using x coordinates
    left_mouth_x = dictionary_name[0]['MOUTH_LEFT']['x']  # Using the mouth position for the dominant side
    right_mouth_x = dictionary_name[0]['MOUTH_RIGHT']['x']  # Opposite side mouth

    mouth_distance = abs(left_mouth_x - right_mouth_x) * 2  # Distance multiplied by 2

    # Compare the distances and return a message
    if shoulder_distance > mouth_distance:
        message = "Your shoulder is moving too much"
    else:
        message = "Your shoulder movement is good"

    return message

def head_movement(dictionary_name, start_frame, final_frame,handedness):
    """
    Goal: check if the head is moving too much from the starting frame to the finishing frame

    Variables:
    - dictionary_name: dictionary containing body keypoints coordinates
    - start_frame: the frame number for the start
    - Final frame:  the finishing frame where the check ends
    - frame_number: the frame to calculate the angle at
    - handedness: the dominant hand of the golfer

    """
    side = "RIGHT" if handedness == "LEFT" else "RIGHT"

    # Get 'y' coordinates for the RIGHT_EAR and MOUTH_RIGHT in both frames
    ear_y_start_frame = dictionary_name[start_frame][f'{side}_EAR']['y']
    ear_y_final_frame = dictionary_name[final_frame][f'{side}_EAR']['y']
    mouth_y_start_frame = dictionary_name[start_frame][f'MOUTH_{side}']['y']
    mouth_y_final_frame = dictionary_name[final_frame][f'MOUTH_{side}']['y']

    # Ensure all required landmarks are present

    # Calculate the y distance of the RIGHT_EAR between the two frames
    ear_movement = abs(ear_y_final_frame - ear_y_start_frame)

    # Calculate the average y distance between the RIGHT_EAR and MOUTH_RIGHT across the two frames
    average_ear_to_mouth_distance = (abs(ear_y_start_frame - mouth_y_start_frame) +
                                     abs(ear_y_final_frame - mouth_y_final_frame)) / 2

    # Compare the distances and return a message
    if ear_movement > average_ear_to_mouth_distance:
        message = "Your head is moving too much"
    else:
        message = "Your head movement is good"

    return message





def overswing(data_dictionary, frame_number):

    """
    Goal: Check the shaft angle for the shaft in the given frame and print the appropriate message.

    Parameters:
    - dictionary_name: Dictionary containing detection data for each frame.
    - frame_number: The frame number to check the shaft angle for.

    """
    frame = next((item for item in dictionary_name if item["frame"] == frame_number), None)


    shaft_angle = frame["shaft_angle"]

    if club_angle < 85:
        return "You have an overswing"
    else:
        return "You don't have an overswing"



def feet_width(dictionary_name, handedness):
    '''
    Goal: to determine the stance width based on foot and shoulder coordinates.
    Variables:
    - dictionary_name: dictionary containing body keypoints coordinates
    '''
    # Assuming only one frame to check
    frame_data = list(dictionary_name.values())[0]

    left_toe = frame_data.get('LEFT_FOOT_INDEX', {}).get('x', None)
    right_toe = frame_data.get('RIGHT_FOOT_INDEX', {}).get('x', None)
    left_heel = frame_data.get('LEFT_HEEL', {}).get('x', None)
    right_heel = frame_data.get('RIGHT_HEEL', {}).get('x', None)
    left_shoulder = frame_data.get('LEFT_SHOULDER', {}).get('x', None)
    right_shoulder = frame_data.get('RIGHT_SHOULDER', {}).get('x', None)



    if handedness=="RIGHT":
        if left_heel > left_shoulder and right_heel < right_shoulder:
            message = "Your stance is too wide"
        elif left_toe < left_shoulder and right_toe> right_shoulder:
            message =  "Your stance is too narrow"
        else:
            message = "Your stance is perfect"
    else:
        if right_heel > right_shoulder and left_heel < left_shoulder:
            message = "Your stance is too wide."
        elif right_toe < right_shoulder and left_toe > left_shoulder:
            message =  "Your stance is too narrow"
        else:
            message = "Your stance is perfect"

    return message

# Hand position
def hand_position(dictionary_name,inside_joint, outside_joint, frame_number, hand_joint_name) :
    """
    Goal: Checks if the hands are too inside or outside

    Varaibles:
    - dictionary_name: dictionary containing body keypoints coordinates
    - inside_joint: joint that is a marker if the swing is too inside
    - outside_joint: joint that is a marker if the swing is too outside
    - hand_joint_name: part of the hand that is used as the indicator for the checks

    """
    if dictionary_name[frame_number][hand_joint_name]['x']<dictionary_name[frame_number][inside_joint]['x']:
        message= "Your swing is too inside"
    elif dictionary_name[frame_number][hand_joint_name]['x']>dictionary_name[frame_number][outside_joint]['x']:
        message= "Your swing is too outhandedness"
    else:
        message = "Your swing is on path"
    return message

def hand_position_P2_side (body_landmarks_side,P2_frame_side,handedness):
    if handedness== "RIGHT":
        return hand_position(body_landmarks_side, 'LEFT_SHOULDER', 'MOUTH_RIGHT',P2_frame_side, 'RIGHT_WRIST')

    else:
        return hand_position(body_landmarks_side, 'RIGHT_SHOULDER', 'MOUTH_LEFT',P2_frame_side, 'RIGHT_LEFT')

    return hand_position(body_landmarks_side, f'{handedness}_SHOULDER', f'MOUTH_{handedness}',P2_frame_side, f'{handedness}_WRIST')

# def get_P2_front (handedness):
#     if handedness== "RIGHT":
#         hip = "LEFT_HIP"
#         elbow = "RIGHT_ELBOW"
#     else:
#         hip = "RIGHT_HIP"
#         elbow = "LEFT_ELBOW"

#     return get_first_landmark(body_landmarks_side,hip, elbow,"x")

def hand_position_P3_side (body_landmarks_side,P2_frame_side,handedness):
    if handedness=="RIGHT":
        hip= "RIGHT_HIP"
        shoulder= "LEFT_SHOULDER"
        pinky= "LEFT_PINKY"
    else :
        hip= "LEFT_HIP"
        shoulder= "RIGHT_SHOULDER"
        pinky= "RIGHT_PINKY"

    return hand_position(body_landmarks_side, hip, shoulder, P3_frame_side, pinky)

def hand_position_P5 (body_landmarks_side,P4_frame_side,handedness):
    if handedness=="RIGHT":
        heel= "RIGHT_HEEL"
        shoulder= "LEFT_SHOULDER"
        wrist= "LEFT_WRIST"
    else :
        heel= "LEFT_HEEL"
        shoulder= "RIGHT_SHOULDER"
        wrist= "RIGHT_WRIST"

    return hand_position(body_landmarks_side, heel, shoulder, P5_frame_side, wrist)



def check_swing_plane(dictionary_name, frame_number, upper_threshold, lower_threshold):
    '''
    Check the club angle for a specific frame against upper and lower thresholds to determine the swing plane.

    Parameters:
    - dictionary_name: A list of dictionaries containing swing data.
    - frame_number: The specific frame number to check.
    - upper_threshold: The upper limit for a good swing plane.
    - lower_threshold: The lower limit for a good swing plane.

    Returns:
    - A message indicating the status of the swing plane for the specified frame.
    '''

    # Find the specific frame in the data
    frame = next((item for item in dictionary_name if item["frame"] == frame_number), None)



    club_angle = frame["club_angle"]

    # Check club_angle against thresholds
    if club_angle < lower_threshold:
        return "Your swing is too flat."
    elif club_angle > upper_threshold:
        return "Your swing is too upright."
    else:
        return "Your swing plane is good."


def check_wrist_hinge(dictionary_name, frame_number, upper_threshold, lower_threshold):
    '''
    Check the club angle for a specific frame against upper and lower thresholds to determine the swing plane.

    Parameters:
    - dictionary_name: A list of dictionaries containing swing data.
    - frame_number: The specific frame number to check.
    - upper_threshold: The upper limit for a good swing hinge.
    - lower_threshold: The lower limit for a good swing hinge.

    Returns:
    - A message indicating the status of the swing plane for the specified frame.
    '''

    # Find the specific frame in the data
    frame = next((item for item in dictionary_name if item["frame"] == frame_number), None)

    club_angle = frame["club_angle"]

    # Check club_angle against thresholds
    if club_angle < lower_threshold:
        return "Your hinge is too hinged"
    elif club_angle > upper_threshold:
        return "You need to hinge your wrist more"
    else:
        return "Your swing plane is good"

def scooping(dictionary_name, frame_number):
    """
    Check if the golfer is scooping the ball based on the shaft angle in the given frame.

    Parameters:
    - dictionary_name: Dictionary containing detection data for each frame.
    - frame_number: The frame number to check.

    Returns:
    - message: A string indicating whether the golfer is scooping the ball or not.
    """
    frame = next((item for item in dictionary_name if item["frame"] == frame_number), None)

    club_angle = frame["club_angle"]

    if shaft_angle > 60 :
        return "The golfer is scooping the ball"
    else:
        return "The golfer is not scooping the ball"

def analyze_golf_swing(body_landmarks_side, body_landmarks_front):


    global halfway_frame_side
    global P1_frame_side, P2_frame_side, P3_frame_side, P4_frame_side
    global P5_frame_side, P7_frame_side, P8_frame_side, P9_frame_side, P10_frame_side

    global P1_frame_front, P2_frame_front, P3_frame_front, P4_frame_front
    global P5_frame_front, P7_frame_front, P8_frame_front, P9_frame_front, P10_frame_front


    golf_club_landmarks_side = 0                
    golf_club_landmarks_front = 0   
    club_type= "iron"


    handedness="RIGHT"
    P7_frame_side=325 #just for testing purposes
    halfway_frame_side = get_halfway_frame(body_landmarks_side)
    P1_frame_side=0
    P10_frame_side= get_last_frame_number(body_landmarks_side)
    P2_frame_side= get_P2_side (handedness,body_landmarks_side)
    # global P7_frame_side=lowest_point_of_golf_club(golf_club_landmarks_side,P3_frame_side)# needs to be updated to json format
    P5_frame_side= get_P5_frame_side (handedness,body_landmarks_side)
    P3_frame_side=get_P3_frame_side (handedness,body_landmarks_side)
    P4_frame_side=max_hand_height(body_landmarks_side,P5_frame_side, "P4",handedness)
    P8_frame_side=get_P8_frame_side (handedness,body_landmarks_side)
    P9_frame_side=get_P9_frame_side (handedness,body_landmarks_side)

    print(P1_frame_side)
    print(P2_frame_side)
    print(P3_frame_side)
    print(P4_frame_side)
    print(P5_frame_side)
    print(P7_frame_side)
    print(P8_frame_side)
    print(P9_frame_side)















    handedness="RIGHT"
    P7_frame_front=366 #just for testing purposes
    halfway_frame_side=get_halfway_frame(body_landmarks_front)
    P1_frame_front=0
    P10_frame_front=get_last_frame_number(body_landmarks_front)
    P4_frame_front=309
    P2_frame_front=get_P2_front (handedness,body_landmarks_front)
    # P7_frame_front=lowest_point_of_golf_club(golf_club_landmarks_front,halfway_frame_side) #needs to be changed to get the obb of the clubhead
    # P4_frame_front=get_P4_front(golf_club_landmarks_front,P7_frame_front) #Will work with the new code
    P3_frame_front= forearm_parallel_to_ground_frame(body_landmarks_front,P4_frame_front,P1_frame_front,handedness, "P3")
    P5_frame_front= forearm_parallel_to_ground_frame(body_landmarks_front,P7_frame_front,P4_frame_front ,handedness, "P5")
    P8_frame_front= get_P8_front(handedness,body_landmarks_front)
    P9_frame_front= get_P9_front(handedness,body_landmarks_front)

    print(P1_frame_front)
    print(P2_frame_front)
    print(P3_frame_front)
    print(P4_frame_front)
    print(P5_frame_front)
    print(P7_frame_front)
    print(P8_frame_front)
    print(P9_frame_front)
    print(P10_frame_front)     











    results = {
        "P1": {},
        "P2": {},
        "P3": {},
        "P4": {},
        "P5": {},
        "P7": {},
        "P8": {}
    }
    toe_distance=toe_distance_calc(body_landmarks_side,handedness)

    # P1
    results["P1"]["Knee Angle"] = knee_angle(body_landmarks_side, 150,170, P1_frame_side,handedness)
    results["P1"]["Distance to the ball"] = distance_from_ball(body_landmarks_side, P1_frame_side,handedness)
    results["P1"]["Spine Angle"] =calc_back_angle(body_landmarks_side,P1_frame_side,handedness,club_type,"P1")
    results["P1"]["Feet Width"] = feet_width(body_landmarks_front,handedness)
    results["P1"]["Stance"]=stance_alignment(body_landmarks_front, toe_distance, P1_frame_side,handedness)
    # results["P1"]["Shaft Lean"]=check_shaft_lean(golf_club_landmarks_front,P1_frame_front,350,100) #uncomment to make it wokr
    results["P1"]["Lead Arm Angle"]=elbow_angle(body_landmarks_front,P1_frame_front,170,180,handedness,"lead")




    # P2
    results["P2"]["Hand Position"] = hand_position_P2_side (body_landmarks_side,P2_frame_side,handedness)
    results["P2"]["Knee Angle"] = knee_angle(body_landmarks_side, 150,170, P2_frame_side,handedness)
    results["P2"]["Spine Angle"] =calc_back_angle(body_landmarks_side,P2_frame_side,handedness,club_type,"P2")
    results["P2"]["Shoulder Sway"]=shoulder_sway(body_landmarks_front, P2_frame_front, P1_frame_front,handedness)
    results["P2"]["Hip Turn"]=hip_turn(body_landmarks_side,P2_frame_side,toe_distance,handedness)
    results["P2"]["Lead Elbow Bend"]=elbow_angle(body_landmarks_front,P2_frame_front,150,170,handedness,"lead")



    # P3
    results["P3"]["Knee Angle"] = knee_angle(body_landmarks_side, 145,170, P3_frame_side,handedness)
    # results["P3"]["Shaft Angle"] = check_swing_plane(body_landmarks_side, P3_frame_side, 65, 55) #uncomment to make it work
    results["P3"]["Spine Angle"] =calc_back_angle(body_landmarks_side,P3_frame_side,handedness,club_type,"P3")
    results["P3"]["Hand Position"] = hand_position_P3_side (body_landmarks_side,P3_frame_side,handedness)
    results["P3"]["Lead Elbow"]=elbow_angle(body_landmarks_front,P3_frame_front,145,170,handedness,"lead")
    results["P3"]["Hip Turn"]=hip_turn(body_landmarks_side,P2_frame_side,toe_distance,handedness)
    results["P3"]["Shoulder Sway"]=shoulder_sway(body_landmarks_front, P3_frame_front, P1_frame_front,handedness)
    # results["P3"]["Wrist Hinge"]=check_wrist_hinge(golf_club_landmarks_front, P3_frame_front, 200, 160)#uncomment to make it work


    # P4

    # results["P4"]["Overswing"] = overswing(golf_club_landmarks_front, P4_frame_side)#uncomment to make it work


    # P5
    results["P5"]["Spine Angle"] =calc_back_angle(body_landmarks_side, P5_frame_side,handedness,club_type,"P5")
    results["P5"]["Head Movement"] = head_movement(body_landmarks_front, P4_frame_side, P5_frame_side,handedness)
    results["P5"]["Hand Position"] = hand_position_P5 (body_landmarks_side,P4_frame_side,handedness)
    # results["P5"]["Shaft Angle"] = check_swing_plane(body_landmarks_side, P5_frame_side, 65, 50) #uncomment to make it work
    # results["P5"]["Casting"]=check_wrist_hinge(golf_club_landmarks_front, P5_frame_front, 180, 140)#uncomment to make it work



    # P7
    results["P7"]["Head Movement"] = head_movement(body_landmarks_side, P1_frame_side, P7_frame_side,handedness)
    results["P7"]["Chicken Wing"] = elbow_angle(body_landmarks_front, P7_frame_front,160, 175, handedness,"lead")
    # results["P7"]["Shaft Lean"] = check_shaft_lean(golf_club_landmarks_front, P7_frame_front,350,100)#uncomment to make it work
    results["P7"]["Spine Angle"] =calc_back_angle(body_landmarks_side, P7_frame_side,handedness, club_type,"P7")




    # P8
    # results["P8"]["Scooping"] = scooping(golf_club_landmarks_front, P8_frame_front) #uncomment to make it work
    results["P8"]["Chicken Wing"] = elbow_angle(body_landmarks_front, P8_frame_front,153, 170, handedness,"trail")







    return results



def analyze_golf_swing_side(body_landmarks_side):

    global halfway_frame_side
    global P1_frame_side, P2_frame_side, P3_frame_side, P4_frame_side
    global P5_frame_side, P7_frame_side, P8_frame_side, P9_frame_side, P10_frame_side    

    golf_club_landmarks_side = 0                
    club_type= "iron"


    handedness="RIGHT"
    P7_frame_side=325 #just for testing purposes
    halfway_frame_side = get_halfway_frame(body_landmarks_side)
    P1_frame_side=0
    P10_frame_side= get_last_frame_number(body_landmarks_side)
    P2_frame_side= get_P2_side (handedness,body_landmarks_side)
    # global P7_frame_side=lowest_point_of_golf_club(golf_club_landmarks_side,P3_frame_side)# needs to be updated to json format
    P5_frame_side= get_P5_frame_side (handedness,body_landmarks_side)
    P3_frame_side=get_P3_frame_side (handedness,body_landmarks_side)
    P4_frame_side=max_hand_height(body_landmarks_side,P5_frame_side, "P4",handedness)
    P8_frame_side=get_P8_frame_side (handedness,body_landmarks_side)
    P9_frame_side=get_P9_frame_side (handedness,body_landmarks_side)

    print(P1_frame_side)
    print(P2_frame_side)
    print(P3_frame_side)
    print(P4_frame_side)
    print(P5_frame_side)
    print(P7_frame_side)
    print(P8_frame_side)
    print(P9_frame_side)



    results = {
        "P1": {},
        "P2": {},
        "P3": {},
        "P5": {},
        "P7": {},
    }
    toe_distance=toe_distance_calc(body_landmarks_side,handedness)

    # P1
    results["P1"]["Knee Angle"] = knee_angle(body_landmarks_side, 150,170, P1_frame_side,handedness)
    results["P1"]["Distance to the ball"] = distance_from_ball(body_landmarks_side, P1_frame_side,handedness)
    results["P1"]["Spine Angle"] =calc_back_angle(body_landmarks_side,P1_frame_side,handedness,club_type,"P1")




    # P2
    results["P2"]["Hand Position"] = hand_position_P2_side (body_landmarks_side,P2_frame_side,handedness)
    results["P2"]["Knee Angle"] = knee_angle(body_landmarks_side, 150,170, P2_frame_side,handedness)
    results["P2"]["Spine Angle"] =calc_back_angle(body_landmarks_side,P2_frame_side,handedness,club_type,"P2")
    results["P2"]["Hip Turn"]=hip_turn(body_landmarks_side,P2_frame_side,toe_distance,handedness)



    # P3
    results["P3"]["Knee Angle"] = knee_angle(body_landmarks_side, 145,170, P3_frame_side,handedness)
    #results["P3"]["Shaft Angle"] = check_swing_plane(body_landmarks_side, P3_frame_side, 65, 55) #uncomment to make it work
    results["P3"]["Spine Angle"] =calc_back_angle(body_landmarks_side,P3_frame_side,handedness,club_type,"P3")
    results["P3"]["Hand Position"] = hand_position_P3_side (body_landmarks_side,P3_frame_side,handedness) 
    results["P3"]["Hip Turn"]=hip_turn(body_landmarks_side,P2_frame_side,toe_distance,handedness)




    # P5
    results["P5"]["Spine Angle"] =calc_back_angle(body_landmarks_side, P5_frame_side,handedness,club_type,"P5")
    results["P5"]["Hand Position"] = hand_position_P5 (body_landmarks_side,P5_frame_side,handedness)
    # results["P5"]["Shaft Angle"] = check_swing_plane(body_landmarks_side, P5_frame_side, 65, 50) #uncomment to make it work




    # P7
    results["P7"]["Head Movement"] = head_movement(body_landmarks_side, P4_frame_side, P7_frame_side,handedness)
    results["P7"]["Spine Angle"] =calc_back_angle(body_landmarks_side, P7_frame_side,handedness, club_type,"P7")

    


    return results



def analyze_golf_swing_front( body_landmarks_front):
    global P1_frame_front, P2_frame_front, P3_frame_front, P4_frame_front
    global P5_frame_front, P7_frame_front, P8_frame_front, P9_frame_front, P10_frame_front

    golf_club_landmarks_front = 0   
    club_type= "iron"


    handedness="RIGHT"
    P7_frame_front=366 #just for testing purposes
    halfway_frame_side=get_halfway_frame(body_landmarks_front)
    P1_frame_front=0
    P10_frame_front=get_last_frame_number(body_landmarks_front)
    P4_frame_front=309
    P2_frame_front=get_P2_front (handedness,body_landmarks_front)
    # P7_frame_front=lowest_point_of_golf_club(golf_club_landmarks_front,halfway_frame_side) #needs to be changed to get the obb of the clubhead
    # P4_frame_front=get_P4_front(golf_club_landmarks_front,P7_frame_front) #Will work with the new code
    P3_frame_front= forearm_parallel_to_ground_frame(body_landmarks_front,P4_frame_front,P1_frame_front,handedness, "P3")
    P5_frame_front= forearm_parallel_to_ground_frame(body_landmarks_front,P7_frame_front,P4_frame_front ,handedness, "P5")
    P8_frame_front= get_P8_front(handedness,body_landmarks_front)
    P9_frame_front= get_P9_front(handedness,body_landmarks_front)

    print(P1_frame_front)
    print(P2_frame_front)
    print(P3_frame_front)
    print(P4_frame_front)
    print(P5_frame_front)
    print(P7_frame_front)
    print(P8_frame_front)
    print(P9_frame_front)
    print(P10_frame_front)     



    results = {
        "P1": {},
        "P2": {},
        "P3": {},
        "P4": {},
        "P5": {},
        "P7": {},
        "P8": {}
    }

    # P1

    results["P1"]["Feet Width"] = feet_width(body_landmarks_front,handedness)
    #results["P1"]["Shaft Lean"]=check_shaft_lean(golf_club_landmarks_front,P1_frame_front,350,100) #uncomment to make it wokr
    results["P1"]["Lead Arm Angle"]=elbow_angle(body_landmarks_front,P1_frame_front,170,180,handedness,"lead")




    # P2
   
    results["P2"]["Shoulder Sway"]=shoulder_sway(body_landmarks_front, P2_frame_front, P1_frame_front,handedness)
    results["P2"]["Lead Elbow Bend"]=elbow_angle(body_landmarks_front,P2_frame_front,150,170,handedness,"lead")



    # P3


    results["P3"]["Lead Elbow"]=elbow_angle(body_landmarks_front,P3_frame_front,145,170,handedness,"lead")
    results["P3"]["Shoulder Sway"]=shoulder_sway(body_landmarks_front, P3_frame_front, P1_frame_front,handedness)
    #results["P3"]["Wrist Hinge"]=check_wrist_hinge(golf_club_landmarks_front, P3_frame_front, 200, 160)#uncomment to make it work


    # P4

    #results["P4"]["Overswing"] = overswing(golf_club_landmarks_front, P4_frame_front)#uncomment to make it work


    # P5
  
    #results["P5"]["Casting"]=check_wrist_hinge(golf_club_landmarks_front, P5_frame_front, 180, 140)#uncomment to make it work



    # P7
    results["P7"]["Chicken Wing"] = elbow_angle(body_landmarks_front, P7_frame_front,160, 175, handedness,"lead")
    #results["P7"]["Shaft Lean"] = check_shaft_lean(golf_club_landmarks_front, P7_frame_front,350,100)#uncomment to make it work

    


    # P8
    #results["P8"]["Scooping"] = scooping(golf_club_landmarks_front, P8_frame_front) #uncomment to make it work
    results["P8"]["Chicken Wing"] = elbow_angle(body_landmarks_front, P8_frame_front,153, 170, handedness,"trail")



    return results




    

# Usage
# club_type= "iron"
# handedness= "RIGHT"
# swing_analysis = analyze_golf_swing(body_landmarks_side, body_landmarks_front, golf_club_landmarks_side, golf_club_landmarks_front,
#                                     P1_frame_side, P1_frame_front, P2_frame_side, P3_frame_side,
#                                     P5_frame_side, P4_frame_side, P7_frame_side, P7_frame_front,
#                                     P8_frame_front, club_type, handedness)

# # To access results
# for phase, measurements in swing_analysis.items():
#     print(f"\n{phase}")
#     for measurement, value in measurements.items():
#         print(f"{measurement}: {value}")
# import json

# with open('body_landmarks_side.json', 'r') as f:
#     data = json.load(f)
# body_landmarks_side = {int(k): v for k, v in data.items()}

# with open('body_landmarks_front.json', 'r') as f:
#     data = json.load(f)
# body_landmarks_front = {int(k): v for k, v in data.items()}
# print(analyze_golf_swing(body_landmarks_side,body_landmarks_front))


if __name__ == '__main__':
    import json

    with open('body_landmarks_side.json', 'r') as f:
        data = json.load(f)
    body_landmarks_side = {int(k): v for k, v in data.items()}
    print(analyze_golf_swing_side(body_landmarks_side))

    with open('body_landmarks_front.json', 'r') as f:
        data = json.load(f)
    body_landmarks_front = {int(k): v for k, v in data.items()}
    print(analyze_golf_swing_front(body_landmarks_front))

    print(analyze_golf_swing(body_landmarks_side,body_landmarks_front))
    


    