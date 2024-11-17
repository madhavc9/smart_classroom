import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize video capture
def initialize_capture():
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Error: Couldn't open camera.")
        return None
    return camera

# Detect landmarks using Mediapipe
def detect_landmarks(frame, face_mesh):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    if results.multi_face_landmarks:
        mesh_coords = [(int(point.x * frame.shape[1]), int(point.y * frame.shape[0])) for point in results.multi_face_landmarks[0].landmark]
        return mesh_coords
    return None

# Calculate blink ratio
def calculate_blink_ratio(mesh_coords, right_eye_indices, left_eye_indices):
    # Calculate distances for right eye
    rh_right = mesh_coords[right_eye_indices[0]]
    rh_left = mesh_coords[right_eye_indices[8]]
    rv_top = mesh_coords[right_eye_indices[12]]
    rv_bottom = mesh_coords[right_eye_indices[4]]
    rh_distance = euclidean_distance(rh_right, rh_left)
    rv_distance = euclidean_distance(rv_top, rv_bottom)
    re_ratio = rh_distance / rv_distance

    # Calculate distances for left eye
    lh_right = mesh_coords[left_eye_indices[0]]
    lh_left = mesh_coords[left_eye_indices[8]]
    lv_top = mesh_coords[left_eye_indices[12]]
    lv_bottom = mesh_coords[left_eye_indices[4]]
    lh_distance = euclidean_distance(lh_right, lh_left)
    lv_distance = euclidean_distance(lv_top, lv_bottom)
    le_ratio = lh_distance / lv_distance

    return (re_ratio + le_ratio) / 2

# Extract eyes from the frame
def extract_eyes(frame, right_eye_coords, left_eye_coords):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(gray)
    cv2.fillPoly(mask, [np.array(right_eye_coords, dtype=np.int32)], 255)
    cv2.fillPoly(mask, [np.array(left_eye_coords, dtype=np.int32)], 255)
    eyes = cv2.bitwise_and(gray, gray, mask=mask)
    eyes[mask == 0] = 155

    r_min_x = min(right_eye_coords, key=lambda item: item[0])[0]
    r_max_x = max(right_eye_coords, key=lambda item: item[0])[0]
    r_min_y = min(right_eye_coords, key=lambda item: item[1])[1]
    r_max_y = max(right_eye_coords, key=lambda item: item[1])[1]
    l_min_x = min(left_eye_coords, key=lambda item: item[0])[0]
    l_max_x = max(left_eye_coords, key=lambda item: item[0])[0]
    l_min_y = min(left_eye_coords, key=lambda item: item[1])[1]
    l_max_y = max(left_eye_coords, key=lambda item: item[1])[1]

    cropped_right = eyes[r_min_y:r_max_y, r_min_x:r_max_x]
    cropped_left = eyes[l_min_y:l_max_y, l_min_x:l_max_x]

    return cropped_right, cropped_left

# Estimate eye position
def estimate_eye_position(cropped_eye):
    h, w = cropped_eye.shape
    blurred_eye = cv2.medianBlur(cropped_eye, 3)
    _, threshed_eye = cv2.threshold(blurred_eye, 130, 255, cv2.THRESH_BINARY)
    piece = int(w / 3)
    right_piece = threshed_eye[:, :piece]
    center_piece = threshed_eye[:, piece:2 * piece]
    left_piece = threshed_eye[:, 2 * piece:]
    return pixel_counter(right_piece, center_piece, left_piece)

# Euclidean distance
def euclidean_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# Pixel counter function
def pixel_counter(first_piece, second_piece, third_piece):
    right_part = np.sum(first_piece == 0)
    center_part = np.sum(second_piece == 0)
    left_part = np.sum(third_piece == 0)
    eye_parts = [right_part, center_part, left_part]
    max_index = eye_parts.index(max(eye_parts))
    if max_index == 0:
        return "RIGHT", (0, 255, 0)  # Right eye
    elif max_index == 1:
        return "CENTER", (255, 255, 0)  # Center eye
    else:
        return "LEFT", (0, 0, 255)  # Left eye

# Main function
def main():
    # Load Mediapipe face mesh model
    face_mesh = mp.solutions.face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Initialize video capture
    camera = initialize_capture()
    if camera is None:
        return

    frame_counter = 0
    total_blinks = 0
    cef_counter = 0
    display_drowsiness = False
    display_frames = 0  # Counter to keep track of frames for displaying drowsiness

    # Define eye indices
    RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

    # Main loop
    while True:
        ret, frame = camera.read()
        if not ret:
            break

        frame_counter += 1

        mesh_coords = detect_landmarks(frame, face_mesh)
        if mesh_coords:
            blink_ratio = calculate_blink_ratio(mesh_coords, RIGHT_EYE, LEFT_EYE)

            if blink_ratio > 5.5:
                cef_counter += 1
                if cef_counter > 6:
                    display_drowsiness = True
                    display_frames = 0
            else:
                if cef_counter > 3:
                    total_blinks += 1
                    cef_counter = 0

            cv2.putText(frame, f'Total Blinks: {total_blinks}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            right_coords = [mesh_coords[p] for p in RIGHT_EYE]
            left_coords = [mesh_coords[p] for p in LEFT_EYE]
            crop_right, crop_left = extract_eyes(frame, right_coords, left_coords)
            eye_position_right, color_right = estimate_eye_position(crop_right)
            eye_position_left, color_left = estimate_eye_position(crop_left)

            cv2.polylines(frame, [np.array(right_coords, dtype=np.int32)], True, (0, 255, 0), 1)
            cv2.polylines(frame, [np.array(left_coords, dtype=np.int32)], True, (0, 255, 0), 1)

            cv2.putText(frame, f'R: {eye_position_right}', (40, 220), cv2.FONT_HERSHEY_SIMPLEX, 1, color_right, 2)
            cv2.putText(frame, f'L: {eye_position_left}', (40, 320), cv2.FONT_HERSHEY_SIMPLEX, 1, color_left, 2)

            if display_drowsiness:
                cv2.putText(frame, "Drowsiness Detected", (10, 60), cv2.FONT_HERSHEY_SIMPLEX,  1, (0, 0, 255), 2)
                display_frames += 1
                if display_frames > 30:  # Display for 30 frames (1 second at 30fps)
                    display_drowsiness = False

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
