import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Start capturing video
cap = cv2.VideoCapture(0)

# Initialize previous point and eraser state
prev_point = None
eraser_active = False

# Create a blank canvas
canvas = np.zeros((480, 640, 3), dtype=np.uint8)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run hand tracking on the frame
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get landmark positions
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            # Calculate position
            index_x, index_y = int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0])
            thumb_x, thumb_y = int(thumb_tip.x * frame.shape[1]), int(thumb_tip.y * frame.shape[0])
            distance = np.sqrt((index_x - thumb_x) ** 2 + (index_y - thumb_y) ** 2)

            # Draw a circle at the finger position on the frame
            cv2.circle(frame, (index_x, index_y), 8, (0, 255, 0), -1)

            # Check if thumb is close to index finger to activate eraser
            if distance < 30:
                eraser_active = True
            else:
                eraser_active = False

            # Draw line or erase canvas if previous point exists
            if prev_point:
                if eraser_active:
                    canvas = np.zeros((480, 640, 3), dtype=np.uint8)  # Erase canvas
                else:
                    cv2.line(canvas, prev_point, (index_x, index_y), (0, 255, 0), 2)

            # Update previous point
            prev_point = (index_x, index_y)

    # Overlay canvas on the frame
    print(f"Frame shape: {frame.shape}")
    print(f"Canvas shape: {canvas.shape}")
    canvas = cv2.resize(canvas, (frame.shape[1], frame.shape[0]))
    frame = cv2.add(frame, canvas)
    canvas_flipped = cv2.flip(canvas, 1)
    frame_flipped = cv2.flip(frame, 1)

    # Display the frame
    cv2.imshow('Finger Tracking', frame_flipped)
    cv2.imshow('Canvas', canvas_flipped)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()