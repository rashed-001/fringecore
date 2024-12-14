import cv2
import mediapipe as mp

# install opencv-python and also mediapipe if not installed in your machine
# you can simply pip install them and chill
# you can run the code by writing python handraisedfringecore.py in the terminal
# or download the code, run the code into your ide.
# I have downloaded your video and then detect the things here.
# I am uploading the vide into my drive and add the link into the video
# happy coding
# Initialize Mediapipe Hands and Drawing Utilities

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Define Desk Positions and Names to detect them by their desk and also by their names
# Map hand coordinates to person name.

def get_name_from_coordinates(x, y, width, height):
    
    desk_regions = [
        (0, width // 7, "Tanvir"),     # Desk 1
        (width // 7, 2 * width // 7, "Shafayet"), # Desk 2
        (2 * width // 7, 3 * width // 7, "Toufiq"), # Desk 3
        (3 * width // 7, 4 * width // 7, "Mufrad"), # Desk 4
        (4 * width // 7, 5 * width // 7, "Imran"), # Desk 5
        (5 * width // 7, 6 * width // 7, "Emon"), # Desk 6
        (6 * width // 7, width, "Anik") # Desk 7
    ]
    for start_x, end_x, name in desk_regions:
        if start_x <= x < end_x:
            return name
    return "Unknown"

# Lets process the Video and detect raised hands and then identify person
def detect_hand_raised(video_path):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert frame to RGB 
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect the raised hands
            results = hands.process(rgb_frame)

            # Draw the landmarks to detect and  detect raised hands
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Get wrist (landmark 0) and middle fingertip (landmark 12)
                    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                    fingertip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

                    # Convert to pixel coordinates
                    wrist_y = int(wrist.y * height)
                    fingertip_y = int(fingertip.y * height)
                    wrist_x = int(wrist.x * width)

                    # Check if hand is raised (fingertip higher than wrist)
                    if fingertip_y < wrist_y:
                        name = get_name_from_coordinates(wrist_x, wrist_y, width, height)
                        print(f"Hand raised by {name}")

                    # Draw hand landmarks
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Display the video
            cv2.imshow('Hand Raised Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

# Provide the video path here

detect_hand_raised("https://drive.google.com/file/d/1udzAZedKYIwAyPJgyGL9X4qDgRw1ws0r/view?usp=sharing")
