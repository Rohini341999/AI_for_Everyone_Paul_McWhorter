import cv2
import mediapipe as mp

# Initialize drawing utilities and face mesh/hand solutions
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

# Configure drawing specifications
drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=1)
circleDrawingSpec = mp_drawing.DrawingSpec(thickness=2, circle_radius=1, color=(0, 255, 0))
circleDrawingSpecHand = mp_drawing.DrawingSpec(thickness=2, circle_radius=1, color=(0, 0, 255))
handDrawingSpec = mp_drawing.DrawingSpec(thickness=2, circle_radius=1, color=(0, 0, 255))
lineDrawingSpec = mp_drawing.DrawingSpec(thickness=2, color=(0, 0, 255))

# Set up webcam capture
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Set width
cap.set(4, 720)  # Set height
frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # Get frame width (corrected variable name)

with mp_face_mesh.FaceMesh(
        max_num_faces=1,  # Limit to one face for better performance
        refine_landmarks=True,
        min_detection_confidence=0.5) as face_mesh:

    with mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:

        while True:
            # Capture frame-by-frame
            success, image = cap.read()
            if not success:
                print("Error: Unable to capture frame")
                break

            # Convert image to RGB for face mesh processing
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Process the image with face mesh and hands detection
            results_face_mesh = face_mesh.process(image)
            results_hands = hands.process(image)

            # Draw face mesh landmarks if detections are found
            if results_face_mesh.multi_face_landmarks:
                for face_landmarks in results_face_mesh.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=image, landmark_list=face_landmarks, connections=mp_face_mesh.FACEMESH_TESSELATION,
                        connection_drawing_spec=lineDrawingSpec, landmark_drawing_spec=circleDrawingSpec
                    )

            # Draw hand landmarks if detections are found
            if results_hands.multi_hand_landmarks:
                for hand_landmarks in results_hands.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image=image, landmark_list=hand_landmarks, connections=mp_hands.HAND_CONNECTIONS,
                        connection_drawing_spec=lineDrawingSpec, landmark_drawing_spec=circleDrawingSpecHand
                    )

            # Flip the image horizontally for a more natural view (optional)
            image = cv2.flip(image, 1)

            # Display the resulting frame with face mesh and hand landmarks overlaid
            cv2.imshow('MediaPipe Face and Hand Mesh', image)

            # Handle quitting with 'q' key
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

# Release resources
cap.release()
# cv2.destroyAllWindows()
