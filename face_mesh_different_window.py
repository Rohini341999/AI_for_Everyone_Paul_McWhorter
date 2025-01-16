import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
mp_Hand = mp.solutions.hands
# For webcam input:
drawing_spec = mp_drawing.DrawingSpec(thickness=3, circle_radius=2)
circle_drawing_spec = mp_drawing.DrawingSpec(
    thickness=2, circle_radius=1, color=(0, 0, 240)
)
circle_drawing_spec_hand = mp_drawing.DrawingSpec(
    thickness=4, circle_radius=4, color=(0, 0, 240)
)
handDrawingSpec = mp_drawing.DrawingSpec(
    thickness=5, circle_radius=2, color=(0, 0, 240)
)
line_drawing_spec = mp_drawing.DrawingSpec(thickness=5, color=(0, 0, 240))

cap = cv2.VideoCapture(0)
cap.set(3, 1330)
cap.set(4, 1080)
frameWidth = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
frameHeight = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

img = np.zeros((int(frameWidth), int(frameHeight), 3), dtype="uint8")
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5, min_tracking_confidence=0.5
)
hand_mesh = mp_Hand.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
while True:
    success, image = cap.read()
    img = np.zeros((int(frameHeight), int(frameWidth), 3), dtype="uint8")
    img.fill(0)
    img[:] = (0, 0, 0)

    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = face_mesh.process(image)
    results2 = hand_mesh.process(image)

    # Draw the face mesh annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_face_landmarks:
        for face_landmark in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=img,
                landmark_list=face_landmark,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=circle_drawing_spec,
                connection_drawing_spec=line_drawing_spec,
            )
        if results2.multi_hand_landmarks:
            for hand in results2.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    img,
                    hand,
                    mp_Hand.HAND_CONNECTIONS,
                    circle_drawing_spec_hand,
                    line_drawing_spec,
                )
        cv2.imshow("MediaPipe FaceMesh", img)
        cv2.moveWindow("MediaPipe FaceMesh", 0, 0)
        cv2.imshow("full", image)
        if cv2.waitKey(5) & 0xFF == ord("q"):
            break
cap.release()
