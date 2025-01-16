# import cv2
# import mediapipe as mp

# mp_face_detection = mp.solutions.face_detection
# FDetect=mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
# mp_drawing = mp.solutions.drawing_utils

# # For webcam input:
# cap = cv2.VideoCapture(0)
# cap.set(3,1280)
# cap.set(4,720)

# while True:
#     _,image = cap.read()
#     results = FDetect.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#     if results.detections:
#         for detection in results.detections:
#             mp_drawing.draw_detection (image, detection)
#     cv2.imshow('MediaPipe Face Detection', image)
#     cv2.moveWindow('MediaPipa Face Detection',0,0)
#     if cv2.waitKey(5) & 0xFF == ord('q'):
#         break

import cv2
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
FDetect = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# For webcam input:
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Set width
cap.set(4, 720)  # Set height

while True:
  _, image = cap.read()
  results = FDetect.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

  # Draw detections on the frame
  if results.detections:
    for detection in results.detections:
      mp_drawing.draw_detection(image, detection)

  # Display the frame with detections
  cv2.imshow('MediaPipe Face Detection', image)

  # Move the window after showing it (ensure it exists first)
  cv2.moveWindow('MediaPipe Face Detection', 0, 0)

  # Wait for a key press and handle quit (increase wait time to avoid premature exit)
  if cv2.waitKey(10) & 0xFF == ord('q'):
    break

# Release resources
cap.release()
cv2.destroyAllWindows()