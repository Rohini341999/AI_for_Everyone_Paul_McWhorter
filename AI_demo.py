import cv2

# Create a video capture object to access the webcam
myCam = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = myCam.read()

    # Check if the frame was successfully read
    if not ret:
        print("Error: Unable to capture frame")
        break

    # Display the frame using cv2.imshow
    cv2.imshow("My WebCam", frame)

    # Move the window to the top-left corner (0, 0)
    cv2.moveWindow("My WebCam", 0, 0)

    # Check if 'q' key is pressed to quit
    if cv2.waitKey(1) == ord("q"):
        break

# Release the webcam resource
myCam.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
