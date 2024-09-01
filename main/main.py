import cv2
import dlib
from scipy.spatial import distance
from imutils import face_utils

# Function to calculate the eye aspect ratio (EAR)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (1.5 * C)
    return ear

# Thresholds and frame count
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 20
COUNTER = 0

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./main/shape_predictor_68_face_landmarks.dat")

# Grab the indexes of the facial landmarks for the left and right eye
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Start the video stream
video_stream = cv2.VideoCapture(0)

while True:
    # Read a frame from the video
    ret, frame = video_stream.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = detector(gray)

    for face in faces:
        # Get the facial landmarks
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        # Extract the left and right eye coordinates
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        # Calculate the EAR for both eyes
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        # Average the EAR
        ear = (leftEAR + rightEAR) / 2.0

        # Draw the contours of the eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # Check if EAR is below the threshold
        if ear < EYE_AR_THRESH:
            COUNTER += 1

            # If the eyes were closed for a sufficient number of frames, alert
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                cv2.putText(frame, "SLEEPING ALERT!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            COUNTER = 0

    # Display the frame
    cv2.imshow("Sleep Detection", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video stream and close windows
video_stream.release()
cv2.destroyAllWindows()
