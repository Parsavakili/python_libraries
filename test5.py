import cv2
import face_recognition
import numpy as np
import os

# Create a directory to store saved faces if not exists
os.makedirs("saved_faces", exist_ok=True)

# Load known faces and their encodings
known_face_encodings = []
known_face_names = []

# Load an image of a known person and encode it
known_image_path = "1.jpg"

if os.path.exists(known_image_path):  # Ensure file exists
    person_image = face_recognition.load_image_file(known_image_path)
    encodings = face_recognition.face_encodings(person_image)

    if encodings:  # Ensure at least one face is found
        known_face_encodings.append(encodings[0])
        known_face_names.append("Known Person")
    else:
        print(f"Error: No face found in '{known_image_path}'. Check the image.")
else:
    print(f"Error: File '{known_image_path}' not found.")

# Initialize webcam
video_capture = cv2.VideoCapture(0)  # Use 0 for default webcam

# Check if webcam opened successfully
if not video_capture.isOpened():
    print("Error: Could not access webcam. Check permissions.")
    exit()

# Initialize variables
face_count = 0  # Counter for naming saved face images
process_this_frame = True

while True:
    ret, frame = video_capture.read()

    if not ret or frame is None:
        print("Failed to capture frame from webcam.")
        break

    # Resize frame to 1/4 size for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = np.array(small_frame[:, :, ::-1], dtype=np.uint8)  # Convert BGR to RGB

    if process_this_frame:
        face_locations = face_recognition.face_locations(rgb_small_frame)

        if face_locations:  # Only proceed if faces are found
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        else:
            face_encodings = []

        face_names = []
        for face_encoding, face_location in zip(face_encodings, face_locations):
            name = "Unknown"

            if known_face_encodings:  # Ensure we have known faces to compare
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

                if any(matches):
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]

            face_names.append(name)

            # Scale back face location since the frame was resized earlier
            top, right, bottom, left = [x * 4 for x in face_location]

            # Ensure bounding box is within frame limits
            h, w, _ = frame.shape
            top, right, bottom, left = max(0, top), min(w, right), min(h, bottom), max(0, left)

            # Save face image
            if top < bottom and left < right:  # Ensure valid cropping dimensions
                face_image = frame[top:bottom, left:right]
                face_filename = f"saved_faces/face_{face_count}.jpg"
                cv2.imwrite(face_filename, face_image)
                print(f"Face saved: {face_filename}")
                face_count += 1

    process_this_frame = not process_this_frame

    # Draw bounding boxes and labels
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top, right, bottom, left = [x * 4 for x in (top, right, bottom, left)]

        # Draw a rectangle around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting frame
    cv2.imshow('Webcam Face Recognition', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
video_capture.release()
cv2.destroyAllWindows()
