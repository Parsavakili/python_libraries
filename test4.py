import cv2
import face_recognition
import numpy as np
import os

# Create directories for storing known and unknown faces
os.makedirs("known_faces", exist_ok=True)
os.makedirs("unknown_faces", exist_ok=True)

# Load known faces and their encodings
known_face_encodings = []
known_face_names = []

# Load known faces from "known_faces" directory
def load_known_faces():
    global known_face_encodings, known_face_names
    for filename in os.listdir("known_faces"):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            path = os.path.join("known_faces", filename)
            person_image = face_recognition.load_image_file(path)
            encodings = face_recognition.face_encodings(person_image)
            if encodings:
                known_face_encodings.append(encodings[0])
                known_face_names.append(os.path.splitext(filename)[0])  # Use filename as name
            else:
                print(f"Warning: No face found in '{filename}'")

# Load known faces initially
load_known_faces()

# Initialize webcam
video_capture = cv2.VideoCapture(0)
if not video_capture.isOpened():
    print("Error: Could not access webcam.")
    exit()

# Initialize variables
face_count = 0  # Counter for naming saved face images
process_this_frame = True

while True:
    ret, frame = video_capture.read()
    if not ret or frame is None:
        print("Failed to capture frame from webcam.")
        break

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = np.array(small_frame[:, :, ::-1], dtype=np.uint8)

    if process_this_frame:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations) if face_locations else []
        face_names = []

        for face_encoding, face_location in zip(face_encodings, face_locations):
            name = "Unknown"
            save_directory = "unknown_faces"
            
            if known_face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                if any(matches):
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]
                        save_directory = "known_faces"

            face_names.append(name)

            # Scale back face location to full frame size
            top, right, bottom, left = [x * 4 for x in face_location]
            h, w, _ = frame.shape
            top, right, bottom, left = max(0, top), min(w, right), min(h, bottom), max(0, left)
            
            if top < bottom and left < right:
                face_image = frame[top:bottom, left:right]
                face_filename = f"{save_directory}/face_{face_count}.jpg"
                cv2.imwrite(face_filename, face_image)
                print(f"Face saved: {face_filename}")
                face_count += 1

    process_this_frame = not process_this_frame

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top, right, bottom, left = [x * 4 for x in (top, right, bottom, left)]
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    cv2.imshow('Webcam Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
