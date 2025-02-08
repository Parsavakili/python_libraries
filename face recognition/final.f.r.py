import cv2
import face_recognition
import numpy as np
import os
import time
from collections import defaultdict

# Constants
MAX_SAMPLES = 5
FACE_MATCH_THRESHOLD = 0.5
BLUR_THRESHOLD = 1000
PROCESSING_SCALE = 0.5
DETECTION_MODEL = 'hog'

# Get script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Ensure folders exist in the script directory
KNOWN_FACES_DIR = os.path.join(SCRIPT_DIR, "known_faces")
UNKNOWN_FACES_DIR = os.path.join(SCRIPT_DIR, "unknown_faces")

os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
os.makedirs(UNKNOWN_FACES_DIR, exist_ok=True)

def calculate_sharpness(image):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    return cv2.Laplacian(gray, cv2.CV_64F).var()

class FaceRecognizer:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.pending_faces = defaultdict(list)
        self.load_known_faces()

    def load_known_faces(self):
        self.known_face_encodings = []
        self.known_face_names = []
        
        for person_name in os.listdir(KNOWN_FACES_DIR):
            person_dir = os.path.join(KNOWN_FACES_DIR, person_name)
            if not os.path.isdir(person_dir):
                continue
                
            encodings = []
            for filename in os.listdir(person_dir):
                if filename.endswith(('.jpg', '.png')):
                    path = os.path.join(person_dir, filename)
                    image = face_recognition.load_image_file(path)
                    face_encs = face_recognition.face_encodings(image)
                    if face_encs:
                        encodings.append(face_encs[0])
            
            if encodings:
                avg_encoding = np.mean(encodings, axis=0)
                self.known_face_encodings.append(avg_encoding)
                self.known_face_names.append(person_name)

    def process_frame(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=PROCESSING_SCALE, fy=PROCESSING_SCALE)
        rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

        face_locations = face_recognition.face_locations(rgb_small_frame, model=DETECTION_MODEL)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        return face_locations, face_encodings

    def recognize_faces(self, face_encodings):
        face_names = []
        for face_encoding in face_encodings:
            if not self.known_face_encodings:  
                face_names.append("Unknown")
                continue

            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=FACE_MATCH_THRESHOLD)
            name = "Unknown"
            
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            
            if len(face_distances) > 0:  
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]
            
            face_names.append(name)
        return face_names

    def handle_new_face(self, frame, face_location):
        top, right, bottom, left = [x * int(1 / PROCESSING_SCALE) for x in face_location]
        face_image = frame[top:bottom, left:right]

        sharpness = calculate_sharpness(face_image)
        if sharpness < BLUR_THRESHOLD:
            return False  

        if not self.pending_faces.get('best_image'):
            self.pending_faces['best_image'] = face_image
            self.pending_faces['best_sharpness'] = sharpness
        else:
            if sharpness > self.pending_faces['best_sharpness']:
                self.pending_faces['best_image'] = face_image
                self.pending_faces['best_sharpness'] = sharpness
        
        self.pending_faces['location'] = face_location
        return True

    def finalize_new_face(self, name):
        if self.pending_faces.get('best_image') is None or self.pending_faces['best_image'].size == 0:
            return

        save_dir = os.path.join(KNOWN_FACES_DIR, name)
        os.makedirs(save_dir, exist_ok=True)
        filename = f"{name}_{int(time.time())}.jpg"
        cv2.imwrite(os.path.join(save_dir, filename), self.pending_faces['best_image'])

        loaded_image = face_recognition.load_image_file(os.path.join(save_dir, filename))
        face_enc = face_recognition.face_encodings(loaded_image)
        if face_enc:
            self.known_face_encodings.append(face_enc[0])
            self.known_face_names.append(name)
        
        self.pending_faces.clear()

recognizer = FaceRecognizer()
video_capture = cv2.VideoCapture(0)
last_face_time = time.time()
current_name = ""
input_active = False

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    face_locations, face_encodings = recognizer.process_frame(frame)
    face_names = recognizer.recognize_faces(face_encodings)

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top, right, bottom, left = [x * int(1 / PROCESSING_SCALE) for x in (top, right, bottom, left)]
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.rectangle(frame, (left, bottom + 20), (right, bottom), color, cv2.FILLED)
        cv2.putText(frame, name, (left + 3, bottom + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)  # Dark font

    for i, name in enumerate(face_names):
        if name == "Unknown" and not input_active:
            if recognizer.handle_new_face(frame, face_locations[i]):
                last_face_time = time.time()
                input_active = True

    if input_active:
        elapsed = time.time() - last_face_time
        remaining = max(15 - elapsed, 0)

        cv2.rectangle(frame, (0, 0), (400, 60), (220, 220, 220), cv2.FILLED)  
        cv2.putText(frame, f"Enter name ({remaining:.1f}s):", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)  
        cv2.putText(frame, current_name, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)  

        if elapsed >= 15:
            recognizer.finalize_new_face("unknown")
            current_name = ""
            input_active = False

    cv2.putText(frame, "Press Q to quit", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)  

    cv2.imshow('Face Recognition', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if input_active:
        if key == 13:  
            if current_name:
                recognizer.finalize_new_face(current_name)
            current_name = ""
            input_active = False
        elif key == 8:  
            current_name = current_name[:-1]
        elif 32 <= key <= 126:
            current_name += chr(key)
    if key == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
