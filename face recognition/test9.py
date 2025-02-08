import cv2
import face_recognition
import numpy as np
import os
import time
from collections import defaultdict

# Configuration
MAX_SAMPLES = 5              # Number of samples to capture for new faces
FACE_MATCH_THRESHOLD = 0.5   # Lower is more strict (default 0.6)
BLUR_THRESHOLD = 1000         # Higher means sharper images (Laplacian var)
PROCESSING_SCALE = 0.5       # Balance between speed and accuracy
DETECTION_MODEL = 'hog'      # 'cnn' more accurate but slower

# Create directories if they don't exist
os.makedirs("known_faces", exist_ok=True)
os.makedirs("unknown_faces", exist_ok=True)

def calculate_sharpness(image):
    """Measure image sharpness using Laplacian variance"""
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
        """Load and average encodings for each person"""
        self.known_face_encodings = []
        self.known_face_names = []
        
        for person_name in os.listdir("known_faces"):
            person_dir = os.path.join("known_faces", person_name)
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
                # Store average encoding for better accuracy
                avg_encoding = np.mean(encodings, axis=0)
                self.known_face_encodings.append(avg_encoding)
                self.known_face_names.append(person_name)

    def process_frame(self, frame):
        """Main processing pipeline for each frame"""
        small_frame = cv2.resize(frame, (0, 0), fx=PROCESSING_SCALE, 
                               fy=PROCESSING_SCALE)
        rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

        # Detect faces using specified model
        face_locations = face_recognition.face_locations(
            rgb_small_frame, model=DETECTION_MODEL
        )
        
        face_encodings = face_recognition.face_encodings(
            rgb_small_frame, face_locations
        )

        return face_locations, face_encodings

    def recognize_faces(self, face_encodings):
        """Match faces against known database"""
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(
                self.known_face_encodings, face_encoding,
                tolerance=FACE_MATCH_THRESHOLD
            )
            name = "Unknown"
            
            face_distances = face_recognition.face_distance(
                self.known_face_encodings, face_encoding
            )
            best_match_index = np.argmin(face_distances)
            
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
            
            face_names.append(name)
        return face_names

    def handle_new_face(self, frame, face_location):
        """Capture and process new face samples"""
        top, right, bottom, left = [x * int(1/PROCESSING_SCALE) 
                                  for x in face_location]
        face_image = frame[top:bottom, left:right]
        
        if calculate_sharpness(face_image) < BLUR_THRESHOLD:
            return False  # Skip blurry images
        
        self.pending_faces['samples'].append(face_image)
        self.pending_faces['location'] = face_location
        return True

    def finalize_new_face(self, name):
        """Save and register new face after capturing samples"""
        if not self.pending_faces.get('samples'):
            return

        # Save best samples and create average encoding
        encodings = []
        for sample in self.pending_faces['samples'][:MAX_SAMPLES]:
            save_dir = os.path.join("known_faces", name)
            os.makedirs(save_dir, exist_ok=True)
            filename = f"{name}_{int(time.time())}.jpg"
            cv2.imwrite(os.path.join(save_dir, filename), sample)
            
            # Generate encoding from saved image
            loaded_image = face_recognition.load_image_file(
                os.path.join(save_dir, filename)
            )
            face_enc = face_recognition.face_encodings(loaded_image)
            if face_enc:
                encodings.append(face_enc[0])

        if encodings:
            avg_encoding = np.mean(encodings, axis=0)
            self.known_face_encodings.append(avg_encoding)
            self.known_face_names.append(name)
        
        self.pending_faces.clear()

# Initialize system
recognizer = FaceRecognizer()
video_capture = cv2.VideoCapture(0)
last_face_time = time.time()
current_name = ""
input_active = False

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Process frame
    face_locations, face_encodings = recognizer.process_frame(frame)
    face_names = recognizer.recognize_faces(face_encodings)

    # Draw UI elements
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top, right, bottom, left = [x * int(1/PROCESSING_SCALE) 
                                  for x in (top, right, bottom, left)]
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.rectangle(frame, (left, bottom + 20), (right, bottom), color, cv2.FILLED)
        cv2.putText(frame, name, (left + 3, bottom + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Handle unknown faces
    for i, name in enumerate(face_names):
        if name == "Unknown" and not input_active:
            if recognizer.handle_new_face(frame, face_locations[i]):
                last_face_time = time.time()
                input_active = True

    # Input handling
    if input_active:
        elapsed = time.time() - last_face_time
        remaining = max(15 - elapsed, 0)
        
        # Draw input UI
        cv2.rectangle(frame, (0, 0), (400, 60), (30, 30, 30), cv2.FILLED)
        cv2.putText(frame, f"Enter name ({remaining:.1f}s):", (10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        cv2.putText(frame, current_name, (10, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Check timeout
        if elapsed >= 15:
            recognizer.finalize_new_face("unknown")
            current_name = ""
            input_active = False

    # Always show exit tip
    cv2.putText(frame, "Press Q to quit", (10, frame.shape[0] - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

    cv2.imshow('Face Recognition', frame)
    
    # Keyboard input
    key = cv2.waitKey(1) & 0xFF
    if input_active:
        if key == 13:  # Enter
            if current_name:
                recognizer.finalize_new_face(current_name)
            current_name = ""
            input_active = False
        elif key == 8:  # Backspace
            current_name = current_name[:-1]
        elif 32 <= key <= 126:
            current_name += chr(key)
    if key == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()