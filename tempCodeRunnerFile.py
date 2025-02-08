
while True:
    # Grab a single frame from the webcam
    ret, frame = video_capture.read()
    
    if not ret or frame is None:
        print("Failed to capture frame from webcam.")
        break

    # Resize frame to 1/4 size for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]  # Convert BGR to RGB
    
    # Process every other frame to optimize performance
    if process_this_frame:
        if face_locations:
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        else:
            face_encodings = []  
        face_names = []
        for face_encoding, face_location in zip(face_encodings, face_locations):
            name = "Unknown"

            if known_face_encodings:  # Ensure we have known faces to compare
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                
                if matches and len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]

            face_names.append(name)

            # Scale back up face locations since the frame was resized earlier
            top, right, bottom, left = [x * 4 for x in face_location]

            # Ensure face bounding box is within frame limits
            h, w, _ = frame.shape
            top = max(0, top)
            right = min(w, right)
            bottom = min(h, bottom)
            left = max(0, left)

            if top < bottom and left < right:  # Ensure valid cropping dimensions
                face_image = frame[top:bottom, left:right]
                face_filename = f"saved_faces/face_{face_count}.jpg"
                cv2.imwrite(face_filename, face_image)
                print(f"Face saved: {face_filename}")
                face_count += 1  # Increment counter

    process_this_frame = not process_this_frame

    # Draw bounding boxes and labels
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a rectangle around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Webcam Face Recognition', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
video_capture.release()
cv2.destroyAllWindows()
