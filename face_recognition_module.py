import cv2
import face_recognition
import os
import numpy as np

class FaceRecognizer:
    def __init__(self, known_faces_dir="known_faces"):
        self.known_faces_dir = known_faces_dir
        self.known_face_encodings = []
        self.known_face_names = []
        self.load_known_faces()

    def load_known_faces(self):
        """Loads and encodes all faces found in the known_faces directory."""
        if not os.path.exists(self.known_faces_dir):
            os.makedirs(self.known_faces_dir)
            print(f"Created directory '{self.known_faces_dir}'. Please put images here to recognize.")
            return

        for filename in os.listdir(self.known_faces_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                filepath = os.path.join(self.known_faces_dir, filename)
                # Load image
                image = face_recognition.load_image_file(filepath)
                
                # Get encoding
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    encoding = encodings[0]
                    name = os.path.splitext(filename)[0]
                    self.known_face_encodings.append(encoding)
                    self.known_face_names.append(name)
                    print(f"Loaded face encoding for: {name}")
                else:
                    print(f"Warning: No face found in {filename}")

    def recognize_faces_from_webcam(self):
        """Starts webcam and recognizes faces in real-time. Yields the name of the recognized person."""
        video_capture = cv2.VideoCapture(0)

        process_this_frame = True

        try:
            while True:
                # Grab a single frame
                ret, frame = video_capture.read()
                if not ret:
                    continue

                if process_this_frame:
                    # Resize frame for faster processing
                    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                    # Convert BGR (OpenCV) to RGB (face_recognition) and ensure memory is contiguous
                    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                    # Find all faces & encodings in current frame
                    face_locations = face_recognition.face_locations(rgb_small_frame)
                    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                    face_names = []
                    for face_encoding in face_encodings:
                        # See if face matches a known face
                        matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                        name = "Unknown"

                        # Use the known face with the smallest distance to the new face
                        face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                        if len(face_distances) > 0:
                            best_match_index = np.argmin(face_distances)
                            if matches[best_match_index]:
                                name = self.known_face_names[best_match_index]

                        face_names.append(name)
                        
                        # In a non-GUI app, we might just print it or yield it
                        # Since we want Gemini to know who is sitting, we will yield or print
                        print(f"Detected: {name}")

                process_this_frame = not process_this_frame
                
                # For this isolated test, we can break after checking one face if needed
                # But since it's real-time, it just runs. User can press Ctrl+C to stop
                
        except KeyboardInterrupt:
            print("Stopping face recognition...")
        finally:
            video_capture.release()

if __name__ == "__main__":
    print("Testing Face Recognition Module...")
    recognizer = FaceRecognizer()
    recognizer.recognize_faces_from_webcam()
