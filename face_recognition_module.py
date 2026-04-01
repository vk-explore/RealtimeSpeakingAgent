import cv2
import face_recognition
import os
import numpy as np
import threading
import time

class FaceRecognizer:
    def __init__(self, known_faces_dir="known_faces"):
        self.known_faces_dir = known_faces_dir
        self.known_face_encodings = []
        self.known_face_names = []
        self.running = False
        self.thread = None
        self.last_seen_person = None
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

    def start_background_recognition(self, callback):
        """Starts webcam face recognition in a background thread and triggers callback on change."""
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._recognition_loop, args=(callback,), daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()

    def recognize_frame(self, frame):
        """Processes a single BGR frame, returning the primary person found."""
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        current_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "unknown"

            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]

            current_names.append(name)
        
        # "none" = no face, "unknown" = unrecognized face, otherwise the person's name
        if not current_names:
            return "none"
        known_names = [n for n in current_names if n != "unknown"]
        return known_names[0] if known_names else "unknown"

    def _recognition_loop(self, callback):
        video_capture = cv2.VideoCapture(0)
        process_this_frame = True

        try:
            while self.running:
                ret, frame = video_capture.read()
                if not ret:
                    time.sleep(0.1)
                    continue

                if process_this_frame:
                    primary_person = self.recognize_frame(frame)
                    
                    if primary_person != self.last_seen_person:
                        self.last_seen_person = primary_person
                        callback(primary_person)

                process_this_frame = not process_this_frame

        finally:
            video_capture.release()

if __name__ == "__main__":
    print("Testing Background Face Recognition Module...")
    recognizer = FaceRecognizer()
    
    def on_face_changed(name):
        print(f"\n[EVENT] Primary person changed to: {name}")
        
    recognizer.start_background_recognition(on_face_changed)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        recognizer.stop()
        print("Stopped.")
