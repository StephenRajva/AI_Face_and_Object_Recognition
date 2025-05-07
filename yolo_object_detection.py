import cv2
import numpy as np
import pyttsx3
import speech_recognition as sr
import face_recognition
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Initialize TTS
engine = pyttsx3.init()
engine.setProperty("rate", 150)

# Speech recognition
recognizer = sr.Recognizer()

# For recognizing known faces
known_faces = []
known_names = []

# To avoid repeating speech
spoken_objects = set()

# Function to ask name via voice or fallback to keyboard
def ask_name():
    print("🤖 AI: Hey, what's your name?")
    engine.say("Hey, what's your name?")
    engine.runAndWait()

    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print("🎤 Listening...")
        try:
            audio = recognizer.listen(source, timeout=8)
            name = recognizer.recognize_google(audio)
            print(f"🗣 You said: {name}")
        except (sr.UnknownValueError, sr.RequestError):
            name = input("⌨️ Enter your name: ")

    response = f"Hi {name}, nice to meet you!"
    print(f"🤖 AI: {response}")
    engine.say(response)
    engine.runAndWait()
    return name

# Start camera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if confidence > 0.6:
                label = model.names[class_id]
                color = (0, 255, 0) if label == "person" else (255, 0, 0)

                # Face recognition if it's a person
                if label == "person":
                    face = frame[y1:y2, x1:x2]
                    rgb_face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    face_encodings = face_recognition.face_encodings(rgb_face)

                    if face_encodings:
                        face_encoding = face_encodings[0]
                        matches = face_recognition.compare_faces(known_faces, face_encoding)
                        name = "Unknown"

                        if True in matches:
                            match_idx = matches.index(True)
                            name = known_names[match_idx]
                            message = f"I know you, {name}!"
                            print(f"🤖 AI: {message}")
                            engine.say(message)
                            engine.runAndWait()
                        else:
                            name = ask_name()
                            known_faces.append(face_encoding)
                            known_names.append(name)
                        label = name  # Update label to name

                else:
                    # Speak object name once
                    if label not in spoken_objects:
                        print(f"🧠 Object Detected: {label}")
                        engine.say(f"{label} detected")
                        engine.runAndWait()
                        spoken_objects.add(label)

                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("YOLOv8 Object & Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Esc to quit
        break

cap.release()
cv2.destroyAllWindows()
