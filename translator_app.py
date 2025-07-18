import cv2
import mediapipe as mp
import numpy as np
import joblib
import pyttsx3
import speech_recognition as sr
import time
import os
from difflib import get_close_matches

# ========== SETUP ==========
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
engine = pyttsx3.init()

# Load trained model
model_path = "sign_model_knn.pkl"
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    print("❌ Model not found. Train it first.")
    exit()

# Dictionary: speech word → sign video
video_dict = {
    "hello": "sign_videos/hello.mp4",
    "ok": "sign_videos/ok.mp4",
    "water": "sign_videos/water.mp4"
}

# ========== FUNCTIONS ==========

# 🔁 Play a sign video
def play_video(path):
    cap = cv2.VideoCapture(path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        cv2.imshow("Voice ➡ Sign", frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# ✋ SIGN ➡ TEXT/SPEECH
def sign_to_text():
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
    cap = cv2.VideoCapture(0)
    spoken_sign = None

    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        prediction = ""

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y])

                if len(landmarks) == 42:
                    prediction = model.predict([landmarks])[0]
                    if prediction != spoken_sign:
                        print(f"🧠 Predicted: {prediction}")
                        engine.say(prediction)
                        engine.runAndWait()
                        spoken_sign = prediction

        cv2.putText(frame, f"Prediction: {prediction}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Sign ➡ Text/Speech", frame)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# 🎤 VOICE ➡ SIGN VIDEO
def voice_to_sign():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎤 Speak a word (e.g., hello, ok, water):")
        r.adjust_for_ambient_noise(source)
        try:
            audio = r.listen(source, timeout=5)
        except sr.WaitTimeoutError:
            print("⏱️ Timeout: No speech detected.")
            return

    try:
        word = r.recognize_google(audio).lower()
        print("🗣️ You said:", word)

        # Exact match
        if word in video_dict:
            play_video(video_dict[word])
        else:
            # Try closest match
            match = get_close_matches(word, video_dict.keys(), n=1)
            if match:
                print(f"🔁 Closest match: {match[0]}")
                play_video(video_dict[match[0]])
            else:
                print("❌ No matching sign video found.")

    except sr.UnknownValueError:
        print("❌ Could not understand your speech.")
    except sr.RequestError as e:
        print(f"❌ API error: {e}")

# ========== MAIN MENU ==========
def main():
    while True:
        print("\n📲 Smart Sign Language Translator")
        print("1. Sign ➡ Text/Speech")
        print("2. Voice ➡ Sign Video")
        print("3. Exit")

        choice = input("Select an option (1/2/3): ")

        if choice == "1":
            sign_to_text()
        elif choice == "2":
            voice_to_sign()
        elif choice == "3":
            print("👋 Exiting...")
            break
        else:
            print("❌ Invalid choice. Try again.")

# Run it
if __name__ == "__main__":
    main()
