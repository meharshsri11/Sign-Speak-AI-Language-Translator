import speech_recognition as sr
import cv2
import time

video_dict = {
    "hello": "sign_videos/hello.mp4",
    "ok": "sign_videos/ok.mp4",
    "water": "sign_videos/water.mp4"
}

# 🔊 Listen from mic
r = sr.Recognizer()
with sr.Microphone() as source:
    print("🎤 Speak a word:")
    r.adjust_for_ambient_noise(source)
    audio = r.listen(source)

try:
    word = r.recognize_google(audio).lower()
    print("You said:", word)

    if word in video_dict:
        cap = cv2.VideoCapture(video_dict[word])

        while True:
            ret, frame = cap.read()

            if not ret:
                print("Replaying...")
                time.sleep(1)
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            cv2.imshow("Sign Language", frame)

            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
    else:
        print("❌ No video found for:", word)

except Exception as e:
    print("Speech error:", e)
