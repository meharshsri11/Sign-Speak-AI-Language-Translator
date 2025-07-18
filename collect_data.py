import cv2
import mediapipe as mp
import csv
import os

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)

# Ask user for label
label = input("Enter the label for this sign (e.g., A, Hello): ")
save_path = f"sign_data/data_{label}.csv"

# Create folder if not exists
if not os.path.exists("sign_data"):
    os.makedirs("sign_data")

# Open CSV file to save data
f = open(save_path, 'w', newline='')
csv_writer = csv.writer(f)

# Open webcam
cap = cv2.VideoCapture(0)
print("[INFO] Press 's' to save frame, 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)

    # Show hand landmarks if detected
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Collect 21 x,y landmark points (42 values)
            row = []
            for lm in hand_landmarks.landmark:
                row.append(lm.x)
                row.append(lm.y)

            # Press 's' to save current hand frame
            key = cv2.waitKey(10) & 0xFF
            if key == ord('s'):
                csv_writer.writerow(row)
                print(f"[SAVED] Frame for '{label}'")

    # Show the webcam feed
    cv2.putText(frame, f"Sign: {label} | Press 's' to save, 'q' to quit", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.imshow("Sign Data Collection", frame)

    # Quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything
cap.release()
cv2.destroyAllWindows()
f.close()
print(f"✅ Data saved to {save_path}")
