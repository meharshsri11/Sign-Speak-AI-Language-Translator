# Sign-Speak-AI-Language-Translator

A real-time hand sign recognition system that detects sign language gestures using a webcam and translates them into text or voice, with optional multi-language support.

---

## 📸 Demo

_(Upload your video in `sign_videos/` and paste a Drive or GitHub link here)_

---

## 🎯 Features

- ✋ Detects hand gestures in real-time using webcam
- 🧠 Uses a trained KNN model to recognize hand signs
- 🗣️ Converts detected signs to speech
- 🌐 Translates text into other languages using Google Translate
- 🔁 Modular code: data collection, model training, sign prediction, translation

---

## 🛠️ Tech Stack

- **Language**: Python
- **ML**: scikit-learn (KNN)
- **Vision**: OpenCV, MediaPipe
- **Speech**: pyttsx3
- **Translation**: googletrans
- **Interface (optional)**: Streamlit / CLI

---

## 📂 Folder Structure

```bash
Sign-AI-Translator/
├── collect_data.py         # Collects hand landmark data
├── hand_tracking.py        # Uses mediapipe for detection
├── predict_sign.py         # Predicts signs using trained model
├── text_to_sign.py         # Converts detected sign to speech/text
├── translator_app.py       # Integrates translation functionality
├── train_model.py          # Trains KNN model
├── sign_model_knn.pkl      # Saved ML model
├── sign_data/              # Stored hand landmarks
├── sign_videos/            # Demo clips (optional)
├── requirements.txt        # Dependencies
└── README.md               # You're reading this!
```

---

## 🧪 How to Run

1. **Clone the repo**
```bash
git clone https://github.com/meharshsri11/Sign-AI-Translator.git
cd Sign-AI-Translator
```

2. **Install requirements**
```bash
pip install -r requirements.txt
```

3. **Collect Sign Data**
```bash
python collect_data.py
```

4. **Train the Model**
```bash
python train_model.py
```

5. **Run the Translator App**
```bash
python translator_app.py
```

---

## 📈 Future Enhancements

- Add support for more gestures
- Deploy as a web/mobile app
- Add GUI for easier interaction

---

## 🤝 Contributing

Pull requests are welcome! For major changes, open an issue first.

---

## 📄 License

MIT License

---

## 🙋‍♂️ Author

- **Harsh Srivastava** – [GitHub](https://github.com/meharshsri11)
