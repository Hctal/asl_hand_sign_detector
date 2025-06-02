# ASL Hand Sign Detector

This project is a real-time American Sign Language (ASL) alphabet detector using computer vision and a trained CNN model. It uses your webcam to detect hand signs and displays the predicted letter live.

---

## 📁 Folder Structure

```
asl_hand_sign_detector/
├── model/
│   └── asl_model.h5          # Trained CNN model for ASL detection
├── dataset/
│   └── asl_alphabet_train/   # Training dataset (downloaded from Kaggle)
├── realtime_detection.py     # Real-time hand sign detection script
├── train_model.py            # CNN model training script
└── README.md                 # Project documentation
```

---

## 🧠 How It Works

1. A CNN is trained on the ASL alphabet dataset (87,000+ labeled hand sign images).
2. The model is saved as `asl_model.h5`.
3. In real-time, the webcam feed is processed using MediaPipe to detect and crop hand landmarks.
4. The cropped hand image is passed through the model.
5. The predicted letter is displayed on-screen.

---

## 📦 Requirements

* Python 3.7+
* TensorFlow
* OpenCV
* MediaPipe
* NumPy

Install all requirements:

```bash
pip install -r requirements.txt
```

(Optional) If you don’t have a requirements.txt yet, just install:

```bash
pip install tensorflow opencv-python mediapipe numpy
```

---

## 🔽 Dataset

Download the dataset from Kaggle:
[https://www.kaggle.com/datasets/grassknoted/asl-alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)

Extract it and place the `asl_alphabet_train/` folder inside `dataset/`.

---

## ▶️ How to Train the Model

```bash
python train_model.py
```

The trained model will be saved as `model/asl_model.h5`.

---

## 🟢 How to Run Real-Time Detection

```bash
python realtime_detection.py
```

* Make ASL signs in front of your webcam.
* The predicted letter will be shown in real-time.
* Press `q` to quit.

---

## 🔧 Optional Improvements

* Add buffering to form full words
* Build GUI with Tkinter or Streamlit
* Improve accuracy with more training epochs or a deeper model
* Convert to TensorFlow Lite and deploy on mobile

---

## 🧑‍💻 Author

* Built using Python, TensorFlow, OpenCV, and MediaPipe
