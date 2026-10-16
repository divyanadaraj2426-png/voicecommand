# Voice-Controlled Smart Home Simulator

A Python-based smart home automation prototype that uses **Speech Recognition** and **Machine Learning (NLP)** via Google Colab to control virtual appliances (like fans and lights) using voice commands.

## 🚀 Features

* **Voice Input Capture:** Uses browser audio recording within Google Colab to capture spoken commands.
* **Speech-to-Text:** Integrates `SpeechRecognition` to convert voice audio into text.
* **Intent Classification:** Uses `scikit-learn` (TF-IDF Vectorizer and Multinomial Naive Bayes classifier) to map natural language variations to precise appliance commands.
* **Smart Home Status Tracker:** Simulates appliance states and prints real-time updates for connected devices.

---

## 🛠️ Tech Stack & Libraries

* **Python 3**
* **scikit-learn** (Machine Learning model for command classification)
* **SpeechRecognition** (Google Web Speech API wrapper)
* **pydub & IPython** (Audio processing and browser-to-Python audio pipeline)

---

## 📋 Code Overview

### 1. Install Dependencies

```bash
!pip install -q scikit-learn SpeechRecognition pydub

```

### 2. Train the NLP Intent Model

The script trains a Naive Bayes text classifier using a set of training phrases to understand various ways of giving commands:

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

commands = [
    "turn on fan", "switch on fan", "fan on",
    "turn off fan", "switch off fan", "fan off",
    "turn on light", "light on",
    "turn off light", "light off",
    "turn on all", "turn off all"
]
labels = [
    "fan_on", "fan_on", "fan_on",
    "fan_off", "fan_off", "fan_off",
    "light_on", "light_on",
    "light_off", "light_off",
    "all_on", "all_off"
]

model = Pipeline([
    ("vectorizer", TfidfVectorizer()),
    ("classifier", MultinomialNB())
])
model.fit(commands, labels)

```

### 3. Voice Recording & Control Loop

The project uses custom JavaScript execution inside Colab cells to record audio from your microphone, convert it to a WAV format in Python, transcribe it, and trigger the corresponding appliance status change.

---

## ⚙️ How to Run

1. Open the [automaticlight Colab Notebook](https://colab.research.google.com/drive/1H2j0uNkyIvE-Ch6jJbS1arOYZ5z2lKc7).
2. Connect to a hosted runtime (Python 3 Google Compute Engine backend).
3. Run all cells sequentially using `Runtime > Run all` or by executing them cell-by-cell.
4. When prompted by `run_voice_command()`, allow microphone permissions in your browser and speak a command (e.g., *"Turn off fan"*) within the 5-second window.
