# Handwritten Digit Recognition (MNIST + Tkinter GUI)

This project implements a **Convolutional Neural Network (CNN)** using TensorFlow/Keras to recognize handwritten digits from the **MNIST dataset**.
It also provides a **Tkinter GUI app** where you can draw digits on a canvas and get real-time predictions from the trained model.

---

## 📂 Project Structure

```
handwritten-digit-recognition/
│── train_mnist.py        # Script to train CNN on MNIST and save model
│── digit_gui.py          # Tkinter app to draw & predict digits
│── mnist_cnn.h5          # Saved CNN model (generated after training)
│── training_history.png  # Loss/accuracy curves (generated after training)
│── requirements.txt      # Python dependencies
│── README.md             # Project documentation
```

---

## 🚀 Features

- Trains a **CNN** on the MNIST dataset (98%+ accuracy).
- Saves trained model as `mnist_cnn.h5`.
- Interactive **Tkinter GUI** to draw digits and get predictions.
- Displays **probabilities for each digit (0–9)**.
- Saves your drawn digit as an image.

---

## ⚙️ Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/<your-username>/handwritten-digit-recognition.git
   cd handwritten-digit-recognition
   ```

2. Create & activate a virtual environment (recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Mac/Linux
   .venv\Scripts\activate      # Windows
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

## 📊 Training the Model

Run the training script to train a CNN on MNIST:

```bash
python train_mnist.py
```

This will:

- Train the CNN
- Save the best model to `mnist_cnn.h5`
- Generate `training_history.png` with accuracy/loss curves

---

## 🎨 Running the GUI

After training, launch the GUI app:

```bash
python digit_gui.py
```

- Draw a digit on the canvas
- Click **Predict** to see results
- Click **Clear** to reset
- Click **Save Image** to export your drawing

---


## 📦 Requirements

- Python 3.8+
- TensorFlow / Keras
- NumPy
- Pillow
- Matplotlib
- Scikit-learn
- SciPy
- Tkinter (comes with Python)

Install all via:

```bash
pip install -r requirements.txt
```

---

## 📝 License

This project is open-source and available under the **MIT License**.
