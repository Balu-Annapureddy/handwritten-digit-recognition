"""
digit_gui.py

- Tkinter app for drawing digits
- Loads saved CNN model 'mnist_cnn.h5'
- Converts drawn image to 28x28 grayscale, preprocesses and predicts
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import sys
import numpy as np
from PIL import Image, ImageOps, ImageDraw
import tkinter as tk
from tkinter import ttk, messagebox
import tensorflow as tf
from tensorflow import keras
model = keras.models.load_model("mnist_cnn.h5", compile=False)

# Suppress TensorFlow logs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

MODEL_PATH = "mnist_cnn.h5"

# checks model exist or not
if not os.path.exists(MODEL_PATH):
    msg = f"Model file '{MODEL_PATH}' not found. Please run train_mnist.py first to create it."
    print(msg)
    MODEL_AVAILABLE = False
else:
    MODEL_AVAILABLE = True
    from tensorflow import keras
    model = keras.models.load_model(MODEL_PATH, compile=False)

# GUI parameters
CANVAS_SIZE = 280 
PEN_WIDTH = 18

class DigitRecognizerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Handwritten Digit Recognizer (MNIST)")
        self.resizable(False, False)

        # top frame
        top = ttk.Frame(self, padding=10)
        top.grid(row=0, column=0)

        # Canvas
        self.canvas = tk.Canvas(top, width=CANVAS_SIZE, height=CANVAS_SIZE, bg='white', cursor='cross')
        self.canvas.grid(row=0, column=0, columnspan=4, pady=(0,10))
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset_last_pos)

        # Controls
        self.predict_btn = ttk.Button(top, text="Predict", command=self.predict)
        self.predict_btn.grid(row=1, column=0, padx=6, sticky="ew")

        self.clear_btn = ttk.Button(top, text="Clear", command=self.clear)
        self.clear_btn.grid(row=1, column=1, padx=6, sticky="ew")

        self.save_btn = ttk.Button(top, text="Save Image", command=self.save_canvas)
        self.save_btn.grid(row=1, column=2, padx=6, sticky="ew")

        self.quit_btn = ttk.Button(top, text="Quit", command=self.quit)
        self.quit_btn.grid(row=1, column=3, padx=6, sticky="ew")

        # Prediction label
        self.result_var = tk.StringVar(value="Draw a digit and click Predict")
        self.result_label = ttk.Label(top, textvariable=self.result_var, font=("Helvetica", 14))
        self.result_label.grid(row=2, column=0, columnspan=4, pady=(10,0))

        # Probability labels for digits
        self.prob_frame = ttk.Frame(top)
        self.prob_frame.grid(row=3, column=0, columnspan=4, pady=(10,0))

        self.prob_vars = [tk.StringVar(value="") for _ in range(10)]
        for i in range(10):
            lbl = ttk.Label(self.prob_frame, textvariable=self.prob_vars[i], width=10, anchor="center")
            lbl.grid(row=i//5, column=i%5, padx=3, pady=2)

        self.image1 = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=255)  # white bg
        self.draw = ImageDraw.Draw(self.image1)

        self.last_x, self.last_y = None, None

    def paint(self, event):
        x, y = event.x, event.y
        if self.last_x is None and self.last_y is None:
            self.last_x, self.last_y = x, y

        # draw on Tkinter canvas
        self.canvas.create_oval(x - PEN_WIDTH//2, y - PEN_WIDTH//2, x + PEN_WIDTH//2, y + PEN_WIDTH//2, fill="black", outline="black")
        # draw on PIL image
        self.draw.ellipse([x - PEN_WIDTH//2, y - PEN_WIDTH//2, x + PEN_WIDTH//2, y + PEN_WIDTH//2], fill=0)
        self.last_x, self.last_y = x, y

    def reset_last_pos(self, event=None):
        self.last_x, self.last_y = None, None

    def clear(self):
        self.canvas.delete("all")
        self.draw.rectangle([0,0,CANVAS_SIZE, CANVAS_SIZE], fill=255)
        self.result_var.set("Draw a digit and click Predict")
        for v in self.prob_vars:
            v.set("")

    def save_canvas(self):
        # save current drawing to a PNG
        path = "captured_digit.png"
        self.image1.save(path)
        messagebox.showinfo("Saved", f"Canvas saved to {path}")

    def preprocess_image(self, pil_img):
        """
        Preprocess canvas image to match MNIST style:
        - Invert colors
        - Crop digit
        - Resize keeping aspect ratio
        - Pad to 28x28
        - Center based on center of mass
        """
        from scipy import ndimage

        # Convert and invert
        img = pil_img.convert("L")
        img = ImageOps.invert(img)
        arr = np.array(img)

        # Threshold (binarize)
        arr = (arr > 50) * 255

        # Find bounding box of digit
        coords = np.column_stack(np.where(arr > 0))
        if coords.size == 0:
            return np.zeros((1,28,28,1), dtype="float32")
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        arr = arr[y0:y1+1, x0:x1+1]

        # Resize keeping aspect ratio (fit into 20x20)
        h, w = arr.shape
        if h > w:
            new_h = 20
            new_w = int(round((w * 20.0) / h))
        else:
            new_w = 20
            new_h = int(round((h * 20.0) / w))
        img = Image.fromarray(arr.astype(np.uint8))
        img = img.resize((new_w, new_h), Image.LANCZOS)

        # Pad to 28x28
        new_img = Image.new("L", (28,28), 0)
        paste_x = (28 - new_w) // 2
        paste_y = (28 - new_h) // 2
        new_img.paste(img, (paste_x, paste_y))

        # Convert to numpy array
        arr = np.array(new_img).astype("float32") / 255.0

        # Center using center of mass
        cy, cx = ndimage.center_of_mass(arr)
        shiftx = np.round(arr.shape[1]/2.0 - cx).astype(int)
        shifty = np.round(arr.shape[0]/2.0 - cy).astype(int)
        arr = ndimage.shift(arr, shift=(shifty, shiftx), mode='constant')

        # Reshape for model
        arr = np.expand_dims(arr, axis=-1)
        arr = np.expand_dims(arr, axis=0)
        return arr

    def predict(self):
        if not MODEL_AVAILABLE:
            messagebox.showerror("Model missing", f"Model '{MODEL_PATH}' not found. Run train_mnist.py first.")
            return

        # Use the in-memory PIL image
        pil_img = self.image1.copy()

        x = self.preprocess_image(pil_img)
        preds = model.predict(x, verbose=0)[0]  # shape (10,)
        top_digit = int(np.argmax(preds))
        prob = float(preds[top_digit])

        # update labels
        self.result_var.set(f"Predicted: {top_digit} (confidence {prob*100:.1f}%)")
        for i in range(10):
            self.prob_vars[i].set(f"{i}: {preds[i]*100:.2f}%")

if __name__ == "__main__":
    app = DigitRecognizerApp()
    app.mainloop()
