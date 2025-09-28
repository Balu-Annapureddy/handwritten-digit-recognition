"""
train_mnist.py

- Loads MNIST
- Preprocesses images
- Builds a CNN (Keras)
- Trains and evaluates
- Saves model to 'mnist_cnn.h5'
"""

import os
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Reproducibility
seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)

# Load MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize and reshape
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32")  / 255.0

# Add channel dimension (rows, cols, 1)
x_train = np.expand_dims(x_train, -1)  # shape (60000, 28, 28, 1)
x_test  = np.expand_dims(x_test, -1)

num_classes = 10
y_train_cat = keras.utils.to_categorical(y_train, num_classes)
y_test_cat  = keras.utils.to_categorical(y_test, num_classes)

# Model
def build_model(input_shape=(28,28,1), num_classes=10):
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(32, kernel_size=3, activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, kernel_size=3, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(128, kernel_size=3, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="mnist_cnn")
    return model

model = build_model()
model.compile(
    loss="categorical_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    metrics=["accuracy"]
)

model.summary()

# Callbacks
checkpoint_path = "mnist_cnn.h5"
callbacks = [
    keras.callbacks.ModelCheckpoint(checkpoint_path, monitor="val_accuracy",
                                    save_best_only=True, verbose=1),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=1)
]

# Train
history = model.fit(
    x_train, y_train_cat,
    batch_size=128,
    epochs=20,
    validation_split=0.1,
    callbacks=callbacks,
    verbose=2
)

# Evaluate
print("\nLoading best weights from checkpoint (if saved) and evaluating on test set...")
if os.path.exists(checkpoint_path):
    model = keras.models.load_model(checkpoint_path)

test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=0)
print(f"\nTest accuracy: {test_acc:.4f}  |  Test loss: {test_loss:.4f}")

# Predictions and classification report
y_pred_probs = model.predict(x_test)
y_pred = np.argmax(y_pred_probs, axis=1)

print("\nClassification report:")
print(classification_report(y_test, y_pred, digits=4))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix (rows=true labels, cols=predicted labels):")
print(cm)

# Optionally show training curves
def plot_history(h):
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(h.history['loss'], label='train_loss')
    plt.plot(h.history['val_loss'], label='val_loss')
    plt.legend(); plt.title('Loss')
    plt.subplot(1,2,2)
    plt.plot(h.history['accuracy'], label='train_acc')
    plt.plot(h.history['val_accuracy'], label='val_acc')
    plt.legend(); plt.title('Accuracy')
    plt.tight_layout()
    plt.savefig("training_history.png")
    print("Saved training_history.png")

try:
    plot_history(history)
except Exception as e:
    print("Could not plot/save history:", e)

print("\nModel saved to:", checkpoint_path)
