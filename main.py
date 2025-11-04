import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Flatten, Dense
import numpy as np
import time
from utils import save_model_to_directory



print("Φόρτωση δεδομένων CIFAR-10...")
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print("Κανονικοποίηση εικόνων...")
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

print("One-Hot Encoding ετικετών...")
num_classes = 10 
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

input_shape = x_train.shape[1:] 

print("\n--- Επιβεβαίωση ---")
print(f"Διαστάσεις x_train: {x_train.shape}")
print(f"Διαστάσεις y_train: {y_train.shape}") 
print(f"Διαστάσεις Εικόνας Εισόδου (Input Shape): {input_shape}")
print(f"Αριθμός Κλάσεων: {num_classes}")

MODEL_DIR = 'models'
MODEL_FILENAME = 'mlp_cifar10_trained_model.keras'
full_path = os.path.join(MODEL_DIR, MODEL_FILENAME)

EPOCHS = 10
BATCH_SIZE = 32

# Έλεγχος για αποθηκευμένο μοντέλο
if os.path.exists(full_path):
    print(f"\nΒρέθηκε αποθηκευμένο μοντέλο. Φόρτωση από: {full_path}...")
    model = tf.keras.models.load_model(full_path) 
    training_time = 0.0 
else:
    model = Sequential([
            Flatten(input_shape=input_shape),
            Dense(128, activation='relu'), 
            Dense(64, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])

    print("\n\nCompile του Μοντέλου...")
    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    model.summary()

    print(f"\nΕκπαίδευση μοντέλου για {EPOCHS} εποχές...")
    start_time = time.time()

    history = model.fit(x_train, y_train, 
                        epochs=EPOCHS, 
                        batch_size=BATCH_SIZE, 
                        validation_data=(x_test, y_test)) 

    end_time = time.time()
    training_time = end_time - start_time   

    print(f"\nΟλοκληρώθηκε η εκπαίδευση.")
    print(f"Χρόνος Εκπαίδευσης: {training_time:.2f} δευτερόλεπτα")

    # Αποθήκευση του εκπαιδευμένου μοντέλου
    save_model_to_directory(model, MODEL_FILENAME)


print("\n--- Τελική Αξιολόγηση ---")
loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Τελικό Ποσοστό Επιτυχίας (Testing Set): {test_accuracy * 100:.2f}%")

_, train_accuracy = model.evaluate(x_train, y_train, verbose=0) 
print(f"Τελικό Ποσοστό Επιτυχίας (Training Set): {train_accuracy * 100:.2f}%")

print(f"\nΣυνολικός Χρόνος Εκπαίδευσης: {training_time:.2f} δευτερόλεπτα") 