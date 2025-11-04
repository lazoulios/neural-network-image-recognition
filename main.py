import tensorflow as tf
from keras.datasets import cifar10
import numpy as np

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
print(f"Διαστάσεις x_train: {x_train.shape}") # Αναμένεται (50000, 32, 32, 3)
print(f"Διαστάσεις y_train: {y_train.shape}") # Αναμένεται (50000, 10)
print(f"Διαστάσεις Εικόνας Εισόδου (Input Shape): {input_shape}")
print(f"Αριθμός Κλάσεων: {num_classes}")