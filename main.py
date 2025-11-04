import tensorflow as tf
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Flatten, Dense
import numpy as np
import time

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

EPOCHS = 10
BATCH_SIZE = 32

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