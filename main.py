import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D 
import numpy as np
import time
import matplotlib.pyplot as plt
from utils import save_model_to_directory


print("LOADING CIFAR-10 DATA...")
(x_train_original, y_train_original), (x_test_original, y_test_original) = cifar10.load_data()

x_train = x_train_original.astype('float32') / 255.0
x_test = x_test_original.astype('float32') / 255.0

print("IMAGE NORMALIZATION...")

print("LABEL ONE-HOT ENCODING...")
num_classes = 10 
y_train = tf.keras.utils.to_categorical(y_train_original, num_classes)
y_test = tf.keras.utils.to_categorical(y_test_original, num_classes)

input_shape = x_train.shape[1:] 

print(f"CNN INPUT SHAPE (32x32x3): {input_shape}")

MODEL_DIR = 'models'
MODEL_FILENAME = 'cnn_32_64_20_EPOCHS_cifar10_trained_model.keras'
full_path = os.path.join(MODEL_DIR, MODEL_FILENAME)

EPOCHS = 20
BATCH_SIZE = 32

# CHECK IF MODEL EXISTS
if os.path.exists(full_path):
    print(f"FOUND MODEL AT: {full_path}...")
    model = tf.keras.models.load_model(full_path) 
    training_time = 0.0 
else:
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape), 
        MaxPooling2D((2, 2)),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        Flatten(), 
        
        Dense(128, activation='relu'), 
        Dense(num_classes, activation='softmax')
    ])

    print("\n\nCOMPILING MODEL...")
    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    model.summary()

    print(f"\nTRAINING MODEL FOR {EPOCHS} EPOCHS...")
    start_time = time.time()

    history = model.fit(x_train, y_train, 
                        epochs=EPOCHS, 
                        batch_size=BATCH_SIZE, 
                        validation_data=(x_test, y_test)) 

    end_time = time.time()
    training_time = end_time - start_time   

    print(f"\nTRAINING COMPLETE.")
    print(f"TRAINING TIME: {training_time:.2f} seconds")

    # SAVING TRAINED MODEL
    save_model_to_directory(model, MODEL_FILENAME)

print("\n--- FINAL EVALUATION ---")

# EXAMPLE WITH PICTURES
if training_time > 0:
    print(f"\nTOTAL TRAINING TIME: {training_time:.2f} seconds")
    # 1. Accuracy Plot
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # 2. Loss Plot
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

else:
    print("\nMODEL LOADED FROM SAVED FILE, NO TRAINING PERFORMED.")

loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"FINAL TEST ACCURACY: {test_accuracy * 100:.2f}%")

_, train_accuracy = model.evaluate(x_train, y_train, verbose=0) 
print(f"FINAL TRAINING ACCURACY: {train_accuracy * 100:.2f}%")

predictions = model.predict(x_test)

predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_test, axis=1)

correct_indices = np.where(predicted_classes == true_classes)[0]
incorrect_indices = np.where(predicted_classes != true_classes)[0]

cifar10_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

fig, axes = plt.subplots(1, 2, figsize=(8, 4)) 
fig.suptitle('MLP CLASSIFICATION EXAMPLES (CIFAR-10)')
plt.subplots_adjust(wspace=0.3) 

# CORRECT EXAMPLE
if len(correct_indices) > 0:
    first_correct_idx = correct_indices[0]
    axes[0].imshow(x_test_original[first_correct_idx])
    
    true_label = cifar10_labels[true_classes[first_correct_idx]]
    pred_label = cifar10_labels[predicted_classes[first_correct_idx]]

    axes[0].set_title(f"CORRECT\nReal: {true_label}\nPredicted: {pred_label}", color='green')
    axes[0].axis('off') 

# WRONG EXAMPLE
if len(incorrect_indices) > 0:
    first_incorrect_idx = incorrect_indices[0]
    axes[1].imshow(x_test_original[first_incorrect_idx])

    true_label = cifar10_labels[true_classes[first_incorrect_idx]]
    pred_label = cifar10_labels[predicted_classes[first_incorrect_idx]]

    axes[1].set_title(f"WRONG\nReal: {true_label}\nPredicted: {pred_label}", color='red')
    axes[1].axis('off')
plt.show()