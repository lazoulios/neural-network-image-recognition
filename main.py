import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from utils import save_model_to_directory


print("LOADING CIFAR-10 DATA...")
(x_train_original, y_train_original), (x_test_original, y_test_original) = cifar10.load_data()

print("IMAGE NORMALIZATION...")
x_train = x_train_original.astype('float32') / 255.0
x_test = x_test_original.astype('float32') / 255.0

print("LABEL ONE-HOT ENCODING...")
num_classes = 10 
y_train = tf.keras.utils.to_categorical(y_train_original, num_classes)
y_test = tf.keras.utils.to_categorical(y_test_original, num_classes)

input_shape = x_train.shape[1:] 

X_train_flat = x_train.reshape(x_train.shape[0], -1) 
X_test_flat = x_test.reshape(x_test.shape[0], -1)   

N_COMPONENTS = 100 
print(f"APPLYING PCA: DIMENSIONALITY REDUCTION FROM 3072 TO {N_COMPONENTS} COMPONENTS...")

pca = PCA(n_components=N_COMPONENTS)
pca.fit(X_train_flat) 

x_train_pca = pca.transform(X_train_flat)
x_test_pca = pca.transform(X_test_flat)

new_input_shape = (N_COMPONENTS,) 

print(f"NEW INPUT SHAPE FOR MLP (AFTER PCA): {new_input_shape}")

x_train = x_train_pca
x_test = x_test_pca
input_shape = new_input_shape

MODEL_DIR = 'models'
MODEL_FILENAME = 'mlp_cifar10_trained_model.keras'
full_path = os.path.join(MODEL_DIR, MODEL_FILENAME)

EPOCHS = 10
BATCH_SIZE = 32

# CHECK IF MODEL EXISTS
if os.path.exists(full_path):
    print(f"FOUND MODEL AT: {full_path}...")
    model = tf.keras.models.load_model(full_path) 
    training_time = 0.0 
else:
    model = Sequential([
            Dense(128, activation='relu', input_shape=new_input_shape), 
            Dense(64, activation='relu'),
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