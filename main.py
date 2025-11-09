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


print("Φόρτωση δεδομένων CIFAR-10...")
(x_train_original, y_train_original), (x_test_original, y_test_original) = cifar10.load_data()

print("Κανονικοποίηση εικόνων...")
x_train = x_train_original.astype('float32') / 255.0
x_test = x_test_original.astype('float32') / 255.0

print("One-Hot Encoding ετικετών...")
num_classes = 10 
y_train = tf.keras.utils.to_categorical(y_train_original, num_classes)
y_test = tf.keras.utils.to_categorical(y_test_original, num_classes)

input_shape = x_train.shape[1:] 

X_train_flat = x_train.reshape(x_train.shape[0], -1) 
X_test_flat = x_test.reshape(x_test.shape[0], -1)   

N_COMPONENTS = 100 
print(f"Εφαρμογή PCA: Μείωση διάστασης από 3072 σε {N_COMPONENTS} συνιστώσες...")

pca = PCA(n_components=N_COMPONENTS)
pca.fit(X_train_flat) 

x_train_pca = pca.transform(X_train_flat)
x_test_pca = pca.transform(X_test_flat)

new_input_shape = (N_COMPONENTS,) 

print(f"Νέα διάσταση εισόδου για το MLP (μετά PCA): {new_input_shape}")

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
    print(f"\nΒρέθηκε αποθηκευμένο μοντέλο. Φόρτωση από: {full_path}...")
    model = tf.keras.models.load_model(full_path) 
    training_time = 0.0 
else:
    model = Sequential([
            Dense(128, activation='relu', input_shape=new_input_shape), 
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

    # SAVING TRAINED MODEL
    save_model_to_directory(model, MODEL_FILENAME)


print("\n--- Τελική Αξιολόγηση ---")


# EXAMPLE WITH PICTURES
if training_time > 0:
    print(f"\nΣυνολικός Χρόνος Εκπαίδευσης: {training_time:.2f} δευτερόλεπτα")
else:
    print("\nΤο μοντέλο φορτώθηκε από αποθηκευμένο αρχείο, χωρίς εκπαίδευση.")

loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Τελικό Ποσοστό Επιτυχίας (Testing Set): {test_accuracy * 100:.2f}%")

_, train_accuracy = model.evaluate(x_train, y_train, verbose=0) 
print(f"Τελικό Ποσοστό Επιτυχίας (Training Set): {train_accuracy * 100:.2f}%")

predictions = model.predict(x_test)

predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_test, axis=1)

correct_indices = np.where(predicted_classes == true_classes)[0]
incorrect_indices = np.where(predicted_classes != true_classes)[0]

cifar10_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

fig, axes = plt.subplots(1, 2, figsize=(8, 4)) 
fig.suptitle('Παραδείγματα Κατηγοριοποίησης MLP (CIFAR-10)')
plt.subplots_adjust(wspace=0.3) 

# CORRECT EXAMPLE
if len(correct_indices) > 0:
    first_correct_idx = correct_indices[0]
    axes[0].imshow(x_test_original[first_correct_idx])
    
    true_label = cifar10_labels[true_classes[first_correct_idx]]
    pred_label = cifar10_labels[predicted_classes[first_correct_idx]]

    axes[0].set_title(f"Correct\nReal: {true_label}\nPredicted: {pred_label}", color='green')
    axes[0].axis('off') 

# WRONG EXAMPLE
if len(incorrect_indices) > 0:
    first_incorrect_idx = incorrect_indices[0]
    axes[1].imshow(x_test_original[first_incorrect_idx])

    true_label = cifar10_labels[true_classes[first_incorrect_idx]]
    pred_label = cifar10_labels[predicted_classes[first_incorrect_idx]]
    
    axes[1].set_title(f"Wrong\nReal: {true_label}\nPredicted: {pred_label}", color='red')
    axes[1].axis('off')
plt.show()