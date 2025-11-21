import os

def save_model_to_directory(model, model_name, model_dir='models'):
    MODEL_DIR = 'models'
    MODEL_FILENAME = 'cnn_64_128_cifar10_trained_model.keras'

    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        print(f"\nΔημιουργήθηκε ο φάκελος: {MODEL_DIR}/")

    full_path = os.path.join(MODEL_DIR, MODEL_FILENAME)

    print(f"Αποθήκευση μοντέλου στο: {full_path}...")
    model.save(full_path)
    print("Το μοντέλο αποθηκεύτηκε επιτυχώς.")


