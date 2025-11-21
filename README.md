# Neural Networks for CIFAR-10 Classification

The goal of this repository is to implement and compare Feedforward Neural Networks (FNNs) for multi-class image classification on the CIFAR-10 dataset.

The project explores two main architectures: a classical Multi-Layer Perceptron (MLP) with dimensionality reduction via PCA, and a modern Convolutional Neural Network (CNN).

## 📊 Key Results

The CNN architecture proved vastly superior due to its effective feature extraction capabilities:

| Architecture | Pre-processing | Peak Test Accuracy | Comments |
| :--- | :--- | :--- | :--- |
| **CNN (64, 128)** | None (Image Input) | **70.00%** | Optimal performance; learned features are superior. |
| **MLP (64/32)** | PCA (100 Components) | 52.88% | Required PCA to achieve reasonable performance. |

-----

## 🚀 Project Setup and Execution

### 1\. Requirements

The project uses Python 3.x and relies on the following major libraries:

  * `TensorFlow / Keras` (for NN implementation and back-propagation)
  * `Scikit-learn` (for PCA)
  * `Matplotlib` (for plotting training curves and examples)

### 2\. Installation

Clone the repository and set up a virtual environment:

```bash
# Clone the repository
git clone https://github.com/lazoulios/neural-network-image-recognition
cd neural-network-image-recognition

# Create and activate the virtual environment
python -m venv venv
source venv/bin/activate  # Use venv\Scripts\activate.bat on Windows CMD

# Install dependencies from requirements.txt
pip install -r requirements.txt
```

### 3\. Running the Models

The repository contains two main scripts. Both scripts use a **"Load or Fit"** logic: they check if the trained model exists in the `models/` folder; if so, they load it instantly; otherwise, they train it from scratch and save the model.

#### A. Running the MLP (with PCA)

Runs the best-performing MLP model (64/32 neurons) using 100 PCA components.

```bash
python mlp.py
```

#### B. Running the CNN (Convolutional Neural Network)

Runs a high-capacity CNN (Conv: 32$\to$64 filters, Dense: 128 neurons). This script will demonstrate the high accuracy results (e.g., $\sim 69\%-70\%$).

```bash
python cnn.py
```

-----

## 📁 Repository Structure

  * `mlp.py`: Main script for the MLP experiments (includes PCA logic).
  * `cnn.py`: Main script for the CNN experiments (demonstrates high performance).
  * `utils.py`: Contains the helper function `save_model_to_directory`.
  * `models/`: Directory where trained Keras models are saved (ignored by Git).
  * `requirements.txt`: List of all required Python packages.