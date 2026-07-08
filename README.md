# EMNIST Micro-ResNet Classification

This project implements a lightweight **Micro-Residual Convolutional Neural Network (Micro-ResNet)** to classify handwritten **digits and uppercase letters** from the EMNIST dataset.

The model classifies 36 handwritten character classes:

* Digits: `0–9`
* Uppercase letters: `A–Z`

The architecture uses residual skip connections, batch normalization, Swish activations, data augmentation, dropout, cosine learning-rate decay, and label smoothing to improve generalization and reduce overfitting.

---

## Dataset

This project uses the **EMNIST ByClass** dataset from TensorFlow Datasets.

The original EMNIST ByClass dataset contains 62 classes. In this project, the dataset is filtered to keep only labels `< 36`, which represent:

```text
0–9 and A–Z
```

The dataset is processed as follows:

* Loads EMNIST using `tensorflow_datasets`
* Filters the dataset to 36 classes
* Normalizes image pixels to the range `[0, 1]`
* Corrects EMNIST image orientation using transpose
* Converts labels to one-hot encoding
* Shuffles the training data using seed `42`
* Uses 10% of the training data as validation data

The processed dataset is cached as:

```bash
EMNIST_FULL_36.npz
```

---

## Model Architecture

The model is defined in:

```python
build_perfect_hybrid_cnn()
```

### Input

```text
28 x 28 x 1 grayscale image
```

### Data Augmentation

The model applies augmentation inside the network:

* Random rotation: `0.05`
* Random zoom: `0.05`
* Random translation: `0.05`

### Initial Convolution

* Conv2D: 32 filters, 3x3 kernel, same padding
* Batch Normalization
* Swish activation

### Residual Block 1

* Conv2D: 32 filters, 3x3 kernel
* Batch Normalization
* Swish activation
* Conv2D: 32 filters, 3x3 kernel
* Batch Normalization
* Residual skip connection
* Swish activation
* MaxPooling2D
* Dropout: `0.10`

### Residual Block 2

* Projection shortcut with 1x1 convolution
* Conv2D: 64 filters, 3x3 kernel
* Batch Normalization
* Swish activation
* Conv2D: 64 filters, 3x3 kernel
* Batch Normalization
* Residual skip connection
* Swish activation
* MaxPooling2D
* Dropout: `0.15`

### Residual Block 3

* Projection shortcut with 1x1 convolution
* Conv2D: 128 filters, 3x3 kernel
* Batch Normalization
* Swish activation
* Conv2D: 128 filters, 3x3 kernel
* Batch Normalization
* Residual skip connection
* Swish activation
* MaxPooling2D

### Classification Head

* GlobalAveragePooling2D
* Dropout: `0.20`
* Dense output layer: 36 units
* Softmax activation

---

## Training Configuration

* Batch size: `64`
* Maximum epochs: `100`
* Optimizer: Adam
* Initial learning rate: `0.001`
* Learning-rate schedule: CosineDecay
* Loss function: Categorical Crossentropy
* Label smoothing: `0.1`
* Metric: Accuracy
* Validation split: `10%`

The notebook uses two callbacks:

### Early Stopping

```python
EarlyStopping(
    monitor="val_accuracy",
    patience=8,
    restore_best_weights=True
)
```

### Model Checkpoint

```python
ModelCheckpoint(
    "Best_EMNIST_Model.h5",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max"
)
```

---

## Results

The EMNIST model achieved:

```text
Best validation accuracy: 93.70%
Best validation epoch: Epoch 13
Final test loss: 0.8105
Final test accuracy: 93.65%
Final test error: 6.35%
```

---

## Class-Level Performance

The notebook evaluates per-class accuracy for all 36 classes.

Some of the strongest classes include:

```text
3: 99.65%
M: 99.66%
7: 99.61%
9: 99.60%
6: 99.51%
```

Some of the most difficult classes include:

```text
I: 33.69%
O: 53.51%
Z: 82.97%
0: 84.18%
L: 89.14%
```

The most confused character pairs are:

```text
I → 1: 65.5%
O → 0: 46.0%
Z → 2: 15.5%
0 → O: 14.8%
D → 0: 7.2%
V → U: 6.5%
G → 6: 6.5%
5 → S: 6.0%
L → 6: 5.6%
Y → 4: 5.4%
```

These errors are reasonable because several EMNIST characters are visually similar, especially `I/1`, `O/0`, and `Z/2`.

---

## Visualizations

The notebook includes:

* Training and validation loss curves
* Training and validation accuracy curves
* Confusion matrix for all 36 classes
* Per-class accuracy report
* Most confused character pairs
* Correct prediction samples with confidence scores
* Incorrect prediction samples with confidence scores

---

## Model Saving

The best model checkpoint is saved as:

```bash
Best_EMNIST_Model.h5
```

The final model is saved using the test error rate in the filename:

```bash
EMNIST_6.35%_Yosef_Budiman.h5
```

---

## Usage

1. Clone the repository:

```bash
git clone https://github.com/yoshimabudiman-spec/mnist_cnn_model.git
cd mnist_cnn_model
```

2. Install dependencies:

```bash
pip install numpy tensorflow tensorflow-datasets matplotlib seaborn scikit-learn gdown
```

3. Open the notebook:

```bash
jupyter notebook emnist_cnn_microresnet.ipynb
```

Or run it in Google Colab.

4. Run the notebook cells in order:

```text
1. Install and import dependencies
2. Set random seeds
3. Define the Micro-ResNet model
4. Load and preprocess the EMNIST dataset
5. Train the EMNIST model
6. Plot training curves
7. Generate the confusion matrix
8. Analyze per-class accuracy
9. Display prediction samples
10. Save the final model
```

---

## Repository Structure

```text
.
├── emnist_cnn_microresnet.ipynb
├── README.md
├── EMNIST_FULL_36.npz
├── Best_EMNIST_Model.h5
└── EMNIST_6.35%_Yosef_Budiman.h5
```

Generated files such as `.npz` datasets and `.h5` models may not appear until the notebook has been executed.

---

## Key Features

* EMNIST 36-class digit and uppercase-letter classification
* Lightweight Micro-ResNet architecture
* Residual skip connections
* Batch normalization
* Swish activation
* Built-in data augmentation
* Dropout regularization
* Label smoothing
* Cosine learning-rate decay
* Early stopping
* Best-model checkpointing
* Confusion matrix analysis
* Per-class accuracy report
* Automatic final model naming based on test error rate

---

## Notes

This notebook does not use a simple sequential CNN architecture. It uses a custom Micro-ResNet architecture with residual connections.

The model does not explicitly use L2 regularization in the current implementation. Most convolutional layers use `use_bias=False` because they are followed by batch normalization.

The model is saved in `.h5` format. TensorFlow may show a warning that HDF5 is a legacy format. For future development, the model can also be saved using the newer Keras format:

```python
emnist_model.save("EMNIST_Model.keras")
```

---

## Authors

Created by:

* Yosef Budiman
* Jack Zheng

Department of Mechanical Engineering
National Cheng Kung University
