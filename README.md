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

The notebook uses two callbacks.

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

## Testing Platform

This repository also includes a separate notebook named:

```bash
testing_platform.ipynb
```

The purpose of `testing_platform.ipynb` is to test the trained `.h5` model generated from the EMNIST training notebook.

This testing notebook is useful when users want to:

* Load a trained EMNIST `.h5` model
* Test the model using a sample handwritten digit or letter image
* Test the model using their own custom handwritten image
* Change the model filename without retraining
* Display the predicted character
* Verify whether the saved model works correctly after training

The testing notebook does not train the model again. It only loads an existing `.h5` file and performs prediction.

---

## Sample Test Images

The repository includes a folder named:

```bash
sample_test_images/
```

This folder stores sample handwritten images that can be used for prediction testing in `testing_platform.ipynb`.

Current sample images:

```text
sample_test_images/
├── image_01.png
├── image_04.png
├── image_06.png
├── image_07.png
├── image_08.png
├── image_09.png
└── image_10.png
```

These sample images can be used directly in `testing_platform.ipynb` to check whether the trained `.h5` model can correctly predict handwritten digits and uppercase letters.

To test one of the sample images, update the image path in `testing_platform.ipynb`:

```python
IMAGE_PATH = "sample_test_images/image_01.png"
```

You can change the filename to test another image:

```python
IMAGE_PATH = "sample_test_images/image_04.png"
IMAGE_PATH = "sample_test_images/image_06.png"
IMAGE_PATH = "sample_test_images/image_07.png"
IMAGE_PATH = "sample_test_images/image_08.png"
IMAGE_PATH = "sample_test_images/image_09.png"
IMAGE_PATH = "sample_test_images/image_10.png"
```

If you want to add your own test image, place the image inside:

```bash
sample_test_images/
```

For example:

```text
sample_test_images/my_handwritten_letter.png
```

Then update the image path:

```python
IMAGE_PATH = "sample_test_images/my_handwritten_letter.png"
```

Recommended image format:

```text
PNG or JPG
```

Recommended image style:

* Single handwritten character only
* Centered character
* High contrast
* Minimal background noise
* Preferably black handwriting on a white background

If the image has black handwriting on a white background, use:

```python
invert=True
```

If the image already has bright handwriting on a dark background, use:

```python
invert=False
```

---

## Creating the Sample Image Folder on GitHub

GitHub does not save empty folders. To create the `sample_test_images/` folder directly on GitHub, create a placeholder file inside it.

In GitHub, click **Add file** → **Create new file**, then type this filename:

```text
sample_test_images/.gitkeep
```

Then commit the file.

This will create the folder:

```text
sample_test_images/
└── .gitkeep
```

After that, you can upload sample images into the folder, such as:

```text
sample_test_images/image_01.png
sample_test_images/image_04.png
sample_test_images/image_06.png
sample_test_images/image_07.png
sample_test_images/image_08.png
sample_test_images/image_09.png
sample_test_images/image_10.png
```

---

## Testing Your Own `.h5` Model

To test your own trained model, place your `.h5` file in the same folder as `testing_platform.ipynb`, or place it inside a dedicated model folder such as:

```bash
models/
```

Then update the model path inside the notebook.

Example if the model is in the main repository folder:

```python
MODEL_PATH = "EMNIST_6.35%_Yosef_Budiman.h5"
```

Example if the model is inside the `models/` folder:

```python
MODEL_PATH = "models/EMNIST_6.35%_Yosef_Budiman.h5"
```

For example, if your model file is named:

```bash
my_emnist_model.h5
```

change the code to:

```python
MODEL_PATH = "my_emnist_model.h5"
```

or:

```python
MODEL_PATH = "models/my_emnist_model.h5"
```

The model can then be loaded using:

```python
import tensorflow as tf

model = tf.keras.models.load_model(MODEL_PATH, compile=False)
```

The argument `compile=False` is used because the model is only needed for inference. This also helps avoid loading issues related to optimizer states, custom loss settings, or label smoothing configuration.

---

## Character Labels

The EMNIST model predicts 36 classes.

The class labels are defined as:

```python
CLASS_NAMES = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
```

The index order is:

```text
0  -> 0
1  -> 1
2  -> 2
3  -> 3
4  -> 4
5  -> 5
6  -> 6
7  -> 7
8  -> 8
9  -> 9
10 -> A
11 -> B
12 -> C
13 -> D
14 -> E
15 -> F
16 -> G
17 -> H
18 -> I
19 -> J
20 -> K
21 -> L
22 -> M
23 -> N
24 -> O
25 -> P
26 -> Q
27 -> R
28 -> S
29 -> T
30 -> U
31 -> V
32 -> W
33 -> X
34 -> Y
35 -> Z
```

---

## Image Preprocessing for Testing

The testing image must be processed in the same general format used by the training model.

The model expects:

```text
28 x 28 grayscale image
```

with shape:

```text
(1, 28, 28, 1)
```

and pixel values normalized to:

```text
0.0 – 1.0
```

Example preprocessing code:

```python
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

def preprocess_image(image_path, invert=True):
    """
    Preprocess an external image for EMNIST model prediction.

    Parameters:
    image_path : str
        Path to the image file.
    invert : bool
        Set to True if the input image has dark handwriting on a light background.
        Set to False if the input image already has light handwriting on a dark background.

    Returns:
    image_array : numpy.ndarray
        Preprocessed image with shape (1, 28, 28, 1).
    display_image : PIL.Image
        Processed image for visualization.
    """

    image = Image.open(image_path).convert("L")

    if invert:
        image = ImageOps.invert(image)

    image = image.resize((28, 28))

    image_array = np.array(image).astype("float32") / 255.0
    image_array = np.expand_dims(image_array, axis=-1)
    image_array = np.expand_dims(image_array, axis=0)

    return image_array, image
```

Use `invert=True` when the image has black handwriting on a white background. Use `invert=False` when the image already looks like EMNIST format, meaning bright handwriting on a dark background.

---

## Single Image Prediction Code

After loading the model and defining the preprocessing function, test one image using:

```python
IMAGE_PATH = "sample_test_images/image_01.png"

image_array, display_image = preprocess_image(IMAGE_PATH, invert=True)

prediction = model.predict(image_array)
predicted_index = np.argmax(prediction)
predicted_label = CLASS_NAMES[predicted_index]
confidence = prediction[0][predicted_index] * 100

print(f"Predicted class: {predicted_label}")
print(f"Confidence: {confidence:.2f}%")

plt.imshow(display_image, cmap="gray")
plt.title(f"Prediction: {predicted_label} ({confidence:.2f}%)")
plt.axis("off")
plt.show()
```

To test another sample image, only change this line:

```python
IMAGE_PATH = "sample_test_images/image_04.png"
```

---

## Top-5 Prediction Code

To inspect the model's most likely predictions, use:

```python
top_k = 5
top_indices = np.argsort(prediction[0])[-top_k:][::-1]

print("Top predictions:")
for rank, index in enumerate(top_indices, start=1):
    label = CLASS_NAMES[index]
    score = prediction[0][index] * 100
    print(f"{rank}. {label}: {score:.2f}%")
```

This is useful for checking visually similar characters such as:

```text
0 and O
1 and I
2 and Z
5 and S
6 and G
U and V
```

---

## Recommended `testing_platform.ipynb` Workflow

Run the testing notebook cells in this order:

```text
1. Import libraries
2. Define the class labels
3. Set the model filename
4. Load the `.h5` model
5. Define the image preprocessing function
6. Set the sample image path
7. Run prediction
8. Display the predicted label and confidence
9. Display top-5 predictions
```

Default test setup:

```python
MODEL_PATH = "EMNIST_6.35%_Yosef_Budiman.h5"
IMAGE_PATH = "sample_test_images/image_01.png"
```

Other available sample images:

```python
IMAGE_PATH = "sample_test_images/image_04.png"
IMAGE_PATH = "sample_test_images/image_06.png"
IMAGE_PATH = "sample_test_images/image_07.png"
IMAGE_PATH = "sample_test_images/image_08.png"
IMAGE_PATH = "sample_test_images/image_09.png"
IMAGE_PATH = "sample_test_images/image_10.png"
```

---

## Google Colab Testing

If you are using Google Colab, upload the `.h5` model and test image manually:

```python
from google.colab import files

uploaded = files.upload()
```

After uploading, update the filenames:

```python
MODEL_PATH = "your_model_file.h5"
IMAGE_PATH = "your_test_image.png"
```

If your test image is stored inside the sample image folder, use:

```python
IMAGE_PATH = "sample_test_images/image_01.png"
```

Then run the prediction cells.

---

## Usage

1. Clone the repository:

```bash
git clone https://github.com/yoshimabudiman-spec/mnist_cnn_model.git
cd mnist_cnn_model
```

2. Install dependencies:

```bash
pip install numpy tensorflow tensorflow-datasets matplotlib seaborn scikit-learn pillow gdown
```

3. Open the training notebook:

```bash
jupyter notebook emnist_cnn_microresnet.ipynb
```

Or run it in Google Colab.

4. Run the training notebook cells in order:

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

5. Open the testing notebook:

```bash
jupyter notebook testing_platform.ipynb
```

6. Update the model path and sample image path:

```python
MODEL_PATH = "EMNIST_6.35%_Yosef_Budiman.h5"
IMAGE_PATH = "sample_test_images/image_01.png"
```

7. Run the testing cells to predict the selected handwritten digit or letter image.

8. To test another image, change only the image path:

```python
IMAGE_PATH = "sample_test_images/image_10.png"
```

9. To test your own image, place it inside:

```bash
sample_test_images/
```

Then update:

```python
IMAGE_PATH = "sample_test_images/your_image_name.png"
```

---

## Repository Structure

```text
.
├── emnist_cnn_microresnet.ipynb
├── testing_platform.ipynb
├── README.md
├── EMNIST_FULL_36.npz
├── Best_EMNIST_Model.h5
├── EMNIST_6.35%_Yosef_Budiman.h5
├── models/
│   └── .gitkeep
└── sample_test_images/
    ├── .gitkeep
    ├── image_01.png
    ├── image_04.png
    ├── image_06.png
    ├── image_07.png
    ├── image_08.png
    ├── image_09.png
    └── image_10.png
```

Generated files such as `.npz` datasets and `.h5` models may not appear until the notebook has been executed.

The `sample_test_images/` folder is used to store sample images for prediction testing. The `.gitkeep` file is only used to make sure GitHub keeps the folder even if no images have been uploaded yet.

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
* Separate testing notebook for `.h5` model inference
* Adjustable model filename for testing custom trained models
* Sample image folder for prediction testing
* Single-image prediction support
* Top-5 prediction confidence output

---

## Notes

This notebook does not use a simple sequential CNN architecture. It uses a custom Micro-ResNet architecture with residual connections.

The model does not explicitly use L2 regularization in the current implementation. Most convolutional layers use `use_bias=False` because they are followed by batch normalization.

The model is saved in `.h5` format. TensorFlow may show a warning that HDF5 is a legacy format. For future development, the model can also be saved using the newer Keras format:

```python
emnist_model.save("EMNIST_Model.keras")
```

When testing external handwritten images, prediction quality depends heavily on preprocessing. For best results, use clear centered characters, high contrast, and minimal background noise.

If the model predicts visually similar characters incorrectly, inspect the top-5 prediction output. EMNIST contains several naturally confusing pairs, especially `I/1`, `O/0`, `Z/2`, `S/5`, and `G/6`.

---

## Authors

Created by:

* Yosef Budiman
* Jack Zheng

Department of Mechanical Engineering  
National Cheng Kung University
