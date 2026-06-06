# Image Classification Baseline: Custom CNN vs. Pre-trained Models

This project evaluates and compares the performance of a custom-built Convolutional Neural Network (CNN) against three powerful, standard pre-trained architectures (**MobileNetV2**, **ResNet50**, and **EfficientNetB0**) on an image classification task using TensorFlow. 

While the notebook contains metadata labels referencing *Fashion-MNIST*, the core implementation builds an end-to-end processing pipeline tailored to analyze and train models on the **CIFAR-10** dataset.

---

##  Project Overview

The repository executes a comparative workflow for modern deep learning vision tasks:
*   **Data Preparation:** Loads a subset of the CIFAR-10 dataset, performs feature scaling/normalization, and dynamically resizes image dimensions.
*   **Performance Optimization:** Uses the `tf.data` API pipeline (`shuffle`, `batch`, `map`, and `prefetch` with `AUTOTUNE`) to prevent I/O bottlenecks during GPU runtime.
*   **Custom Modeling:** Designs a sequential 3-layer Convolutional Neural Network architecture from scratch.
*   **Transfer Learning:** Implements ImageNet-pretrained weights across various architectures with frozen base layers acting as fixed feature extractors.
*   **Result Analytics:** Compiles validation accuracy metrics dynamically and visualizes benchmarks with an output bar chart using `matplotlib`.

---

##  Architecture Details

All models map down to a final $10$-unit Dense layer using a `softmax` activation function paired with `sparse_categorical_crossentropy` loss.

### 1. Custom CNN
* **Inputs:** Shape $(96, 96, 3)$
* **Layers:** 3x alternating `Conv2D` (32, 64, 128 filters with ReLU) and `MaxPooling2D` layers.
* **Classifier:** `Flatten` $\rightarrow$ `Dense` (128 units, ReLU) $\rightarrow$ `Dropout` (0.3) $\rightarrow$ Dense Output.

### 2. Transfer Learning Models (Pre-trained)
The pre-trained model instances drop their dense top layer mapping, freeze their internal convolutional layers (`trainable = False`), and pipe features through a universal classifier configuration:
$$\text{Base Model Layer} \longrightarrow \text{GlobalAveragePooling2D} \longrightarrow \text{Dense (128, ReLU)} \longrightarrow \text{Dropout (0.3)} \longrightarrow \text{Output (10, Softmax)}$$

---

##  Evaluation Benchmarks

Based on training benchmarks execution on a split subset ($10,000$ training images, $2,000$ testing images resized to $96 \times 96$), the following validation accuracies were achieved:

| Model Structure | Top Validation Accuracy | Training Settings |
| :--- | :---: | :---: |
| **MobileNetV2** | **79.00%** | 3 Epochs / Pre-trained |
| **Custom CNN** | **57.65%** | 5 Epochs / From Scratch |
| **ResNet50** | **27.45%** | 3 Epochs / Pre-trained |
| **EfficientNetB0** | **9.75%** | 3 Epochs / Pre-trained |

>  **Note:** `ResNet50` and `EfficientNetB0` require alternative learning rate scheduling, unfreezing/fine-tuning phases, or specific standard preprocessing scaling (e.g., input range handling adjustments) to achieve their optimal operational capacity on this downscaled data arrangement.

---

##  Setup & Installation

### Dependencies
Ensure you have Python 3.10+ and a configured GPU framework environment (recommended). Install the core visualization and compute packages directly via `pip`:

```bash
pip install tensorflow matplotlib pandas
