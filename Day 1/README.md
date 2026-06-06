# Animal Prediction using Deep Learning (ResNet50)

## Overview

This project uses a pre-trained ResNet50 Convolutional Neural Network (CNN) model from PyTorch to identify and classify animals (and other objects) from images. The model is trained on the ImageNet dataset and predicts the most likely class for a given input image.

## Features

* Image classification using ResNet50.
* Uses transfer learning with a pre-trained ImageNet model.
* Automatic image preprocessing.
* Displays the predicted class label.
* Simple and easy-to-use implementation in Jupyter Notebook.

## Technologies Used

* Python
* PyTorch
* TorchVision
* Pillow (PIL)
* Requests
* Jupyter Notebook

## Project Structure

```
AnimalPrediction.ipynb
test1.jpg
README.md
```

## Installation

Install the required libraries:

```bash
pip install torch torchvision pillow requests
```

## How It Works

1. Load the pre-trained ResNet50 model.
2. Preprocess the input image:

   * Resize to 256 pixels
   * Center crop to 224×224
   * Convert to tensor
   * Normalize using ImageNet statistics
3. Pass the image through the model.
4. Obtain prediction scores.
5. Display the class with the highest probability.

## Usage

Place the image you want to classify in the project folder and update the file name if necessary.

```python
predict_image("test1.jpg")
```

Example Output:

```text
Predicted Animal/Class: tiger
```

## Dataset

The project uses the ImageNet class labels provided by PyTorch:

* 1000 object categories
* Includes various animals, birds, mammals, and everyday objects

## Future Enhancements

* Support multiple image predictions.
* Build a graphical user interface (GUI).
* Add confidence scores.
* Fine-tune the model on a custom animal dataset.
* Deploy as a web application using Flask or Streamlit.

## Author

Developed as a Deep Learning and Computer Vision project using PyTorch and ResNet50.

