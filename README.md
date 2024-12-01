# s6erav3

## Overview

This repository contains the implementation of the `s6erav3` model, specifically designed to classify handwritten digits from the MNIST dataset. The model is built using PyTorch and follows a convolutional neural network (CNN) architecture with several convolutional, pooling, and fully connected layers to achieve high accuracy on the MNIST dataset.

### Model Architecture

The `MNISTModel` consists of the following layers:

1. **Convolutional Layers**: The model uses multiple convolutional layers to extract features from the input images. Each convolutional layer is followed by ReLU activation, batch normalization, and dropout to improve generalization.
   - **conv1**: Conv2d layer with 8 filters, kernel size 3x3, and padding of 1.
   - **conv2**: Conv2d layer with 12 filters, kernel size 3x3, and padding of 1.
   - **conv3**: Conv2d layer with 12 filters, kernel size 3x3.
   - **conv4**: Conv2d layer with 16 filters, kernel size 3x3.
   - **conv5**: Conv2d layer with 20 filters, kernel size 3x3, and padding of 1.

2. **Transition Layers**: The model includes transition layers to reduce the spatial dimensions of the feature maps.
   - **transition1**: MaxPooling layer with a 2x2 kernel, followed by a Conv2d layer with 8 filters and kernel size 1x1.
   - **transition2**: MaxPooling layer with a 2x2 kernel.

3. **Global Average Pooling (GAP)**: A global average pooling layer is used to reduce the feature map size to 1x1, which helps in reducing overfitting and the number of parameters.
   - **gap**: AvgPool2d layer with a kernel size of 3x3.

4. **Fully Connected Layer**: The model ends with a fully connected layer that outputs 10 classes, representing the digits from 0 to 9.
   - **fc**: Linear layer with 20 input features and 10 output features.

5. **Activation Function**: The final layer uses `log_softmax` for classification.

### Forward Pass
- The model processes input images of size 28x28 through a series of convolutional and pooling layers, followed by a global average pooling and a fully connected layer.
- The output is a log-softmax distribution over the 10 classes.


## Project Structure

```
s6erav3/
│
├── models/
│   ├── __init__.py
│   ├── model.py              # Model definition with CNN, BatchNorm, and Dropout
│
├── training/
│   ├── __init__.py
│   ├── train.py              # Training script
│
├── data/                     # MNIST data (automatically downloaded)
│
├── tests/
│   ├── __init__.py
│   ├── test_model.py         # Test script for model verification
│
├── .github/
│   └── workflows/
│       └── test_model.yml    # GitHub Actions workflow for automated testing
│
├── requirements.txt          # Required Python packages
├── README.md                 # Project documentation
└── main.py                   # Entry point for training the model
```

## Features
- **Model Size**: The model is designed to have about 6.9k parameters.
- **Batch Normalization**: Added after each convolutional layer to improve training stability.
- **Dropout**: Added before the fully connected layer to reduce overfitting.
- **Training & Validation**: The script trains the model and evaluates its performance on the validation set, printing both accuracies.

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/piygr/s6erav3.git
   cd mnist_pytorch
   ```

2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Running the Training Script

To train the model and evaluate it, run:
```sh
python main.py
```
This script will:
- Train the model for one epoch.
- Print the model size, training accuracy, and validation accuracy.
- Save the trained model to `mnist_model.pth`.

### Training Logs
Below are the training logs that demonstrate the model's progress over multiple epochs:

```
Model Size: 0.03 MB
Total Parameters: 6926
Epoch [1/20], Training Accuracy: 92.26%
Epoch [1/20], Validation Accuracy: 98.07%
Epoch [2/20], Training Accuracy: 97.11%
Epoch [2/20], Validation Accuracy: 98.75%
Epoch [3/20], Training Accuracy: 97.53%
Epoch [3/20], Validation Accuracy: 98.73%
Epoch [4/20], Training Accuracy: 97.83%
Epoch [4/20], Validation Accuracy: 98.80%
Epoch [5/20], Training Accuracy: 97.97%
Epoch [5/20], Validation Accuracy: 99.10%
Epoch [6/20], Training Accuracy: 98.23%
Epoch [6/20], Validation Accuracy: 98.84%
Epoch [7/20], Training Accuracy: 98.25%
Epoch [7/20], Validation Accuracy: 99.05%
Epoch [8/20], Training Accuracy: 98.56%
Epoch [8/20], Validation Accuracy: 99.31%
Epoch [9/20], Training Accuracy: 98.65%
Epoch [9/20], Validation Accuracy: 99.33%
Epoch [10/20], Training Accuracy: 98.71%
Epoch [10/20], Validation Accuracy: 99.42%
Epoch [11/20], Training Accuracy: 98.71%
Epoch [11/20], Validation Accuracy: 99.33%
Epoch [12/20], Training Accuracy: 98.72%
Epoch [12/20], Validation Accuracy: 99.39%
Epoch [13/20], Training Accuracy: 98.80%
Epoch [13/20], Validation Accuracy: 99.37%
Epoch [14/20], Training Accuracy: 98.77%
Epoch [14/20], Validation Accuracy: 99.37%
Epoch [15/20], Training Accuracy: 98.77%
Epoch [15/20], Validation Accuracy: 99.39%
Epoch [16/20], Training Accuracy: 98.80%
Epoch [16/20], Validation Accuracy: 99.36%
Epoch [17/20], Training Accuracy: 98.75%
Epoch [17/20], Validation Accuracy: 99.36%
Epoch [18/20], Training Accuracy: 98.75%
Epoch [18/20], Validation Accuracy: 99.36%
Epoch [19/20], Training Accuracy: 98.88%
Epoch [19/20], Validation Accuracy: 99.36%
Epoch [20/20], Training Accuracy: 98.74%
Epoch [20/20], Validation Accuracy: 99.37%
Model training successful: 99.42% accuracy in 20 epochs
```

## Running Tests

To verify the model has fewer than 25,000 parameters, run the test script:
```sh
python tests/test_model.py
```

Or you can run all tests using `pytest`:
```sh
pytest tests/
```

## GitHub Actions Workflow

The repository includes a GitHub Actions workflow for continuous integration. The workflow tests:
- The model has fewer than 25,000 parameters.
- The training process runs without errors.

The workflow can be found at `.github/workflows/test_model.yml`.

## Requirements
- Python 3.8+
- PyTorch 2.0.0
- torchvision 0.15.0
- numpy

Install all dependencies via `pip`:
```sh
pip install -r requirements.txt
```

## License

This project is licensed under the MIT License.