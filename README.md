This project demonstrates a classification task for identifying different bean leaf lesions using Pytorch model and a pre-trained deep learning model, GoogLeNet, fine-tuned on a custom dataset.

## Project Overview

The goal of this project is to classify images of bean leaves into different lesion categories. This is achieved by leveraging transfer learning with a state-of-the-art convolutional neural network.

## Dataset

The dataset used in this project is the "Bean Leaf Lesions Classification" dataset from Kaggle. It contains images of bean leaves categorized by different lesion types.

## Dependencies

The following Python libraries are required to run this notebook. You can install them using the `requirements.txt` file generated previously.

- `opendatasets`
- `torch`
- `torchvision`
- `scikit-learn`
- `matplotlib`
- `Pillow`
- `pandas`
- `numpy`

## Model Architecture

This project utilizes a ppytorch model and a pre-trained GoogLeNet model from `torchvision.models`. The final classification layer of the GoogLeNet model is replaced and fine-tuned for the specific number of classes in the bean leaf lesion dataset.

## Key Steps

The notebook performs the following key steps:

1.  **Dataset Download**: Downloads the "Bean Leaf Lesions Classification" dataset from Kaggle.
2.  **Data Loading and Preprocessing**: Loads the image paths and labels, splits the data into training and testing sets, and applies image transformations (resizing, converting to tensor, normalization).
3.  **Custom Dataset and DataLoader**: Defines a custom `Dataset` class to handle image loading and labeling, and uses `DataLoader` for efficient batch processing.
4.  **Model Initialization**: Loads a pre-trained GoogLeNet model and modifies its final fully connected layer to match the number of output classes in our dataset.
5.  **Training Configuration**: Sets up the loss function (`CrossEntropyLoss`) and optimizer (`Adam`).
6.  **Model Training**: Trains the fine-tuned GoogLeNet model for a specified number of epochs, tracking training loss and accuracy.
7.  **Model Evaluation**: Evaluates the trained model on the test set to report its accuracy.
8.  **Visualization**: Plots the training loss and accuracy over epochs to visualize the model's learning progress.

## How to Run

1.  Ensure you have a Kaggle account and your API credentials are set up for `opendatasets` to download the data.
2.  Install the required dependencies listed in `requirements.txt`.
3.  Execute the cells in the provided Jupyter notebook sequentially.
