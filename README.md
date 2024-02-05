# Skin Cancer Prediction
This GitHub repository is dedicated to the exploration of feature extraction and classification methodologies employing state-of-the-art deep learning techniques. The project is organized into three distinct scripts, each addressing a specific facet of the classification pipeline.

## Feature Extraction Framework 
The first script illuminates the capabilities of three Convolutional Neural Networks (CNNs): VGG, Xception, and Inception.

## Classifier Cross Validation Suite 
Proceed to the Cross Validation and Prediction script, where five robust classifiers: Linear Discrimination, Perceptron, LinearSVC, KNN, and Logistic Regression, engage in a systematic evaluation for classification excellence.

## CNN Prediction Efficacy Analysis 
Conclude the journey with a CNN prediction using the VGG architecture.

## Project Objective 
Our primary goal is to conduct a thorough comparative analysis between the outcomes of feature extraction coupled with CNN prediction and the stand-alone CNN prediction methodology.

# HAM10000 Dataset

## Overview

The HAM10000 dataset is a collection of dermatoscopic images consisting of 10015 skin lesions. It is widely used for research and development in the field of dermatology and machine learning for skin cancer diagnosis.

## Dataset Contents

- **Images:** The dataset includes high-quality dermoscopic images of pigmented skin lesions.
- **Labels:** Each image is labeled with information about the diagnosis, including various types of skin lesions such as melanoma, nevus, and seborrheic keratosis.

## Usage

Researchers and developers can utilize this dataset for tasks such as:

- Training and evaluating machine learning models for skin cancer classification.
- Studying the characteristics of different skin lesions.
- Advancing research in dermatology and medical imaging.

## Citation

> P. Tschandl, "The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions," Harvard Dataverse, 2018, Volume 4. [Link to the arcticle](https://arxiv.org/abs/1803.10417v3)

## Data Source

The dataset is publicly available and can be accessed from [here](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000?rvi=1).

## Acknowledgments

We would like to express gratitude to the creators of the HAM10000 dataset for their valuable contribution to the field of dermatology and medical imaging.



# Scripts
## Image Feature Extraction Script

This script is designed for extracting features from images using pre-trained convolutional neural network (CNN) models. The extracted features are saved in LibSVM format.

### Dependencies

- PIL: Python Imaging Library for handling images.
- keras: Deep learning library for neural networks.
- numpy: Numerical operations library.
- os: Operating system interface for file and directory operations.

### Usage

1. Set the file paths for data and directories.

    ```python
    drive_path = 'C:\\Users\\jpedr\\OneDrive\\Documentos\\TCC\\Codigos\\CancerDePele\\cancer\\' 
    entrada = drive_path + 'data.txt' 
    dir_dataset = drive_path + 'data\\' 
    dir_destino = drive_path + 'libsvm\\' 
    ```

2. Read the content of 'data.txt' file, which contains image filenames and their corresponding classes.

    ```python
    arq = open(entrada, 'r')
    conteudo_entrada = arq.readlines()
    arq.close() 
    ```

3. Set directories for dataset and destination (LibSVM files).

    ```python
    if not os.path.exists(dir_destino):
        os.makedirs(dir_destino)
    ```

4. Define image dimensions.

    ```python
    img_rows, img_cols = 299, 299 
    ```

5. Extract features using different CNN models (VGG19, Xception, InceptionV3) and save them in LibSVM format.

## Cross Validation And Prediction Script

### Overview

This Python script performs classification on a given dataset using multiple classifiers and evaluates their performance. The classifiers used include Linear Discriminant Analysis (LDA), Perceptron, Linear Support Vector Machine (LinearSVM), K-Nearest Neighbors (KNN), and Logistic Regression.

### Requirements

- Python 3
- scikit-learn
- pylab (matplotlib)

### Usage

1. Clone the repository or download the script.
2. Ensure the required libraries are installed
3. Run the script with the desired dataset. Uncomment the corresponding line in the `__main__` section for the dataset you want to use.

```python
if __name__ == "__main__":
   ##main("cancer/libsvm/data_VGG.txt") ## Using VGG features
   ##main("cancer/libsvm/data_Xception.txt") ## Using Xception features
   main("cancer/libsvm/data_Inception.txt") ## Using Inception features
```

### Dataset

The script loads datasets in the SVMLight format. Ensure your dataset is formatted accordingly.

### Cross-Validation

The script uses Repeated Stratified K-Fold cross-validation with 2 splits and 5 repeats. You can modify these parameters in the `RepeatedStratifiedKFold` instantiation.

### Classifiers

- Linear Discriminant Analysis (LDA)
- Perceptron
- Linear Support Vector Machine (LinearSVM)
- K-Nearest Neighbors (KNN)
- Logistic Regression

### Results

The script prints the accuracy, precision and F1-Score of each classifier for each fold during cross-validation.

### Note

Make sure to adjust the `max_iter` parameter for classifiers like LinearSVM and Logistic Regression based on your dataset size and complexity.

## CNN Prediction Script

This repository contains code for a skin cancer classification model using the VGG16 architecture. The model is implemented using TensorFlow and Keras. The dataset used for training and testing is provided in the 'data' folder.

### Prerequisites
- TensorFlow
- OpenCV
- NumPy
- Scikit-learn
- Matplotlib

Make sure to install the required dependencies before running the code.

### Dataset
The dataset is organized in the 'data' folder. The file 'data.txt' contains the names and labels of the images. The images are preprocessed by resizing them to (224, 224) pixels.

### Model Architecture
The VGG16 model is used as a base model, and the top layers are modified for the specific classification task. The last 4 layers of the VGG16 model are frozen, and additional fully connected layers are added to adapt the model to the skin cancer classification.

### Training
The model is trained for 10 epochs using the Adam optimizer and categorical crossentropy loss. The training and validation accuracy and loss are plotted using Matplotlib for analysis.

### Usage
1. Clone the repository.
2. Ensure all dependencies are installed.
3. Run the code provided.

Note: Update the 'drive_path' variable in the code to the correct path where the dataset is located.

Feel free to customize the code and experiment with different hyperparameters to improve the model's performance.
