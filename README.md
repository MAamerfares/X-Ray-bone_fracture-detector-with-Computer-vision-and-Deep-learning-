# Bone Fracture Detection using Deep Learning

This project utilizes a deep learning model to detect and classify bone fractures from X-ray images. The model is built with TensorFlow and Keras and employs a modern object detection architecture to identify the location and type of fracture.

## Table of Contents

-   [Project Overview](#project-overview)
-   [Dataset](#dataset)
-   [Model Architecture](#model-architecture)
-   [Getting Started](#getting-started)
    -   [Prerequisites](#prerequisites)
    -   [Installation](#installation)
-   [Usage](#usage)
    -   [Data Preparation](#data-preparation)
    -   [Training](#training)
    -   [Evaluation](#evaluation)
-   [Performance](#performance)
-   [Contributing](#contributing)

## Project Overview

The goal of this project is to provide an automated system for detecting bone fractures in X-ray images. The model is trained to identify different types of fractures across various parts of the body, including the elbow, finger, forearm, hand, and shoulder. For each of these categories, the model also classifies the image as either positive (containing a fracture) or negative.

## Dataset

The model was trained on the "XR Bones Dataset for Bone Fracture Detection" available on Kaggle. The dataset is organized in the YOLO format.

-   **Training Set:**
    -   23,700 images
    -   21,000 corresponding label files
    -   *Note: The ~2,700 images without labels serve as negative examples to help the model learn to avoid false positives.*
-   **Validation Set:**
    -   1,000 images
    -   1,000 corresponding label files

The dataset is divided into 10 classes:
*   `XR_ELBOW_positive`
*   `XR_FINGER_positive`
*   `XR_FOREARM_positive`
*   `XR_HAND_positive`
*   `XR_SHOULDER_positive`
*   `XR_ELBOW_negative`
*   `XR_FINGER_negative`
*   `XR_FOREARM_negative`
*   `XR_HAND_negative`
*   `XR_SHOULDER_negative`

A comprehensive Exploratory Data Analysis (EDA) was performed to understand the class distribution and bounding box characteristics, which informed the use of class weights during training to handle class imbalance.

## Model Architecture

The model is a custom-built object detector inspired by state-of-the-art architectures.

1.  **Backbone:** A custom backbone network with residual connections is used for feature extraction. Residual connections help in training deeper networks and improve feature learning.
2.  **Feature Pyramid Network (FPN):** An FPN is used to create a rich, multi-scale feature pyramid. This allows the model to detect fractures of various sizes effectively.
3.  **Detection Head:** The model has three detection heads for:
    *   **Classification:** Predicting the class of the detected object (e.g., `XR_ELBOW_positive`).
    *   **Bounding Box Regression:** Predicting the precise coordinates of the fracture.
    *   **Objectness Score:** Determining if an object is present in a given region.

The model is compiled with the Adam optimizer and uses binary cross-entropy for the classification and objectness losses, and mean squared error for the bounding box regression loss.

## Getting Started

### Prerequisites

-   Python 3.8+
-   TensorFlow 2.x
-   CUDA and cuDNN (for GPU support)
-   The required Python packages can be installed via pip:

```bash
pip install tensorflow numpy pandas matplotlib seaborn opencv-python scikit-learn tqdm
