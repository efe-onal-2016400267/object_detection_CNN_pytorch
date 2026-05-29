# Object Detection with Custom YOLO-like CNN and ResNet-18 Fine-Tuning in PyTorch

A lightweight, custom YOLO-style object detection pipeline implemented in PyTorch for detecting cats and dogs. The project covers the entire deep learning workflow: from custom dataset parsing and aspect-ratio-preserving preprocessing, to data augmentation for class balancing, custom loss formulation, model training, evaluation with mAP@50, confusion matrix analysis, and real-time video inference.

This repository features two distinct model architectures: a custom CNN built from scratch and a transfer learning approach using a fine-tuned ResNet-18 backbone.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset & Preprocessing](#dataset--preprocessing)
3. [Model Architectures](#model-architectures)
   - [Custom CNN (My YOLO)](#custom-cnn-my-yolo)
   - [ResNet-18 Backbone & Fine-Tuning](#resnet-18-backbone--fine-tuning)
4. [Custom YOLO Loss Function](#custom-yolo-loss-function)
5. [Data Augmentation & Balancing](#data-augmentation--balancing)
6. [Training & Evaluation](#training--evaluation)
7. [Inference & Video Processing](#inference--video-processing)
8. [Results & Visualization](#results--visualization)
9. [How to Run](#how-to-run)

---

## Project Overview
This repository implements a single-stage object detector inspired by the YOLO (You Only Look Once) architecture. The model divides the input image into a $7 \times 7$ grid. Each grid cell is responsible for predicting:
- **Bounding Box Coordinates ($x, y, w, h$)**: Relative to the grid cell ($x, y$) and the entire image ($w, h$).
- **Objectness Score**: Confidence that an object exists in the cell.
- **Class Probabilities**: Probabilities for the `Cat` and `Dog` classes.

The project is fully self-contained in a Jupyter Notebook (`object_detection_yolo.ipynb`) and is designed to run on both CPU and CUDA-enabled GPUs.

---

## Dataset & Preprocessing
The model is trained on the [Kaggle Dog and Cat Detection Dataset](https://www.kaggle.com/datasets/andrewmvd/dog-and-cat-detection), which contains images and Pascal VOC XML annotations.

### Preprocessing Pipeline:
1. **Aspect-Ratio-Preserving Resize**:
   - The larger dimension of the image is scaled to fit the target input size ($112 \times 112$).
   - The shorter dimension is padded symmetrically with black pixels to maintain the aspect ratio without distortion.
2. **Bounding Box Scaling**:
   - Bounding box coordinates are scaled and shifted to match the resized and padded image.
3. **Data Splitting**:
   - Stratified split ($70\%$ train, $14\%$ validation, $16\%$ test) to ensure class distributions are preserved across splits.
   - Specific multi-object images (e.g., `Cats_Test736.png`) are excluded to simplify the single-object prediction task.

---

## Model Architectures

### Custom CNN (My YOLO)
A custom convolutional neural network designed from scratch:
- **Feature Extractor**: 5 convolutional blocks with `LeakyReLU(0.1)` activations, Batch Normalization, and Max Pooling.
- **Fully Connected Layers**:
  - `Linear(32 * 7 * 7, 512)` -> `LeakyReLU` -> `Linear(512, 343)`
- **Output Reshaping**:
  - Reshaped to `[Batch, 7, 7, 7]` representing a $7 \times 7$ grid with 7 channels:
    - `[0:4]`: Bounding box coordinates ($x, y, w, h$)
    - `[4]`: Objectness score
    - `[5:7]`: Class scores (Cat, Dog)
- **Activations**: Sigmoid is applied to all outputs to constrain predictions between $0$ and $1$.

### ResNet-18 Backbone & Fine-Tuning
To boost performance, the project leverages transfer learning by adapting a pretrained **ResNet-18** model:
- **Backbone**: Pretrained ResNet-18 (excluding the final average pooling and fully connected layers) is used as a powerful feature extractor.
- **Fine-Tuning Strategy**:
  - The backbone parameters are unfrozen (`param.requires_grad = True`) to allow end-to-end fine-tuning on the custom cat/dog dataset.
  - A custom fully connected head maps the $512 \times 7 \times 7$ feature map to the $343$-dimensional output.
- **Activations**: Sigmoid for bounding boxes and objectness, and Softmax for class scores.

---

## Custom YOLO Loss Function
The custom `CompositeLoss` computes a multi-task loss over the $7 \times 7$ grid:
1. **Grid Assignment**: Maps the ground truth bounding box center to a specific grid cell $(i, j)$.
2. **Coordinate Loss**:
   - Mean Squared Error (MSE) on the center offsets ($x, y$) and the square root of dimensions ($\sqrt{w}, \sqrt{h}$) for the responsible grid cell.
   - Scaled by $\lambda_{\text{coord}} = 5.0$.
3. **Objectness Loss**:
   - MSE loss pushing the predicted objectness score of the responsible cell to $1$.
4. **No-Objectness Loss**:
   - MSE loss pushing the predicted objectness score of empty cells to $0$.
   - Scaled by $\lambda_{\text{noobj}} = 0.5$ to handle class imbalance (most grid cells are empty).
5. **Classification Loss**:
   - MSE loss between predicted class probabilities and the one-hot encoded ground truth class for the responsible cell.

---

## Data Augmentation & Balancing
To improve model generalization and address class imbalance:
- **Class Balancing**: Minority class (cats) images are augmented using brightness, contrast, saturation, and hue adjustments to match the majority class (dogs).
- **Sample Size Expansion**: Applies a series of random transformations including Color Jitter, Gaussian Blur, Solarization, Sharpness adjustments, and Autocontrast to expand the training set size.

---

## Training & Evaluation
- **Optimizer**: Adam optimizer with a learning rate of $0.001$.
- **Early Stopping**: Monitored on validation loss with a tolerance of 10 epochs.
- **Metrics**:
  - **mAP@50**: Mean Average Precision at IoU threshold 0.5 using `torchmetrics`.
  - **Classification Metrics**: Accuracy, Precision, Recall, F1-Score, and AUPRC.
  - **Confusion Matrix**: Includes an "Unmatched" category to track false positives and false negatives.

---

## Inference & Video Processing
The project includes a complete pipeline for running inference on video files:
1. Reads video frames sequentially using OpenCV.
2. Preprocesses each frame (resize, pad, normalize).
3. Runs the model to predict bounding boxes, objectness, and class.
4. Filters predictions using an objectness threshold.
5. Maps the predicted coordinates back to the original video dimensions (reversing the padding and scaling).
6. Draws bounding boxes and class labels on the frames and writes them to an output video file.

---

## Results & Visualization
The notebook provides rich visualization tools:
- **Batch Visualization**: Displays input images with ground truth bounding boxes.
- **Prediction Visualization**: Overlays predicted bounding boxes (with confidence scores) and ground truth boxes on test images.
- **Loss Curves**: Plots training and validation curves for total loss, coordinate loss, objectness loss, and classification loss.
- **Misclassification Analysis**: Displays images where the model made incorrect class predictions or failed to localize objects.

---

## How to Run

### Prerequisites
Install the required dependencies:
```bash
pip install torch torchvision torcheval torchmetrics scikit-learn matplotlib seaborn opencv-python
```

### Running the Notebook
1. Open `object_detection_yolo.ipynb` in Jupyter Notebook or Google Colab.
2. Ensure you have your Kaggle API credentials configured if you want to download the dataset automatically, or manually place the dataset in `./cat_dog_dataset`.
3. Run the cells sequentially to:
   - Download and preprocess the dataset.
   - Train the custom CNN or fine-tune the ResNet-18 model.
   - Evaluate performance and plot metrics.
   - Run video inference on your custom videos.
