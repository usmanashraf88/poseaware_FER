# Pose-Aware Facial Emotion Recognition using Hybrid Deep Learning

## 🧠 Description

This project implements a **Pose-Aware Facial Emotion Recognition (FER)** system using a hybrid deep learning approach. The system combines **Spatial Transformer Networks (STN)** for pose normalization, **multiple pre-trained CNN/Transformer models** for robust feature extraction, and an **XGBoost classifier** for final emotion classification.

The project is designed for high-accuracy emotion classification on facial image datasets like **RAF-DB**, combining the power of CNNs (EfficientNet, ResNet), Transformers (ViT), and tree-based ensemble models.

---

## 📁 Dataset Information

We use the **Real-world Affective Face Database (RAF-DB)**:
- Contains ~30,000 facial images labeled with basic emotion categories.
- Images are labeled by annotators with 7 emotion categories: Happy, Sad, Angry, Disgust, Fear, Surprise, Neutral.
- File names follow the format `label_xxxx.jpg`.

📌 **Note**: You need to download the dataset separately and set `root_dir` accordingly in the code.

---

## 🧾 Code Information

The project contains the following key components:

### 1. `RAFDBDataset` (PyTorch Dataset)
Handles loading, labeling, and transforming RAF-DB images.

### 2. `STN` (Spatial Transformer Network)
Used to normalize face poses before feature extraction.

### 3. `FeatureExtractor`
Concatenates features from three pre-trained models:
- `EfficientNet-B0`
- `ResNet50`
- `ViT (Vision Transformer)`

### 4. `PoseAwareFER`
A unified deep learning model that incorporates STN, feature extraction, and a fully connected emotion classifier.

### 5. Training and Evaluation
- Model trained with CrossEntropyLoss and Adam optimizer.
- Accuracy printed per epoch.
- Features are later used to train an `XGBoostClassifier` for enhanced performance.

---

## ▶️ Usage Instructions

### 1. Prepare the Dataset
Download and extract **RAF-DB**, then set the correct path in:

```python
dataset = RAFDBDataset(root_dir="path_to_rafdb", transform=transform)
