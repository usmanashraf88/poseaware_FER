# poseaware_FER
this project leverages the deep feature representations from pre-trained CNN models including EfficientNet, Vision Transformers, and ResNet for more discriminative and robust feature extraction. 
This project implements a Pose-Invariant Facial Expression Recognition (FER) system using deep feature extraction, hybrid ensemble learning, and attention-based pose normalization. The key innovations include:0

Deep Feature Extraction: Using pre-trained EfficientNet, ResNet, and Vision Transformer (ViT) to obtain robust facial features.

Pose Normalization: Employing a Spatial Transformer Network (STN) for face alignment and self-attention mechanisms to handle pose variations.

Hybrid Ensemble Learning: Refining predictions with a stacked neural network and an XGBoost classifier for better classification accuracy.

Adaptive Weighting for Decision Fusion: Unlike traditional fixed probability distributions, the proposed model dynamically adjusts feature weights.
