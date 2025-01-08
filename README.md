# Shoppin_AI_ML

This project implements an Image Search System that uses both deep learning-based feature extraction (CNN and Autoencoders) and traditional computer vision techniques (SIFT and ORB) for image retrieval. The system utilizes the CIFAR-10 dataset to train models and evaluate their ability to retrieve similar images from a query.

Key Features:
CNN-based feature extraction using a pre-trained ResNet-18 model.
Autoencoder for unsupervised feature learning.
Traditional feature extraction using SIFT (Scale-Invariant Feature Transform) and ORB (Oriented FAST and Rotated BRIEF).
FAISS for efficient nearest-neighbor search on deep features (CNN and Autoencoder).
KNN (k-Nearest Neighbors) for classifying traditional features (SIFT and ORB).
Evaluation using accuracy, and visualization of query results.

Requirements
The following libraries are required to run the script:

PyTorch for building and training CNN and Autoencoder models.
FAISS for fast nearest-neighbor search.
OpenCV for traditional feature extraction (SIFT and ORB).
Scikit-learn for k-NN classification.
Matplotlib for visualizing results.
TQDM for progress bars during training and evaluation.

pip install torch torchvision faiss-gpu opencv-python scikit-learn matplotlib tqdm
