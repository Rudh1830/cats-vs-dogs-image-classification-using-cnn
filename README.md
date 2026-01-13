# Cats vs Dogs Image Classification using CNN

This project implements a Convolutional Neural Network (CNN) to classify images of cats and dogs.
The model learns visual features such as edges, textures, and shapes from images and predicts whether a given image contains a cat or a dog.

📌 Project Overview

Image classification is a core computer vision problem.
In this project, a CNN is trained on labeled images of cats and dogs to perform binary classification.

The workflow includes:

Data preprocessing and normalization

CNN model construction

Model training and validation

Performance evaluation on unseen images

🛠️ Technologies Used

Python

TensorFlow / Keras

NumPy

Matplotlib

Google Colab (recommended for training)

📂 Dataset

Cats vs Dogs dataset

Two classes:

Cat

Dog
🧠 Model Architecture

The CNN follows a standard architecture:

Convolutional layers (feature extraction)

ReLU activation

MaxPooling layers (downsampling)

Fully connected (Dense) layers

Sigmoid output layer (binary classification)

The model is designed to balance performance and simplicity, making it suitable for learning and demonstration purposes.

⚙️ Preprocessing Steps

Image resizing to a fixed shape

Pixel value normalization (0–1 range)

Batch loading using data generators

Optional data augmentation (rotation, flipping)

📈 Training

Loss function: Binary Crossentropy

Optimizer: Adam

Evaluation metrics: Accuracy

Trained over multiple epochs with validation monitoring

📊 Results

The trained model is able to distinguish cats and dogs with good accuracy on validation data.

Performance depends on:

Dataset size

Number of epochs

Model depth and regularization

This project focuses on conceptual correctness and implementation clarity, not leaderboard-level optimization.

🚀 How to Run

Clone the repository:

git clone https://github.com/Rudh1830/cats-vs-dogs-image-classification-using-cnn.git


Open the notebook in Google Colab (recommended)

Upload the dataset or mount Google Drive

Run cells sequentially to:

Preprocess data

Train the model

Evaluate predictions

📌 Output

Training & validation accuracy curves

Loss curves

Predicted class labels for test images

📚 Learning Outcomes

Through this project, you will understand:

How CNNs work for image classification

Image preprocessing pipelines

Binary classification using deep learning

Overfitting vs generalization in CNNs

🔮 Future Improvements

Use pretrained models (VGG16, ResNet)

Hyperparameter tuning

Better data augmentation

Deployment using Flask or Streamlit

👤 Author

Rudresh
AI & Data Science Student
