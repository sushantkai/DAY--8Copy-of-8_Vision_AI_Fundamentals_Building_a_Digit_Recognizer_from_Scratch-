📦 Fashion-MNIST Classification Project

This project explores how model complexity affects performance in image classification tasks using the Fashion-MNIST dataset. Three deep learning models were built, trained, and compared:

ANN (Artificial Neural Network)

Basic CNN

Deeper CNN

The goal is to understand how convolutional layers improve performance and whether increased depth always leads to better results.

🚀 Project Overview

Fashion-MNIST is a dataset of 70,000 grayscale images (28×28) belonging to 10 different fashion categories.
This project builds multiple neural network architectures and compares them using:

Accuracy

Loss

Training curves

Confusion matrices

Prediction analysis

📚 Agenda & Steps
1. Dataset Setup

Import all necessary libraries.

Load the Fashion-MNIST dataset.

Normalize pixel values to the range 0–1.

Reshape images for CNN input.

One-hot encode labels.

Verify the shapes of processed data.

2. Model Building

Three models with increasing complexity were created:

🔹 1) Basic ANN Model

Flatten layer

Dense(128, ReLU)

Dense(64, ReLU)

Dense(10, Softmax)

Works, but not ideal for image data.

🔹 2) Basic CNN Model

Conv2D → MaxPooling

Conv2D → MaxPooling

Dense layers

Significantly better accuracy than ANN.

🔹 3) Deeper CNN Model

Additional Conv2D layers

Batch Normalization

Dropout

Larger capacity to capture features

Expected to perform best (but results depend on dataset size and regularization).

3. Model Training

Models trained using training + validation split.

EarlyStopping used to avoid overfitting.

ModelCheckpoint used to save best model based on validation loss.

History stored for accuracy/loss visualization.

4. Model Evaluation

After training, the best saved weights for each model were loaded and tested on the Fashion-MNIST test set.

Evaluation included:

Test accuracy & loss

Accuracy and loss curves

Confusion matrix visualizations

Class-wise performance comparison

5. Prediction Analysis

Using the best model (Basic CNN):

Predictions were generated on test images.

Correct and incorrect predictions were visualized.

This helped understand model strengths and weaknesses for each class.

🏆 Results & Conclusion
Key Findings

The Basic CNN model performed the best overall, giving the highest accuracy and lowest loss.

The ANN model was simple and fast but performed worse than both CNN models.

The Deeper CNN model, although more complex, did not outperform the Basic CNN consistently.
Reasons could include:

Dataset size

Overfitting

Architectural choices

Excessive regularization

Final Conclusion

A moderately complex CNN (Basic CNN) is ideal for Fashion-MNIST.
More layers do not always guarantee better performance.

CNNs clearly outperform ANNs for image classification tasks.

📁 Project Structure
├── data_preprocessing.ipynb
├── ann_model.ipynb
├── cnn_basic_model.ipynb
├── cnn_deeper_model.ipynb
├── training_history_plots/
├── confusion_matrices/
├── predictions_visualization/
└── README.md


(Adjust based on your project folder.)

🧰 Technologies Used

Python

TensorFlow / Keras

NumPy, Pandas

Matplotlib, Seaborn

Scikit-learn

📌 Future Improvements

Hyperparameter tuning

Using data augmentation

Trying ResNet / MobileNet architectures

Applying transfer learning

Using mixed-precision training for faster results
