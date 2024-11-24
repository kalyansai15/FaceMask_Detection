# FaceMask_Detection

![image](https://github.com/user-attachments/assets/c7719c80-2395-4254-b35f-8daf4c675ee6)

This project uses a Deep Neural Network, more specifically a Convolutional Neural Network, to differentiate between images of people with and without masks. The CNN manages to get an accuracy of 98.2% on the training set and 97.3% on the test set. Then the stored weights of this CNN are used to classify as mask or no mask, in real time, using OpenCV. With the webcam capturing the video, the frames are preprocessed and and fed to the model to accomplish this task. The model works efficiently with no apparent lag time between wearing/removing mask and display of prediction.

---

## Features

- **Real-time Detection**: Uses webcam or video input to detect whether individuals are wearing face masks in real-time.
- **High Accuracy**: Powered by deep learning models (CNN) trained on a dataset of masked and unmasked faces.
- **Easy Integration**: The system can be integrated into various applications, such as security systems, health monitoring applications, and public spaces.
- **Visualization**: The system provides visual feedback with bounding boxes around detected faces, indicating whether a mask is being worn.
- **User-Friendly Interface**: Simple setup and usage with minimal configuration required for running the system.

---

## Installation

### Prerequisites

Before running the Face Mask Detection system, ensure you have the following installed:

- **Python** (>= 3.6)
- **TensorFlow** (for model training and prediction)
- **OpenCV** (for image processing and video capture)
- **Keras** (for model handling)
- **Numpy** (for numerical operations)
- **Matplotlib** (for plotting, if required)

You can install the dependencies using `pip`:

```bash
pip install tensorflow opencv-python keras numpy matplotlib
```

---

# Steps to Install
1. **Clone the Repository:**
Clone this repository to your local machine using Git.
```bash
git clone https://github.com/kalyansai15/FaceMask_Detection.git
cd FaceMask_Detection
```
2. **Download the Pre-trained Model:**

The model has been pre-trained on a face mask dataset. You can either train it yourself or download the pre-trained weights from this <a href="https://www.kaggle.com/datasets/andrewmvd/face-mask-detection" dowload>link</a>.

This dataset consists of 4095 images belonging to two classes:

. with_mask: 2165 images<br/>
. without_mask: 1930 images<br/>

3. **Run the Script:**

To start the face mask detection system, simply run the following script:
```bash
python train.py
python test.py
```
This will open a webcam feed and detect face masks in real-time.

---

# Tech Stack
- ** Python: The programming language used for building the application.
- ** TensorFlow and Keras: For training and running the Convolutional Neural Network (CNN) model.
- ** OpenCV: For image and video processing (capturing frames, applying models).
- ** NumPy: For numerical operations and array handling.
- ** Matplotlib: For visualizing results (optional).

# Folder Structure
```plaintext 
FaceMask_Detection/
│
├── dataset/                  # Dataset of images for training (masked vs unmasked)
├── model/                    # Trained model weights and architecture
├── scripts/                  # Python scripts for training and testing
│   ├── detect_mask_video.py  # Script for real-time mask detection
│   ├── train_model.py        # Script for training the face mask detection model
│   └── utils.py              # Utility functions for model training and testing
│
├── README.md                 # This file
└── requirements.txt          # Python dependencies
```

---

# How it Works
1. **Dataset:**
The dataset consists of images of people wearing face masks and not wearing face masks. The images are labeled, which allows the model to learn and differentiate between the two categories during training.

2. **Model:**
A Convolutional Neural Network (CNN) is used to classify images based on the presence of face masks. The model is trained using TensorFlow and Keras.

3. **Real-time Detection:**
Once trained, the model is used to process images or video streams. When a face is detected in the frame, the model classifies it as "Mask" or "No Mask" and draws a bounding box around the face, indicating the mask status.

4. **Result:**
The system continuously shows the real-time video feed with feedback on whether the individual is wearing a mask or not.

--- 

# Contributing
We welcome contributions! If you want to enhance the project or add new features, feel free to fork the repository, make your changes, and submit a pull request. Please follow the project's coding conventions and document your changes.

---




