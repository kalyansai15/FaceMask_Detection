# FaceMask_Detection

This project uses a Deep Neural Network, more specifically a Convolutional Neural Network, to differentiate between images of people with and without masks. The CNN manages to get an accuracy of 98.2% on the training set and 97.3% on the test set. Then the stored weights of this CNN are used to classify as mask or no mask, in real time, using OpenCV. With the webcam capturing the video, the frames are preprocessed and and fed to the model to accomplish this task. The model works efficiently with no apparent lag time between wearing/removing mask and display of prediction.

<b>The model is capable of predicting multiple faces with or without masks at the same time
