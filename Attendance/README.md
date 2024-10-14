# Real-time Face Recognition with OpenCV

This project showcases real-time face recognition using OpenCV in Python. It employs computer vision techniques to detect and recognize faces from live video streams.

## Overview

Facial recognition has become a pivotal technology in various domains, including security, access control, and user authentication. This project harnesses the capabilities of OpenCV, a powerful computer vision library, to implement a real-time face recognition system.

## Key Features

- **Face Detection**: The system utilizes Haar cascades for face detection in real-time video feeds.
- **Dataset Creation**: Users can capture images of faces to create a training dataset for the recognition model.
- **Model Training**: The captured face images are used to train a Fisher Face Recognizer model for accurate recognition.
- **Live Recognition**: Once trained, the system can recognize faces in real-time from the webcam feed.

## Usage

1. **Dataset Creation**:
   - Execute `create_dataset.py` to capture face images and store them in the `datasets` directory. These images serve as the training dataset for the recognition model.

2. **Model Training**:
   - Train the face recognition model using the captured images by running `train_model.py`. This process generates a model file `model.yml` in the project directory.

3. **Real-time Recognition**:
   - Launch `face_recognition.py` to initiate real-time face recognition using the webcam. The system will detect and recognize faces from the live video feed.

## Configuration

- **Haar Cascade File**: Ensure the presence of `haar_cascade.xml` in the project directory. This file is essential for accurate face detection.
- **Parameter Adjustment**: Customize face detection and recognition parameters in the provided scripts as per specific requirements, such as scaling factor and minimum neighbors.

## Contributions

Contributions to this project are welcome! Whether it's bug fixes, enhancements, or new features, feel free to open issues or submit pull requests to improve the functionality of this face recognition system.

## License

This project is licensed under the MIT License. Refer to the [LICENSE](LICENSE) file for detailed terms and conditions.
