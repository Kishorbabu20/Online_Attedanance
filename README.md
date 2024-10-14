### **Overview:**

This project is an **Online Face Recognition-based Attendance System** that automates the process of capturing face images, training a face recognition model, and recording attendance based on the recognized faces. The application consists of two main parts: 
1. **Face Capture**: Users can capture and store images of individuals to be used as training data.
2. **Face Recognition & Attendance**: The system recognizes faces in real-time using a webcam and marks attendance in a CSV file, providing a simple and efficient method for attendance tracking.

### **Description:**

The application uses **OpenCV** and **LBPH (Local Binary Patterns Histogram)** face recognition to detect and identify individuals in real-time. When a face is detected by the webcam, it compares the face with the pre-trained dataset and recognizes the person if their face is stored in the dataset. Based on the recognition result, it marks the attendance and stores it in a CSV file with the date and timestamp.

### **Key Features:**

1. **Face Capture & Dataset Generation:**
   - The app allows users to input the name and department of the individual and captures 50 images from the webcam.
   - These images are resized and stored in a folder (`datasets`) named after the individual and their department for later training.

2. **Face Recognition & Attendance Marking:**
   - The system uses a **pre-trained LBPH model** for face recognition. The model is trained on the captured face images stored in the datasets.
   - The application marks attendance in an `attendance.csv` file with the person's name, timestamp, and attendance status (either "Present" for known faces or "Unknown" for unrecognized faces).
   - Attendance is marked once per individual to avoid multiple entries for the same person during the session.

3. **Confidence Threshold for Recognition:**
   - The system uses a minimum confidence threshold (set at 70) to determine whether the recognized person is known or unknown.
   - Lower confidence indicates a closer match, resulting in successful identification.

4. **Automatic Detection Window Closure:**
   - The detection window automatically closes after 2 seconds once a face is detected, optimizing the workflow.

### **Usage:**

1. **Face Capture:**
   - Run the script to open a simple GUI (using Tkinter) that allows you to enter a name and department for a new individual.
   - Start capturing face images by clicking "Start Capture". The system will collect 50 face images and store them in the `datasets` directory, organized by name and department.

2. **Face Recognition & Attendance:**
   - After training the model with the captured face images, start the face recognition script.
   - The webcam will detect faces in real-time, recognize the individuals based on the trained data, and mark attendance in a CSV file.
   - If an unknown face is detected, it will be labeled as "Unknown" and recorded as such.

### **Configuration:**

- **Environment Setup**:
   - Ensure you have the following libraries installed:
     ```bash
     pip install opencv-python opencv-python-headless numpy
     ```
   - OpenCV is used for image capture and processing, while `numpy` handles array operations for the images.

- **Dataset Configuration:**
   - The captured images are stored in the `datasets` directory, with subdirectories for each individual (name_department).
   - Images are stored as `.png` files and are resized to 130x100 pixels.

- **LBPH Face Recognition Model:**
   - The **Local Binary Patterns Histogram (LBPH)** model is used to recognize faces based on the grayscale images captured from the webcam.
   - The model is trained on the images stored in the `datasets` directory.
   - The model uses the pre-trained **Haar Cascade** classifier for face detection before passing the images to the recognizer.

- **Attendance Configuration:**
   - Attendance is marked in a CSV file (`attendance.csv`) with columns for Name, Timestamp, and Status (Present or Unknown).
   - If the CSV file does not exist, the system creates it when the recognition process starts.
   - The recognized person’s name is written along with the current timestamp to record their attendance.

### **Code Explanation:**

#### **1. Face Capture App** (`FaceCaptureApp`):
- **Objective**: To capture face images and store them in a dataset directory.
- **Process**: 
  - The user enters their name and department.
  - The system captures 50 face samples using the webcam, processes them (grayscale conversion and resizing), and saves them in the dataset folder named after the individual.

#### **2. Face Recognition & Attendance Marking**:
- **Training the Model**: The system loads the images from the dataset folder and trains an **LBPH** face recognizer model on those images.
- **Face Detection & Recognition**:
  - The webcam detects faces in real-time using Haar Cascade classifiers.
  - For each detected face, the system compares it with the trained model.
  - Based on the confidence level, the system either recognizes the person or marks them as "Unknown".
  - Attendance is marked in a CSV file with the name (or "Unknown") and the time of detection.

### **Example Workflow**:

1. **Face Capture**:
   - The user inputs their name ("John") and department ("AI").
   - The system captures 50 face images and stores them in `datasets/John_AI`.

2. **Training**:
   - The system reads all datasets, prepares them for training, and trains the LBPH model.

3. **Recognition & Attendance**:
   - The system opens the webcam and starts recognizing faces.
   - If "John" is recognized, the system marks his attendance in `attendance.csv` with the timestamp.
   - If an unknown person is detected, they are labeled as "Unknown Person", and this is also recorded.

---

This project provides a simple yet powerful tool for **automated face recognition and attendance marking**, streamlining the process of managing attendance in various environments such as classrooms or offices.
