import numpy as np
import cv2
import os
import csv
from datetime import datetime
import time

haar_file = 'haarcascade_frontalface_default.xml'  # Using the standard Haar cascade
datasets = 'datasets'
attendance_file = 'attendance.csv'

print('Training...')

(images, labels, names, id) = ([], [], {}, 0)

# Prepare the dataset
for (subdirs, dirs, files) in os.walk(datasets):
    for subdir in dirs:
        names[id] = subdir
        sub_path = os.path.join(datasets, subdir)

        for filename in os.listdir(sub_path):
            path = os.path.join(sub_path, filename)
            label = id
            images.append(cv2.imread(path, 0))
            labels.append(int(label))

        id += 1

(width, height) = (130, 100)
(images, labels) = [np.array(lis) for lis in [images, labels]]

# Train the model using LBPH
model = cv2.face.LBPHFaceRecognizer_create()
model.train(images, labels)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + haar_file)
webcam = cv2.VideoCapture(0)

recognized = set()  # Track recognized faces to avoid multiple attendance markings

# Function to mark attendance
def mark_attendance(name):
    with open(attendance_file, 'a', newline='') as file:
        writer = csv.writer(file)
        now = datetime.now()
        time_string = now.strftime('%Y-%m-%d %H:%M:%S')
        writer.writerow([name, time_string, 'Present' if name != 'Unknown Person' else 'Unknown'])
    print(f"Attendance marked for {name} at {time_string}")

# Check if the attendance file exists, create if not
if not os.path.isfile(attendance_file):
    with open(attendance_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Name', 'Timestamp', 'Status'])

# Set a minimum confidence threshold for recognizing known individuals
MIN_CONFIDENCE = 70  # Adjusted threshold for better recognition

start_time = None  # Track when a face is detected
detected = False  # Flag to indicate face detection

while True:
    ret, im = webcam.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:  # Check if faces are detected
        detected = True
        if start_time is None:
            start_time = time.time()  # Initialize start time when first face is detected

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face_resize = cv2.resize(face, (width, height))
        face_resize = cv2.equalizeHist(face_resize)  # Equalize the histogram for better contrast

        # Predict the identity of the face
        prediction = model.predict(face_resize)
        confidence = prediction[1]

        # Check the confidence level
        if confidence < MIN_CONFIDENCE:  # If confidence is below the threshold, treat as known person
            name = names[prediction[0]]
            cv2.rectangle(im, (x, y), (x+w, y+h), (0, 255, 0), 3)
            cv2.putText(im, f'{name} - {confidence:.0f}', (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (51, 255, 255), 2)

            if name not in recognized:
                mark_attendance(name)
                recognized.add(name)
        else:  # If confidence is too high, it's an unknown person
            cv2.rectangle(im, (x, y), (x+w, y+h), (0, 0, 255), 3)
            cv2.putText(im, 'Unknown', (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            if 'Unknown Person' not in recognized:
                mark_attendance('Unknown Person')
                recognized.add('Unknown Person')

    cv2.imshow('OpenCV - Face Recognition', im)

    # Close the detection window 2 seconds after detecting a face
    if detected and start_time is not None and (time.time() - start_time) > 2:
        break

    if cv2.waitKey(10) == 27:  # Exit on 'Esc' key
        break

webcam.release()
cv2.destroyAllWindows()
