import cv2
import os
import tkinter as tk
from tkinter import simpledialog, messagebox
from tkinter import ttk

class FaceCaptureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Capture")

        self.create_widgets()
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.web_cam = cv2.VideoCapture(0)
        self.count = 1

    def create_widgets(self):
        # Create UI Elements
        tk.Label(self.root, text="Enter the name of the person:").pack(pady=5)
        self.name_entry = tk.Entry(self.root)
        self.name_entry.pack(pady=5)

        tk.Label(self.root, text="Select Department:").pack(pady=5)
        self.department_entry = tk.Entry(self.root)
        self.department_entry.pack(pady=5)

        self.capture_button = tk.Button(self.root, text="Start Capture", command=self.start_capture)
        self.capture_button.pack(pady=10)

        self.quit_button = tk.Button(self.root, text="Quit", command=self.root.quit)
        self.quit_button.pack(pady=10)

    def start_capture(self):
        name = self.name_entry.get().strip()
        department = self.department_entry.get().strip()

        if not name or not department:
            messagebox.showwarning("Input Error", "Please fill in all fields.")
            return

        dataset_folder = f'datasets/{name}_{department}'
        if not os.path.isdir(dataset_folder):
            os.makedirs(dataset_folder)

        self.capture_faces(dataset_folder)

    def capture_faces(self, folder):
        (width, height) = (130, 100)  # Size of the face images
        while self.count <= 50:  # Capture 50 face samples for better dataset quality
            print(f"Capturing image {self.count}...")
            ret, im = self.web_cam.read()
            if not ret:
                messagebox.showerror("Camera Error", "Unable to access the webcam.")
                break

            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
                face = gray[y:y + h, x:x + w]
                face_resize = cv2.resize(face, (width, height))
                cv2.imwrite(f'{folder}/{self.count}.png', face_resize)
                self.count += 1

            cv2.imshow('OpenCV - Face Capture', im)
            if cv2.waitKey(10) == 27:  # Exit if 'Esc' key is pressed
                break

        self.web_cam.release()
        cv2.destroyAllWindows()
        messagebox.showinfo("Capture Complete", "Face capture complete.")

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceCaptureApp(root)
    root.mainloop()
