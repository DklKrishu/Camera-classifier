import tkinter as tk
from tkinter import simpledialog, messagebox
import cv2 as cv
import os
import PIL.Image, PIL.ImageTk
import model  # Import your model module
import camera  # Import your camera module


class App:
    def __init__(self, window=tk.Tk(), window_title="Camera Classifier"):
        self.window = window
        self.window_title = window_title

        self.counters = [1, 1]  # Counter for each class
        self.model = model.Model()  # Initialize the model
        self.auto_predict = False  # Auto prediction toggle
        self.camera = camera.Camera()  # Initialize the camera

        self.init_gui()  # Set up the GUI
        self.delay = 15
        self.update()

        self.window.attributes("-topmost", True)
        self.window.mainloop()

    def init_gui(self):
        self.canvas = tk.Canvas(self.window, width=self.camera.width, height=self.camera.height)
        self.canvas.pack()

        self.btn_toggleauto = tk.Button(self.window, text="Auto Prediction", width=50, command=self.auto_predict_toggle)
        self.btn_toggleauto.pack(anchor=tk.CENTER, expand=True)

        self.classname_one = simpledialog.askstring("Classname One", "Enter the name of the first class:", parent=self.window)
        self.classname_two = simpledialog.askstring("Classname Two", "Enter the name of the second class:", parent=self.window)

        self.btn_class_one = tk.Button(self.window, text=self.classname_one, width=50, command=lambda: self.save_for_class(1))
        self.btn_class_one.pack(anchor=tk.CENTER, expand=True)

        self.btn_class_two = tk.Button(self.window, text=self.classname_two, width=50, command=lambda: self.save_for_class(2))
        self.btn_class_two.pack(anchor=tk.CENTER, expand=True)

        self.btn_train = tk.Button(self.window, text="Train Model", width=50, command=self.train_model_wrapper)
        self.btn_train.pack(anchor=tk.CENTER, expand=True)

        self.btn_predict = tk.Button(self.window, text="Predict", width=50, command=self.predict)
        self.btn_predict.pack(anchor=tk.CENTER, expand=True)

        self.btn_reset = tk.Button(self.window, text="Reset", width=50, command=self.reset)
        self.btn_reset.pack(anchor=tk.CENTER, expand=True)

        self.class_label = tk.Label(self.window, text="CLASS")
        self.class_label.config(font=("Arial", 20))
        self.class_label.pack(anchor=tk.CENTER, expand=True)

    def train_model_wrapper(self):
        try:
            print(f"Training model with counters: {self.counters}")
            self.model.train_model(self.counters)
        except Exception as e:
            print(f"Error during training: {e}")
            messagebox.showerror("Training Error", f"An error occurred during training: {str(e)}")

    def auto_predict_toggle(self):
        self.auto_predict = not self.auto_predict

    def save_for_class(self, class_num):
        ret, frame = self.camera.get_frame()
        if not os.path.exists("1"):
            os.mkdir("1")
        if not os.path.exists("2"):
            os.mkdir("2")

        cv.imwrite(f'{class_num}/frame{self.counters[class_num-1]}.jpg', cv.cvtColor(frame, cv.COLOR_RGB2GRAY))
        img = PIL.Image.open(f'{class_num}/frame{self.counters[class_num - 1]}.jpg')
        img.thumbnail((150, 150), PIL.Image.Resampling.LANCZOS)
        img.save(f'{class_num}/frame{self.counters[class_num - 1]}.jpg')

        self.counters[class_num - 1] += 1

    def reset(self):
        for folder in ['1', '2']:
            for file in os.listdir(folder):
                file_path = os.path.join(folder, file)
                if os.path.isfile(file_path):
                    os.unlink(file_path)

        self.counters = [1, 1]
        self.model = model.Model()  # Reinitialize the model
        self.class_label.config(text="CLASS")

    def update(self):
        if self.auto_predict:
            print(self.predict())

        ret, frame = self.camera.get_frame()

        if ret:
            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.window.after(self.delay, self.update)

    def predict(self):
        ret, frame = self.camera.get_frame()
        if not ret:
            print("Error: Could not get frame.")
            messagebox.showerror("Prediction Error", "Could not retrieve a valid frame for prediction.")
            return None

        try:
            # Ensure the frame is grayscale
            if len(frame.shape) == 3:  # Check if it's RGB
                frame = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)

            # Resize to match training data dimensions
            resized_frame = cv.resize(frame, (120, 140))  # Use dimensions from the model training

            # Flatten the frame
            flattened_frame = resized_frame.flatten()

            # Predict the class
            prediction = self.model.predict(flattened_frame)

            # Update the GUI label based on prediction
            if prediction == 1:
                self.class_label.config(text=self.classname_one)
                return self.classname_one
            elif prediction == 2:
                self.class_label.config(text=self.classname_two)
                return self.classname_two
            else:
                self.class_label.config(text="Unknown Class")
                return "Unknown Class"

        except Exception as e:
            print(f"Error during prediction: {e}")
            messagebox.showerror("Prediction Error", f"An error occurred during prediction: {str(e)}")
            return None
