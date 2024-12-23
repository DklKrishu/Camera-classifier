from sklearn.svm import LinearSVC
import numpy as np
import cv2 as cv


class Model:
    def __init__(self):
        # Initialize the LinearSVC model
        self.model = LinearSVC()

    def train_model(self, counters):
        # Use Python lists to collect image data and labels
        img_list = []  
        class_list = []

        # Process images for class 1
        for i in range(1, counters[0]):
            img_path = f'1/frame{i}.jpg'
            img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)  # Read image as grayscale
            if img is not None:
                img_resized = cv.resize(img, (120, 140))  # Resize to 120x140
                img_flattened = img_resized.flatten()  # Flatten into a 1D array
                img_list.append(img_flattened)  # Append image data to list
                class_list.append(1)  # Append class label
            else:
                print(f"Error reading image: {img_path}")

        # Process images for class 2
        for i in range(1, counters[1]):
            img_path = f'2/frame{i}.jpg'
            img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)  # Read image as grayscale
            if img is not None:
                img_resized = cv.resize(img, (120, 140))  # Resize to 120x140
                img_flattened = img_resized.flatten()  # Flatten into a 1D array
                img_list.append(img_flattened)  # Append image data to list
                class_list.append(2)  # Append class label
            else:
                print(f"Error reading image: {img_path}")

        # Convert lists to NumPy arrays
        img_list = np.array(img_list, dtype=np.float32)
        class_list = np.array(class_list, dtype=np.int32)

        # Train the model if valid data exists
        if len(img_list) > 0:
            self.model.fit(img_list, class_list)
            print("Model successfully trained!")
        else:
            print("No valid images found for training!")

    def predict(self, frame):
        try:
            # Check if the frame is already grayscale
            if len(frame.shape) == 3:  # If it's a color image
                gray_frame = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
            else:
                gray_frame = frame  # Frame is already grayscale

            # Resize and flatten the frame
            resized_frame = cv.resize(gray_frame, (120, 140))  # Resize to 120x140
            flattened_frame = resized_frame.flatten()  # Flatten into a 1D array

            # Predict the class using the trained model
            prediction = self.model.predict([flattened_frame])
            return prediction[0]

        except Exception as e:
            print(f"Error during prediction: {e}")
            return None
