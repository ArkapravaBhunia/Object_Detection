import torch  # Import PyTorch library
import numpy as np  # Import NumPy library
import cv2  # Import OpenCV library
from time import time, sleep  # Import time and sleep functions from time library
from ultralytics import YOLO  # Import YOLO class from ultralytics library
import pyttsx3  # Import pyttsx3 library for text-to-speech
import threading  # Import threading library for multi-threading
from queue import Queue  # Import Queue class from queue library
import supervision as sv  # Import supervision library



class ObjectDetection:

    def __init__(self, capture_index):
        # Set capture index
        self.capture_index = capture_index

        # create a stop event for the thread
        self.stop_event = threading.Event()

        # Check if CUDA is available and set device accordingly
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Set device to CUDA if available, else CPU
        print("Using Device: ", self.device)  # Print the device being used

        # Load the YOLO model
        self.model = self.load_model()

        # Get class names from the model
        self.CLASS_NAMES_DICT = self.model.model.names

        # Initialize box annotator for drawing bounding boxes on the frame
        self.box_annotator = sv.BoxAnnotator(sv.ColorPalette.default(), thickness=3, text_thickness=3, text_scale=1.5)

         # Set last announcement time to current time
        self.last_announcement_time = time()
        # Set announcement interval to 5 seconds
        self.announcement_interval = 5


    def load_model(self):
        # Load a pretrained YOLOv8n model and fuse it for faster inference
        model = YOLO("asset\\best39.pt")
        # Fuse the model for faster inference
        model.fuse()

        return model


    def predict(self, frame):
        # Run inference on the frame using the YOLO model and return the results
        results = self.model(frame)

        return results


    def plot_bboxes(self, results, frame):

        # function for writing the result_string in result.txt
        def write_result(result_string):
            with open('result.txt', mode='w') as result_file:
                result_file.write(result_string)

        result_string = ''  # Initialize result string
        det = results[0].boxes  # Get detection boxes from results
        if len(det) == 0:  # If no detections were found
            result_string = 'no detections'  # Set result string to 'no detections'
        else:  # If detections were found
            for c in det.cls.unique():  # For each unique class in the detections
                n = (det.cls == c).sum()  # Count the number of detections for that class
                result_string += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)},"  # Add the count and class name to the result string

            write_result(result_string)  # Write the result string to a file


            # Setup detections for visualization
            detections = sv.Detections(
                xyxy=results[0].boxes.xyxy.cpu().numpy(),  # Get bounding box coordinates
                confidence=results[0].boxes.conf.cpu().numpy(),  # Get confidence scores
                class_id=results[0].boxes.cls.cpu().numpy().astype(int),  # Get class IDs
            )

            # Format custom labels for visualization from detections variable
            self.labels = [f"{self.CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"  # Create labels with class name and confidence score
                           for _,_, confidence, class_id, tracker_id
                           in detections]

            # Annotate and display frame
            frame = self.box_annotator.annotate(scene=frame, detections=detections, labels=self.labels)  # Annotate the frame with bounding boxes and labels

            # Check if any object is too close
            thresholds = {'person': 0.6, 'car': 0.4}  # Define thresholds for each object class
            warning_given = False  # Initialize warning_given flag to False
            for det in results[0].boxes:  # For each detection in the results
                class_name = self.model.names[int(det.cls)]  # Get the class name of the detection
                if class_name in thresholds:  # If the class name is in the thresholds
                    print(det.xyxy.shape)
                    x1, y1, x2, y2 = det.xyxy[0]  # Get the coordinates of the bounding box
                   

                    bbox_area = (x2 - x1) * (y2 - y1)  # Calculate the area of the bounding box
                    frame_area = frame.shape[0] * frame.shape[1]  # Calculate the area of the frame
                    area_percentage = bbox_area / frame_area  # Calculate the percentage of the frame area occupied by the bounding box
                    if area_percentage > thresholds[class_name]:  # If the percentage is greater than the threshold for that class
                        warning_given = True  # Set warning_given flag to True
                        warning_message = f"Warning: {class_name} is too close"  # Generate a warning message that the object is too close
                        print(warning_message)  # Print the warning message
                        self.speak_result(warning_message)  # Use text-to-speech engine to speak the warning message

            # Announce result every announcement interval (5 seconds) if no warning was given
            if not warning_given and time() - self.last_announcement_time > self.announcement_interval:
                self.speak_result(result_string)  # Use text-to-speech engine to speak the detection
                self.last_announcement_time = time()  # Update last announcement time to current time

        # Return annotated frame
        return frame
    

    def _speak_result(self, result_string):
        print(f"_speak_result called with result_string: {result_string}")  # Print the result string (for debug)

        # as pyttsx3 engine is not thread-safe and cannot be shared between multiple threads
        # so we create a new local pyttsx3 engine instance for each thread instead of sharing the same engine instance
        engine = pyttsx3.init() # Initialize a new pyttsx3 engine instance
        
        engine.say(result_string)  # Use the pyttsx3 engine to speak the result string
        engine.runAndWait()  # Wait for the pyttsx3 engine to finish speaking

        if engine._inLoop:  # If the pyttsx3 engine is in a loop
            engine= None  # Set the pyttsx3 engine to None

        if self.stop_event.is_set():  # If the stop event is set
            print("exiting thread")  # Print a message that the thread is exiting(for debug)  
            return  # Return from the method to exit the thread
            
        

    # create a separate thread to run the speech generation code asynchronously
    def speak_result(self, result_string):
        thread = threading.Thread(target=self._speak_result, args=(result_string,))  # Create a new thread to run the _speak_result method   
        thread.start()  # start the thread




    def __call__(self):
        cap = cv2.VideoCapture(self.capture_index)  # Open video capture device for corresponding capture_index
        assert cap.isOpened()  # Assert that the capture device is opened
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Set frame width
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Set frame height

       

        while True:
            start_time = time()  # Get start time

            ret, frame = cap.read()  # Read a frame from the capture device
            assert ret  # Assert that the frame was read successfully

            results = self.predict(frame)  # Run inference on the frame
            if cv2.waitKey(5) & 0xFF == 27:
                self.stop_event.set()  # Set stop event if ESC key is pressed
                print("stop_event set")
                break

            frame = self.plot_bboxes(results, frame)  # Plot bounding boxes on the frame
            if cv2.waitKey(5) & 0xFF == 27:
                self.stop_event.set()  # Set stop event if ESC key is pressed
                print("stop_event set")
                break

            end_time = time()  # Get end time

            fps = 1 / (np.round(end_time - start_time, 2) + 1e-6)  # Calculate Frames per Second

            cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)  # Put FPS text on the frame

            cv2.imshow('Blind Assistant System', frame)  # Show the frame in a window

            if cv2.waitKey(5) & 0xFF == 27:
                self.stop_event.set()  # Set stop event if ESC key is pressed
                print("stop_event set")
                break

            if cv2.getWindowProperty('Blind Assistant System', cv2.WND_PROP_VISIBLE) < 1:
                self.stop_event.set()  # Set stop event if window is closed
                print("stop_event set")
                break
        
        cap.release()  # Release the capture device
        cv2.destroyAllWindows()  # Destroy all windows


if __name__ == '__main__':
    detector = ObjectDetection(capture_index=0)   # Create an ObjectDetection instance with capture index of 0
    detector()  # Run the object detection