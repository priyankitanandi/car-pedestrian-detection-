import numpy as np
import cv2
import time

car_cascade = 'cascades/haarcascade_car.xml' # Specifies the path to the pre-trained Haar cascade classifier XML file for car detection
car_classifier = cv2.CascadeClassifier(car_cascade) 
capture = cv2.VideoCapture('cars.avi') # Opens the video file named “cars.avi” for reading frames

# Processing Frames
while capture.isOpened():

    response, frame = capture.read() # Reads the next frame from the video
    if response:

    	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Converts the frame to grayscale.

    	cars = car_classifier.detectMultiScale(gray, 1.2, 3) # Detects cars in the grayscale frame using the cascade classifier.

    	for (x, y, w, h) in cars: # The for loop iterates over the detected cars and draws rectangles around them using cv2.rectangle.
    		cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,0), 3)
    		cv2.imshow('Cars', frame) # Displays the frame with rectangles drawn around the detected cars.

    	if cv2.waitKey(1) & 0xFF == ord('q'):  # If the ‘q’ key is pressed, the loop breaks.
        	break
    else:
    	break    # If there’s no response (end of video), the loop also breaks.	

        	
capture.release()  # Releases the video file.
cv2.destroyAllWindows()  # Closes any open windows.