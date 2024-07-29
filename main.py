import json
import time
from flask import Flask, render_template, request, Response, url_for
import cv2
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)

# Creating the Flask class object
@app.route('/cam')
def sign_page():
    return render_template("1.html")

@app.route('/mask', methods=['GET', 'POST'])
def mask():
    model = load_model('F:/MAJOR PROJECT/model-017.model')
    trained_data = cv2.CascadeClassifier('F:/MAJOR PROJECT/haarcascade_frontalface_default.xml')

    # Grabbing the webcam
    webcam = cv2.VideoCapture(0)
    labels_dict = {0: 'MASK', 1: 'NO MASK'}
    color_dict = {0: (0, 255, 0), 1: (0, 0, 255)}

    while True:
        frame_read, frame = webcam.read()
        # Convert to black and white
        greyscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect faces
        face_coordinates = trained_data.detectMultiScale(greyscale_frame)

        for (x, y, w, h) in face_coordinates:
            face_img = greyscale_frame[y:y + w, x:x + w]
            resized = cv2.resize(face_img, (100, 100))
            normalized = resized / 255.0
            reshaped = np.reshape(normalized, (1, 100, 100, 1))
            result = model.predict(reshaped)
            label = np.argmax(result, axis=1)[0]

            cv2.rectangle(frame, (x, y), (x + w, y + h), color_dict[label], 2)
            cv2.rectangle(frame, (x, y), (x + w, y - 40), color_dict[label], -1)
            cv2.putText(frame, labels_dict[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Face Mask Detection App", frame)
        key = cv2.waitKey(1)

        if key == 27:
            break

    webcam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    app.run(debug=True)
