from flask import Flask, request, render_template, send_file, Response, redirect, url_for
from werkzeug.utils import secure_filename
import io
from ultralytics import YOLO
import numpy as np
from PIL import Image
import cv2
import os
import base64

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'


class Detection:
    def __init__(self):
        self.model = YOLO("model/best.pt")

    def predict(self, img, classes=[], conf=0.5):
        if classes:
            results = self.model.predict(img, classes=classes, conf=conf)
        else:
            results = self.model.predict(img, conf=conf)

        return results

    def predict_and_detect(self, img, classes=[], conf=0.5, rectangle_thickness=2, text_thickness=1):
        results = self.predict(img, classes, conf)
        for result in results:
            for box in result.boxes:
                cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                              (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (255, 0, 0), rectangle_thickness)
                cv2.putText(img, f"{result.names[int(box.cls[0])]}",
                            (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                            cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), text_thickness)
        return img, results

    def detect_from_image(self, image):
        result_img, _ = self.predict_and_detect(image, classes=[], conf=0.5)
        return result_img


detection = Detection()


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return 'No file part'

        file = request.files['image']
        if file.filename == '':
            return 'No selected file'

        if file:
            # Read the image directly from the file object in memory
            img = Image.open(file).convert("RGB")
            img = np.array(img)
            img = cv2.resize(img, (512, 512))
            
            # Run the detection
            img = detection.detect_from_image(img)
            output = Image.fromarray(img)

            # Save the result to a BytesIO object instead of a file
            buf = io.BytesIO()
            output.save(buf, format="PNG")
            buf.seek(0)

            # Encode the image to base64 to embed in HTML
            img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

            # Render the template with the result
            return render_template('index2.html', result=img_base64, filename=file.filename)

    return render_template('index2.html')


if __name__ == '__main__':
    app.run()