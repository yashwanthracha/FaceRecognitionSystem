from flask import Flask, render_template, Response, request
import cv2
import numpy as np
from infere_mtcnn import Infer  # Import your Infer class
import os

app = Flask(__name__)
# Initialize the video capture object
video_capture = cv2.VideoCapture(0)  # 0 for default webcam

# Initialize the Infer class
infer = Infer()

def gen_frames():
    while True:
        success, frame = video_capture.read()  # Read frame from webcam
        if not success:
            break
        else:
            # Process frame
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # Yield frame

@app.route('/')
def index():
    return render_template('index.html')  # Render index.html template

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/search_faces', methods=['GET', 'POST'])
def search_faces_endpoint():
    if request.method == 'POST':
        # Perform face search on the captured frame
        success, frame = video_capture.read()
        
        if not success:
            return "Failed to capture frame from webcam."

        # Save frame to a temporary file
        temp_img_path = 'temp_frame.jpg'
        cv2.imwrite(temp_img_path, frame)
        
        # Perform face search using Infer class
        results = infer.main(temp_img_path)
        
        # Append results to the CSV file
        infer.save_results_to_csv(results)

        # Display the results
        return render_template('results.html', results=results)
    
    return render_template('index.html')  # Render index.html if GET request

if __name__ == '__main__':
    app.run(debug=True)
