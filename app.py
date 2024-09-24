from flask import Flask, render_template, Response, request, redirect, url_for
import cv2 as cv
import mediapipe as mp
import time
import math
import numpy as np

app = Flask(__name__)

# Global Variables
TOTAL_BLINKS = 0
CEF_COUNTER = 0
CLOSED_EYES_FRAME = 3

# Eye landmark indices
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
mp_face_mesh = mp.solutions.face_mesh

def landmarksDetection(img, results):
    img_height, img_width = img.shape[:2]
    return [(int(point.x * img_width), int(point.y * img_height)) for point in results.multi_face_landmarks[0].landmark]

def euclideanDistance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def blinkRatio(landmarks):
    rh_right = landmarks[RIGHT_EYE[0]]
    rh_left = landmarks[RIGHT_EYE[8]]
    rv_top = landmarks[RIGHT_EYE[12]]
    rv_bottom = landmarks[RIGHT_EYE[4]]

    lh_right = landmarks[LEFT_EYE[0]]
    lh_left = landmarks[LEFT_EYE[8]]
    lv_top = landmarks[LEFT_EYE[12]]
    lv_bottom = landmarks[LEFT_EYE[4]]

    rhDistance = euclideanDistance(rh_right, rh_left)
    rvDistance = euclideanDistance(rv_top, rv_bottom)
    lvDistance = euclideanDistance(lv_top, lv_bottom)
    lhDistance = euclideanDistance(lh_right, lh_left)

    reRatio = rhDistance / rvDistance
    leRatio = lhDistance / lvDistance

    return (reRatio + leRatio) / 2

def generate_frames():
    global TOTAL_BLINKS, CEF_COUNTER
    TOTAL_BLINKS = 0  # Reset total blinks
    cap = cv.VideoCapture(0)

    with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
        start_time = time.time()
        while True:
            success, frame = cap.read()
            if not success:
                break
            
            frame = cv.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv.INTER_CUBIC)
            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                landmarks = landmarksDetection(frame, results)
                ratio = blinkRatio(landmarks)

                if ratio > 5.5:
                    CEF_COUNTER += 1
                    cv.putText(frame, 'Blinking', (30, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
                else:
                    if CEF_COUNTER > CLOSED_EYES_FRAME:
                        TOTAL_BLINKS += 1
                        CEF_COUNTER = 0

                cv.putText(frame, f'Total Blinks: {TOTAL_BLINKS}', (30, 150), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv.LINE_AA)

                for eye_indices in [LEFT_EYE, RIGHT_EYE]:
                    cv.polylines(frame, [np.array([landmarks[i] for i in eye_indices], dtype=np.int32)], isClosed=True, color=(0, 255, 0), thickness=1)

            if time.time() - start_time > 30:
                break

            ret, buffer = cv.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start():
    name = request.form.get('name')
    if name:
        return render_template('video_feed.html', name=name)
    return redirect(url_for('index'))

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/result')
def result():
    message = "Unhealthy Blinks:\n irregular blinks—keep an eye on your eye health." if TOTAL_BLINKS < 7 else "Healthy Blinks:\n healthy blinks—your eyes are in good shape!"
    return render_template('result.html', message=message)

@app.route('/restart')
def restart():
    global TOTAL_BLINKS
    TOTAL_BLINKS = 0
    return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(debug=True)
