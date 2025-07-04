
import torch
import cv2
import numpy as np
from datetime import datetime
from PIL import Image as PILImage
import io
import base64
import os
import time
import sys

# Firebase imports (assumed to be installed and initialized in the Colab notebook)
try:
    import firebase_admin
    from firebase_admin import credentials, firestore
    db = None
    if firebase_admin._apps:
        db = firestore.client()
        print("Firebase DB client successfully accessed in run_detector.py.")
    else:
        print("Firebase Admin SDK not initialized outside run_detector.py. Firestore logging will be skipped.")
except ImportError:
    db = None
    print("Firebase Admin SDK not installed or configured. Skipping Firestore logging in run_detector.py.")


def log_detection_to_firestore(label, xmin, ymin, xmax, ymax, confidence, timestamp, source_type="image", image_b64=None):
    """Logs detection data to Firestore."""
    global db

    if db is None:
        print("Firestore DB client not initialized. Skipping logging.") # This print can stay for initial verification
        return

    try:
        doc_ref = db.collection('detections').document()
        data = {
            'label': label,
            'bbox': {'xmin': float(xmin), 'ymin': float(ymin), 'xmax': float(xmax), 'ymax': float(ymax)},
            'confidence': float(confidence),
            'timestamp': timestamp,
            'source_type': source_type,
        }
        if image_b64:
            data['image_b64'] = image_b64

        doc_ref.set(data)
        # print(f"Logged detection: {label} at {timestamp}") # Uncomment this if you want verbose console logging
    except Exception as e:
        print(f"Error logging to Firestore: {e}")

def load_yolov5_model(weights_path='yolov5s.pt'):
    """Loads a YOLOv5 model."""
    try:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path, force_reload=False)
        model.conf = 0.25
        model.iou = 0.45
        return model
    except Exception as e:
        print(f"Error loading YOLOv5 model: {e}")
        return None

def detect_and_log_image(model, image_path, confidence_threshold=0.25):
    """Performs detection on a static image, draws boxes, and logs to Firestore."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not load image from {image_path}")
            return None

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = model(img_rgb)

        detections = results.pandas().xyxy[0]
        timestamp = datetime.now().isoformat()

        img_with_boxes = img.copy()

        buffered = io.BytesIO()
        PILImage.fromarray(img_rgb).save(buffered, format="JPEG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        print(f"\n--- Detections for {image_path} (Confidence Threshold: {confidence_threshold}) ---")
        if detections.empty:
            print("No objects detected above the confidence threshold.")
        else:
            print(detections[['name', 'confidence', 'xmin', 'ymin', 'xmax', 'ymax']])

        for *xyxy, conf, cls, name in detections.values:
            if conf >= confidence_threshold:
                xmin, ymin, xmax, ymax = map(int, xyxy)
                label = name
                confidence = float(conf)

                # RE-ENABLED FIRESTORE LOGGING
                log_detection_to_firestore(label, xmin, ymin, xmax, ymax, confidence, timestamp, "static_image", img_b64)

                color = (0, 255, 0)
                cv2.rectangle(img_with_boxes, (xmin, ymin), (xmax, ymax), color, 2)
                cv2.putText(img_with_boxes, f'{label} {confidence:.2f}', (xmin, ymin - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        print("--- End Detections ---")
        return img_with_boxes

    except Exception as e:
        print(f"Error during static image detection: {e}")
        return None

def run_webcam_detection(model, confidence_threshold=0.25):
    print("Starting webcam detection. Please grant camera access when prompted.")

    from IPython.display import display, Javascript, Image as IPDisplayImage
    from google.colab.output import eval_js
    from base64 import b64decode, b64encode

    def get_js_code():
        js = Javascript('''
            async function getPhoto(quality = 0.8) {
              const video = document.createElement('video');
              const stream = await navigator.mediaDevices.getUserMedia({video: true});
              document.body.appendChild(video);
              video.srcObject = stream;
              await video.play();
              google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);
              return new Promise((resolve) => {
                const once = () => {
                  const canvas = document.createElement('canvas');
                  canvas.width = video.videoWidth;
                  canvas.height = video.videoHeight;
                  canvas.getContext('2d').drawImage(video, 0, 0);
                  const dataUrl = canvas.toDataURL('image/jpeg', quality);
                  stream.getVideoTracks()[0].stop();
                  document.body.removeChild(video);
                  resolve(dataUrl);
                };
                video.addEventListener('ended', once);
                video.addEventListener('pause', once);
              });
            }
            async function takePhoto(quality = 0.8) {
              const video = document.createElement('video');
              const stream = await navigator.mediaDevices.getUserMedia({video: true});
              document.body.appendChild(video);
              video.srcObject = stream;
              await video.play();
              google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);
              let photo = '';
              try {
                await new Promise((resolve) => video.onplaying = resolve);
                const canvas = document.createElement('canvas');
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                canvas.getContext('2d').drawImage(video, 0, 0);
                photo = canvas.toDataURL('image/jpeg', quality);
              } finally {
                stream.getVideoTracks()[0].stop();
                video.remove();
              }
              return photo;
            }
            ''')
        display(js)

    get_js_code()

    def js_to_image(js_reply):
        jpeg_bytes = b64decode(js_reply.split(',')[1])
        img_np = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_np, flags=1)
        return img

    while True:
        try:
            js_reply = eval_js('takePhoto()')
            img = js_to_image(js_reply)

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = model(img_rgb)

            detections = results.pandas().xyxy[0]
            timestamp = datetime.now().isoformat()

            img_display = img.copy()

            buffered = io.BytesIO()
            PILImage.fromarray(img_rgb).save(buffered, format="JPEG")
            img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

            for *xyxy, conf, cls, name in detections.values:
                if conf >= confidence_threshold:
                    xmin, ymin, xmax, ymax = map(int, xyxy)
                    label = name
                    confidence = float(conf)

                    # RE-ENABLED FIRESTORE LOGGING
                    log_detection_to_firestore(label, xmin, ymin, xmax, ymax, confidence, timestamp, "webcam_live", img_b64)

                    color = (0, 255, 0)
                    cv2.rectangle(img_display, (xmin, ymin), (xmax, ymax), color, 2)
                    cv2.putText(img_display, f'{label} {confidence:.2f}', (xmin, ymin - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            _, ret_arr = cv2.imencode('.jpg', img_display)
            jpg_as_text = base64.b64encode(ret_arr).decode()
            display(IPDisplayImage(data=b64decode(jpg_as_text)), display_id='live_feed')

            eval_js('document.getElementById("live_feed").src = "data:image/jpeg;base64," + "' + jpg_as_text + '";')

        except Exception as e:
            print(f"Error in webcam loop: {e}")
            break


def run_video_file_detection(model, confidence_threshold=0.25, video_path=None):
    """Runs object detection on a video file, draws boxes, and logs to Firestore."""
    print("DEBUG: Inside run_video_file_detection function with the latest code.") # Keep this for now

    if video_path is None:
        print("Error: No video path provided for run_video_file_detection.")
        return

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        print(f"Processing video: {video_path} (FPS: {fps}, Resolution: {frame_width}x{frame_height})")

        from IPython.display import HTML
        from base64 import b64encode
        from google.colab.patches import cv2_imshow

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model(img_rgb)

            detections = results.pandas().xyxy[0]
            timestamp = datetime.now().isoformat()

            buffered = io.BytesIO()
            PILImage.fromarray(img_rgb).save(buffered, format="JPEG")
            img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

            print(f"\n--- Detections for frame {frame_count} (Confidence Threshold: {confidence_threshold}) ---")
            if detections.empty:
                print("No objects detected in this frame.")
            else:
                print(detections[['name', 'confidence', 'xmin', 'ymin', 'xmax', 'ymax']])

            for *xyxy, conf, cls, name in detections.values:
                if conf >= confidence_threshold:
                    xmin, ymin, xmax, ymax = map(int, xyxy)
                    label = name
                    confidence = float(conf)

                    # RE-ENABLED FIRESTORE LOGGING
                    log_detection_to_firestore(label, xmin, ymin, xmax, ymax, confidence, timestamp, "video_file", img_b64)

                    color = (0, 255, 0)
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                    cv2.putText(frame, f'{label} {confidence:.2f}', (xmin, ymin - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            cv2_imshow(frame)

            delay = int(1000 / fps)
            time.sleep(delay / 1000.0)

        cap.release()
        print("Video processing finished.")

    except Exception as e:
        print(f"Error during video file detection: {e}")

if __name__ == '__main__':
    print("This script is designed to be imported and run in Google Colab cells.")
    print("Please execute the Colab cells in sequence.")
