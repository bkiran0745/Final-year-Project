import cv2
import os
import base64
import requests
from ultralytics import YOLO
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, Attachment, FileContent, FileName, FileType, Disposition
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Path to your trained YOLO model
MlModel = "C:\\Users\\dines\\Downloads\\best.pt"

# Your SendGrid and Google API Keys from environment variables
SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def send_email(coordinates, files, pothole_data):
    latitude, longitude = coordinates
    severity_summary = "".join(
        f"<li>Pothole {idx + 1}: {info['severity']} - {info['repair_suggestion']}</li>"
        for idx, info in enumerate(pothole_data)
    )

    message = Mail(
        from_email="3br21ai035@bitm.edu.in",
        to_emails="saisudhajavalkar06@gmail.com",
        subject="Detailed Pothole Detection Report",
        html_content=f"""
        <p><b>Pothole Detection Summary</b></p>
        <p><b>Location Details:</b></p>
        <ul>
            <li>Latitude: {latitude}</li>
            <li>Longitude: {longitude}</li>
        </ul>
        <p><b>Total Potholes Detected:</b> {len(pothole_data)}</p>
        <p><b>Pothole Details:</b></p>
        <ul>
            {severity_summary}
        </ul>
        <p><b>Attachments:</b> Annotated frames are attached below for review.</p>
        """
    )

    for file_path in files:
        with open(file_path, 'rb') as f:
            file_data = f.read()
            encoded_file_data = base64.b64encode(file_data).decode()
            file_name = os.path.basename(file_path)

            encoded_file = Attachment(
                FileContent(encoded_file_data),
                FileName(file_name),
                FileType("image/jpeg"),
                Disposition("attachment")
            )

            message.add_attachment(encoded_file)

    try:
        sg = SendGridAPIClient(SENDGRID_API_KEY)
        response = sg.send(message)
        print(f"Email sent! Status Code: {response.status_code}")
    except Exception as e:
        print(f"Error sending email: {e}")

def get_google_coordinates():
    """
    Fetches the current geolocation using Google Maps Geolocation API.

    Returns:
        tuple: (latitude, longitude) if successful, else (None, None)
    """
    try:
        url = f"https://www.googleapis.com/geolocation/v1/geolocate?key={GOOGLE_API_KEY}"
        response = requests.post(url)
        data = response.json()
        
        if 'location' in data:
            latitude = data['location']['lat']
            longitude = data['location']['lng']
            return latitude, longitude
        else:
            print("Error: 'location' not found in the API response.")
    except Exception as e:
        print(f"Error fetching coordinates from Google API: {e}")
    return None, None

# Load the YOLOv8 model
model = YOLO(MlModel)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Unable to access the webcam.")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

output_dir = "C:\\Users\\dines\\Downloads\\Newin"
frames_dir = os.path.join(output_dir, "frames")
os.makedirs(frames_dir, exist_ok=True)

video_output_path = os.path.join(output_dir, "pothole_detection.avi")
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_writer = cv2.VideoWriter(video_output_path, fourcc, fps, (frame_width, frame_height))

print("Real-Time Pothole Detection Initialized. Press 'q' to quit.")

frame_count = 0
pothole_data = []

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to capture frame.")
        break

    results = model.predict(
        source=frame,
        save=True,
        project=output_dir,
        name="video_results"
    )

    annotated_frame = results[0].plot() if hasattr(results[0], "plot") else frame
    frame_output_path = os.path.join(frames_dir, f"frame_{frame_count}.jpg")
    cv2.imwrite(frame_output_path, annotated_frame)

    video_writer.write(annotated_frame)
    cv2.imshow('Pothole Detection', annotated_frame)

    for bbox in results[0].boxes.xyxy:  # Bounding box coordinates
        width = bbox[2] - bbox[0]  # x2 - x1
        height = bbox[3] - bbox[1]  # y2 - y1
        area = width * height

        if area < 5000:
            severity = "Minor"
            suggestion = "Cold Mix Asphalt repair recommended."
        elif 5000 <= area < 20000:
            severity = "Moderate"
            suggestion = "Hot Mix Asphalt repair recommended."
        else:
            severity = "Severe"
            suggestion = "Full Depth Repair required."

        pothole_data.append({
            "severity": severity,
            "repair_suggestion": suggestion
        })

    latlng = get_google_coordinates()
    if latlng and all(latlng):
        send_email(latlng, [frame_output_path], pothole_data)

    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
video_writer.release()
cv2.destroyAllWindows()

print(f"Pothole detection completed.")
print(f"Annotated video saved to: {video_output_path}")
print(f"Annotated frames saved to: {frames_dir}")
