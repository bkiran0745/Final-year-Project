import cv2
import os
from ultralytics import YOLO

# Path to your trained YOLO model (ensure this is your pothole detection model)
MlModel = "C:\\Users\\dines\\Downloads\\best (1).pt"

# Load the trained YOLOv8 model
model = YOLO(MlModel)

# Open webcam feed
cap = cv2.VideoCapture(0)

# Check if the webcam is accessible
if not cap.isOpened():
    print("Error: Unable to access the webcam.")
    exit()

# Get the video frame width, height, and FPS
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Create output directories for saving video and frames
output_dir = "C:\\Users\\dines\\Downloads\\Out_vid\\frames"
frames_dir = os.path.join(output_dir, "frames")
os.makedirs(frames_dir, exist_ok=True)

# Define the codec and initialize the video writer
video_output_path = os.path.join(output_dir, "pothole_detection.avi")
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for .avi format
video_writer = cv2.VideoWriter(video_output_path, fourcc, fps, (frame_width, frame_height))

print("Real-Time Pothole Detection Initialized. Press 'q' to quit.")

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to capture frame.")
        break

    # Run YOLOv8 on the frame
    results = model.predict(
    source=frame,        # Path to the test video
    save=True,                # Save output predictions
    project=output_dir,       # Directory for saving results
    name="video_results"      # Name of the result directory
)

    # Annotate the frame with pothole detections
    annotated_frame = results[0].plot() if hasattr(results[0], "plot") else frame

    # Save the annotated frame as an image
    frame_output_path = os.path.join(frames_dir, f"frame_{frame_count}.jpg")
    cv2.imwrite(frame_output_path, annotated_frame)

    # Write the annotated frame to the video
    video_writer.write(annotated_frame)

    # Display the annotated frame
    cv2.imshow('Pothole Detection', annotated_frame)
    frame_count += 1

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
video_writer.release()
cv2.destroyAllWindows()

print(f"Pothole detection completed.")
print(f"Annotated video saved to: {video_output_path}")
print(f"Annotated frames saved to: {frames_dir}")
