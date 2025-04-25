# Only Testing with images

from ultralytics import YOLO

# Paths
pretrained_model = "C:\\Users\\dines\\Downloads\\best (1).pt"  # Path to the pretrained model
test_video = "C:\\Users\\dines\\Downloads\\input_01.mp4"       # Path to the test video
output_video = "C:\\Users\\dines\\Downloads\\Out_vid\\Vid_out.mp4"  # Output video path
output_dir = "C:\\Users\\dines\\Downloads\\Out_vid"            # Directory for saving results

# Load the YOLOv8 model
model = YOLO(pretrained_model)

# Test the model on a new video
print("Testing the model on a new video...")
results = model.predict(
    source=test_video,        # Path to the test video
    save=True,                # Save output predictions
    project=output_dir,       # Directory for saving results
    name="video_results"      # Name of the result directory
)

# Ensure the output video path is correctly set if necessary
print(f"Testing complete. Output video saved at: {output_video}")
