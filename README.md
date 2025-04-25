# Final-year-Project
This project is a real-time object detection system built with Python and YOLOv5. It takes in video input, runs detection using a pretrained model (`best.pt`), and outputs results based on various scripts like ‘Main.py’, ‘detect.py’, and others.

Project Structure

‘’’
├── best.pt        # Trained YOLOv5 model weights
├── ki.mp4         # Input video file for object detection
├── CGPS.py        # GPS-related logic (if applicable)
├── detect.py      # YOLO detection logic
├── Main.py        # Main entry script
├── Main1.py       # Alternative main script
├── Real_check.py  # Validation or check logic
├── Real_detect.py # Real-time detection handling
├── trial.py       # Experimental script
‘’’

Requirements

- Python 3.7+
- PyTorch
- OpenCV
- NumPy
- YOLOv5 dependencies (if using Ultralytics version)

Install dependencies with:

“ bash pip install torch torchvision torchaudio opencv-python numpy ”

How to Run

1. Clone the repository:

““”
bash git clone https://github.com/bkiran0745/Final-year-Project.git
cd Final-year-Project
“””

2. Run the detection:

“ bash python Main.py “

> You can also try:
> “ python Main1.py “
> “ python Real_detect.py “
> “ python detect.py “

3. Change input video (optional):

Make sure ‘ki.mp4’ is in the project directory or modify the script to use a different input source (e.g., webcam or another video).

Example Output

The system will open a video window with real-time object detection results displayed.

Model Info

The detection uses ‘best.pt’, a custom-trained YOLOv5 model. Ensure this file is placed correctly in the directory.
