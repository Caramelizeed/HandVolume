# HandVolume: Hand Gesture Audio Control System

HandVolume is an interactive hand gesture-based audio control system that uses computer vision to track hand gestures and adjust audio frequency and volume in real-time. The project uses OpenCV, MediaPipe, and Pygame to capture and interpret hand movements from the camera, allowing users to control sound properties such as frequency and volume via hand gestures.

## Features
- **Hand Gesture Recognition**: Utilizes MediaPipe to detect hand landmarks and track gestures.
- **Audio Control**: Adjusts frequency and volume based on hand gestures, with a sine wave generator that responds to changes.
- **Waveform and Frequency Visualization**: Displays real-time audio waveform and frequency graph.
- **Modern UI**: Enhanced UI with smooth animations, sliders, and visualizations for an immersive user experience.

## Requirements
- Python 3.x
- OpenCV
- Mediapipe
- Pygame
- NumPy

You can install the required libraries by running:
```bash
pip install opencv-python mediapipe pygame numpy
