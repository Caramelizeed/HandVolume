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
```

## How It Works
1. **Webcam Input**: The program continuously captures video frames from the webcam.
2. **Hand Landmark Detection**: MediaPipe is used to detect and track the position of hand landmarks in each frame.
3. **Gesture Recognition**: By analyzing the hand landmarks, the system recognizes specific gestures to control audio properties:
   - **Volume Control**: Using the left hand's gestures (e.g., pinch or index finger up).
   - **Frequency Control**: Using the right hand's gestures (e.g., pinch or index finger up).
4. **Audio Generation**: The system generates a sine wave, and its frequency and amplitude are adjusted based on the detected hand gestures.
5. **Visualization**: The waveform and frequency of the generated audio are displayed in real-time on the UI, providing immediate feedback to the user.

## Features Explained

### Hand Gesture Recognition
The system uses MediaPipe to detect key hand landmarks (21 points per hand) and track their movements. These landmarks are interpreted to recognize gestures:
- **Left Hand**: Controls Volume.
- **Right Hand**: Controls Frequency.

### Audio Control
- **Frequency Control**: The right hand's gestures adjust the frequency of the sine wave. By moving your hand up or down, you can increase or decrease the frequency in real-time.
- **Volume Control**: The left hand's gestures control the volume of the sound. Pinching your fingers together decreases the volume, while keeping the index finger extended increases it.

### Visualization
- **Waveform Visualization**: Displays the sine wave in real-time, showing the audio waveform as it is played.
- **Frequency Graph**: Shows how the frequency changes over time, giving a visual representation of the pitch changes based on your hand gestures.

### Interactive UI
The graphical interface provides:
- **Volume Slider**: Displays the current volume and allows manual adjustments.
- **Frequency Slider**: Shows and allows changes to the current frequency.
- **Instructions Panel**: Displays tips on how to use the hand gestures to control the system.
- **Waveform Display**: Shows the audio waveform in real-time as you adjust volume and frequency.
- **Frequency Graph**: Displays a dynamic graph showing frequency changes over time.

## Installation & Setup
1. Clone this repository to your local machine:
```bash
git clone https://github.com/Caramelizeed/HandVolume.git
cd HandVolume
```

2. Install the necessary Python libraries:
```bash
pip install opencv-python mediapipe pygame numpy
```

3. Launch the program:
```bash
python hand_volume_control.py
```

You should see a webcam feed with the real-time hand tracking. Use hand gestures to control the audio properties.

## How to Use

### Volume Control:
- **Pinch Gesture**: Decreases the volume.
- **Index Finger Up**: Increases the volume.

### Frequency Control:
- **Pinch Gesture**: Decreases the frequency.
- **Index Finger Up**: Increases the frequency.

### UI Components:
- **Volume and Frequency Sliders**: These sliders display and allow you to adjust the volume and frequency manually.
- **Waveform Visualization**: Displays the generated sine wave in real-time.
- **Frequency Graph**: Displays changes in frequency over time.

### Example Use Case:
- You are listening to a sine wave generated at a default frequency of 440 Hz (A4 note).
- **Right hand (Frequency)**: Moving your right hand up will increase the pitch (frequency) of the sine wave to a higher note.
- **Left hand (Volume)**: Pinching your left hand will lower the volume, while extending your index finger will raise it.

## Troubleshooting
- **Issue with Webcam**: If the webcam feed does not appear, ensure that your camera is not being used by another application.
- **Gestures not detected properly**: Ensure your hands are clearly visible in the camera frame. MediaPipe works best when hands are centered and in clear view.
- **Audio Lag or Stuttering**: This may happen if the system resources are heavily used. Try closing other applications to free up resources.

## Future Improvements
- Add more advanced gesture recognition (e.g., controlling other audio parameters such as pitch modulation, effects).
- Implement multi-hand recognition for more complex controls.
- Add support for different audio formats or complex sound synthesis.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- **OpenCV**: For computer vision tasks like video capture and hand detection.
- **MediaPipe**: For the hand tracking and landmark detection.
- **Pygame**: For audio playback and waveform visualization.
- **NumPy**: For handling numerical operations related to sound synthesis.

## Author
HandVolume was created by Allison Burgers. For more information or contributions, feel free to reach out or open an issue on the repository!
