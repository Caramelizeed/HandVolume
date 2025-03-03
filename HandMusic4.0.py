import cv2
import mediapipe as mp
import numpy as np
import pygame
from pygame import mixer
import math
import time
from collections import deque

class HandGestureAudioControl:
    def __init__(self):
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        
        # Set a fixed window size for better UI layout
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Initialize mediapipe hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Custom drawing specs for better visualization
        self.landmark_drawing_spec = self.mp_draw.DrawingSpec(
            color=(0, 255, 0), thickness=2, circle_radius=4)
        self.connection_drawing_spec = self.mp_draw.DrawingSpec(
            color=(255, 255, 255), thickness=2)
            
        # Initialize pygame for audio
        pygame.init()
        mixer.init()
        
        # Audio settings
        self.sample_rate = 44100
        self.frequency = 440  # Starting frequency (A4)
        self.max_frequency = 880
        self.min_frequency = 220
        self.volume = 0.5  # Starting volume (0-1)
        
        # Generate initial sound
        self.generate_tone()
        
        # Start playing the sound
        self.sound_channel = mixer.Channel(0)
        self.sound_channel.play(self.sound, loops=-1)
        
        # Landmarks for gestures
        self.thumb_tip = 4
        self.index_tip = 8
        self.middle_tip = 12
        
        # Enhanced UI elements
        self.background_color = (20, 20, 30)  # Darker, modern background
        self.base_font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.8  # Larger font
        self.text_color = (220, 220, 220)  # Slightly off-white for better reading
        self.title_color = (0, 255, 255)  # Cyan
        self.accent_color = (65, 105, 225)  # Royal Blue
        self.volume_color = (50, 205, 50)  # Lime Green
        self.frequency_color = (255, 165, 0)  # Orange
        
        # Animation smoothing
        self.frequency_history = deque(maxlen=10)
        self.volume_history = deque(maxlen=10)
        for _ in range(10):
            self.frequency_history.append(self.frequency)
            self.volume_history.append(self.volume)
            
        # Improved waveform visualization - BIGGER
        self.waveform_width = 500
        self.waveform_height = 180
        self.waveform_points = []
        self.frequency_display_history = deque(maxlen=200)  # Store frequency history for visualization
        for _ in range(200):
            self.frequency_display_history.append(self.frequency)
        self.update_waveform()
        
        # Hand gesture trail
        self.trail_points = {
            "Left": deque(maxlen=20),
            "Right": deque(maxlen=20)
        }
        
        # FPS calculation
        self.prev_frame_time = 0
        self.new_frame_time = 0
        self.fps = 0
        
        # Default frame dimensions - will be updated in run loop
        self.frame_width = 1280
        self.frame_height = 720
        
    def generate_tone(self):
        """Generate a sine wave tone at the current frequency"""
        duration = 1  # seconds
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        wave = np.sin(2 * np.pi * self.frequency * t) * self.volume
        
        # Scale to 16-bit range and convert to int16
        wave = (wave * 32767).astype(np.int16)
        
        # Create stereo sound (make sure it's contiguous)
        stereo = np.empty((wave.shape[0], 2), dtype=np.int16)
        stereo[:, 0] = wave  # Left channel
        stereo[:, 1] = wave  # Right channel
        
        self.sound = pygame.sndarray.make_sound(stereo)
    
    def update_audio(self):
        """Update the playing audio with new frequency and volume"""
        # Smooth transitions by using moving average
        self.frequency = sum(self.frequency_history) / len(self.frequency_history)
        self.volume = sum(self.volume_history) / len(self.volume_history)
        
        # Update frequency display history
        self.frequency_display_history.append(self.frequency)
        
        self.generate_tone()
        self.sound_channel.stop()
        self.sound_channel.play(self.sound, loops=-1)
        
        # Update waveform visualization
        self.update_waveform()
        
    def update_waveform(self):
        """Update the waveform visualization"""
        points = 200
        
        # Generate current sine wave
        t = np.linspace(0, 4 * np.pi, points)  # Show more cycles for higher frequencies
        amplitude = self.waveform_height/2 * self.volume
        
        # Scale wavelength based on frequency (higher frequency = shorter wavelength)
        # Normalize based on min/max frequency range
        frequency_factor = (self.frequency - self.min_frequency) / (self.max_frequency - self.min_frequency)
        scaled_frequency = self.min_frequency + (self.max_frequency - self.min_frequency) * frequency_factor
        
        # Generate the waveform with the current frequency
        y_values = amplitude * np.sin(t * (scaled_frequency / self.min_frequency))
        
        self.waveform_points = []
        for i in range(points):
            x = int(i * (self.waveform_width / points))
            y = int(self.waveform_height/2 + y_values[i])
            self.waveform_points.append((x, y))
        
    def calculate_distance(self, p1, p2):
        """Calculate distance between two points"""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def detect_gestures(self, hand_landmarks, handedness):
        """Detect gestures and return actions"""
        landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
        
        # Check if index finger is pointing up
        is_finger_up = landmarks[self.index_tip][1] < landmarks[self.index_tip - 2][1]
        
        # Check if pinching (distance between thumb and index finger)
        pinch_distance = self.calculate_distance(landmarks[self.thumb_tip][:2], landmarks[self.index_tip][:2])
        is_pinching = pinch_distance < 0.05
        
        # Add to trail
        index_tip_px = (int(landmarks[self.index_tip][0] * self.frame_width), 
                         int(landmarks[self.index_tip][1] * self.frame_height))
        self.trail_points[handedness].append(index_tip_px)
        
        return is_finger_up, is_pinching
    
    def draw_modern_slider(self, image, title, value, min_val, max_val, x, y, width, color):
        """Draw a modern-looking slider with value"""
        normalized_value = (value - min_val) / (max_val - min_val)
        slider_value = int(normalized_value * width)
        
        # Draw label
        cv2.putText(image, f"{title}:", (x, y+10), 
                  self.base_font, self.font_scale*1.2, self.text_color, 2)
        
        # Draw slider background
        cv2.rectangle(image, (x + 150, y-15), (x + 150 + width, y+15), (50, 50, 50), -1)
        cv2.rectangle(image, (x + 150, y-15), (x + 150 + width, y+15), (100, 100, 100), 1)
        
        # Draw slider value
        cv2.rectangle(image, (x + 150, y-15), (x + 150 + slider_value, y+15), color, -1)
        
        # Draw value text
        if title == "Volume":
            text = f"{int(value * 100)}%"
        else:
            text = f"{int(value)} Hz"
            
        cv2.putText(image, text, (x + 150 + width + 20, y+10), 
                  self.base_font, self.font_scale*1.2, self.text_color, 2)
        
    def draw_frequency_graph(self, image, x, y, width, height):
        """Draw a graph showing frequency changes over time"""
        # Background
        cv2.rectangle(image, (x, y), (x + width, y + height), (30, 30, 40), -1)
        cv2.rectangle(image, (x, y), (x + width, y + height), (70, 70, 80), 2)
        
        # Title
        cv2.putText(image, "Frequency Over Time", (x + width//2 - 100, y - 15),
                   self.base_font, self.font_scale*1.2, self.frequency_color, 2)
        
        # Y-axis labels
        cv2.putText(image, f"{self.max_frequency} Hz", (x - 75, y + 20),
                   self.base_font, 0.7, self.text_color, 1)
        cv2.putText(image, f"{self.min_frequency} Hz", (x - 75, y + height - 10),
                   self.base_font, 0.7, self.text_color, 1)
        
        # Draw horizontal grid lines
        for i in range(1, 5):
            y_pos = y + (height * i // 5)
            cv2.line(image, (x, y_pos), (x + width, y_pos), (50, 50, 60), 1)
            
        # Draw the frequency history line
        freq_history = list(self.frequency_display_history)
        points = []
        
        for i, freq in enumerate(freq_history):
            if i >= width:
                break
                
            # Normalize frequency to fit in the graph
            normalized_freq = 1 - (freq - self.min_frequency) / (self.max_frequency - self.min_frequency)
            y_pos = int(y + normalized_freq * height)
            x_pos = x + width - i
            
            points.append((x_pos, y_pos))
            
        # Draw the connected lines
        for i in range(1, len(points)):
            cv2.line(image, points[i-1], points[i], self.frequency_color, 2)
            
        # Draw current frequency indicator
        current_freq = freq_history[0]
        normalized_current = 1 - (current_freq - self.min_frequency) / (self.max_frequency - self.min_frequency)
        current_y = int(y + normalized_current * height)
        cv2.circle(image, (x + width, current_y), 7, (255, 255, 255), -1)
        cv2.circle(image, (x + width, current_y), 5, self.frequency_color, -1)
    
    def draw_modern_waveform(self, image, x, y):
        """Draw a modern-looking waveform visualization"""
        # Background for waveform
        cv2.rectangle(image, (x - 10, y - 10),
                     (x + self.waveform_width + 10, y + self.waveform_height + 10), (30, 30, 40), -1)
        cv2.rectangle(image, (x - 10, y - 10),
                     (x + self.waveform_width + 10, y + self.waveform_height + 10), (70, 70, 80), 2)
        
        # Title
        cv2.putText(image, "Current Waveform", (x + self.waveform_width//2 - 110, y - 15),
                   self.base_font, self.font_scale*1.2, self.volume_color, 2)
        
        # Center line
        cv2.line(image, (x, y + self.waveform_height//2), 
                (x + self.waveform_width, y + self.waveform_height//2), 
                (70, 70, 80), 1)
                   
        # Draw the waveform
        points = [(x + x_point, y + y_point) for x_point, y_point in self.waveform_points]
        
        # Draw the wave
        for i in range(1, len(points)):
            cv2.line(image, points[i-1], points[i], self.volume_color, 2)
            
        # Add labels
        cv2.putText(image, f"Frequency: {int(self.frequency)} Hz", 
                  (x + 15, y + self.waveform_height - 20),
                  self.base_font, 0.7, self.frequency_color, 2)
        
        cv2.putText(image, f"Volume: {int(self.volume * 100)}%", 
                  (x + self.waveform_width - 180, y + self.waveform_height - 20),
                  self.base_font, 0.7, self.volume_color, 2)
            
    def draw_instructions_panel(self, image, x, y, width, height):
        """Draw an instructions panel"""
        # Background
        cv2.rectangle(image, (x, y), (x + width, y + height), (30, 30, 40), -1)
        cv2.rectangle(image, (x, y), (x + width, y + height), (70, 70, 80), 2)
        
        # Title
        cv2.putText(image, "CONTROLS", (x + width//2 - 60, y + 30),
                   self.base_font, self.font_scale*1.2, self.accent_color, 2)
        
        # Left hand instructions
        cv2.putText(image, "LEFT HAND", (x + 20, y + 70), 
                  self.base_font, 0.9, (255, 80, 80), 2)
        cv2.putText(image, "Controls Volume", (x + 20, y + 95), 
                  self.base_font, 0.8, self.text_color, 1)
                  
        # Right hand instructions
        cv2.putText(image, "RIGHT HAND", (x + 20, y + 130), 
                  self.base_font, 0.9, (80, 80, 255), 2)
        cv2.putText(image, "Controls Frequency", (x + 20, y + 155), 
                  self.base_font, 0.8, self.text_color, 1)
        
        # Gesture instructions
        cv2.putText(image, "INDEX FINGER UP", (x + width//2 + 20, y + 70), 
                  self.base_font, 0.8, self.text_color, 1)
        cv2.putText(image, "Increases Value", (x + width//2 + 30, y + 95), 
                  self.base_font, 0.7, self.text_color, 1)
        
        cv2.putText(image, "PINCH GESTURE", (x + width//2 + 20, y + 130), 
                  self.base_font, 0.8, self.text_color, 1)
        cv2.putText(image, "Decreases Value", (x + width//2 + 30, y + 155), 
                  self.base_font, 0.7, self.text_color, 1)
        
        # Quit instruction
        cv2.putText(image, "Press 'q' to quit", (x + width//2 - 70, y + height - 20), 
                  self.base_font, 0.7, self.text_color, 1)
            
    def draw_ui(self, image):
        """Draw enhanced UI elements onto the image"""
        h, w, _ = image.shape
        self.frame_width, self.frame_height = w, h
        
        # Create a clean overlay for the UI
        ui_overlay = np.zeros_like(image)
        
        # Top panel - Make taller
        cv2.rectangle(ui_overlay, (0, 0), (w, 180), self.background_color, -1)
        cv2.line(ui_overlay, (0, 180), (w, 180), self.accent_color, 3)
        
        # Bottom panel - Make taller
        cv2.rectangle(ui_overlay, (0, h-200), (w, h), self.background_color, -1)
        cv2.line(ui_overlay, (0, h-200), (w, h-200), self.accent_color, 3)
        
        # Blend the UI overlay with the original image
        alpha = 0.8
        cv2.addWeighted(ui_overlay, alpha, image, 1 - alpha, 0, image)
        
        # Title with accent
        title_bg = image.copy()
        cv2.rectangle(title_bg, (w//2 - 280, 10), (w//2 + 280, 60), self.accent_color, -1)
        cv2.addWeighted(title_bg, 0.7, image, 0.3, 0, image)
        cv2.putText(image, "Hand Gesture Audio Control", (w//2 - 250, 45), 
                    self.base_font, 1.3, (255, 255, 255), 2)
        
        # Modern sliders for Volume and Frequency - Make wider
        self.draw_modern_slider(image, "Volume", self.volume, 0, 1, 30, 110, 300, self.volume_color)
        self.draw_modern_slider(image, "Frequency", self.frequency, self.min_frequency, self.max_frequency, 30, 160, 300, self.frequency_color)
        
        # Waveform and frequency visualizations - Place side by side with larger sizes
        
        # Left side: Frequency graph
        graph_x = 30
        graph_y = 200
        graph_width = w//2 - 60
        graph_height = h - 420
        self.draw_frequency_graph(image, graph_x, graph_y, graph_width, graph_height)
        
        # Right side: Waveform
        # Update waveform dimensions to match the new layout
        self.waveform_width = w//2 - 60
        self.update_waveform()  # Recalculate with new width
        self.draw_modern_waveform(image, w//2 + 30, graph_y)
        
        # Instructions panel at the bottom
        instructions_x = 30
        instructions_y = h - 190
        instructions_width = w - 60
        instructions_height = 180
        self.draw_instructions_panel(image, instructions_x, instructions_y, instructions_width, instructions_height)
        
        # Draw hand gesture trails
        for hand, trail in self.trail_points.items():
            color = (255, 80, 80) if hand == "Left" else (80, 80, 255)
            for i in range(1, len(trail)):
                if trail[i-1] and trail[i]:  # Make sure points exist
                    thickness = int(np.sqrt(64/float(i+1)) * 2.5)
                    cv2.line(image, trail[i-1], trail[i], color, thickness)
        
        # Show FPS in top-right corner with background
        fps_text = f"FPS: {int(self.fps)}"
        fps_size = cv2.getTextSize(fps_text, self.base_font, self.font_scale, 1)[0]
        cv2.rectangle(image, (w - fps_size[0] - 20, 10), (w - 10, 40), (0, 0, 0), -1)
        cv2.putText(image, fps_text, (w - fps_size[0] - 15, 30), 
                  self.base_font, self.font_scale, (0, 255, 0), 1)
        
        return image
        
    def run(self):
        """Main loop for hand tracking and audio control"""
        audio_update_needed = False
        
        while self.cap.isOpened():
            success, image = self.cap.read()
            if not success:
                print("Failed to capture image")
                continue
            
            # Calculate FPS
            self.new_frame_time = time.time()
            self.fps = 1/(self.new_frame_time - self.prev_frame_time) if self.prev_frame_time > 0 else 30
            self.prev_frame_time = self.new_frame_time
            
            # Flip the image horizontally for a mirror effect
            image = cv2.flip(image, 1)
            
            # Convert to RGB and process with MediaPipe
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.hands.process(image_rgb)
            
            # Draw hand landmarks
            if results.multi_hand_landmarks:
                left_hand_present = False
                right_hand_present = False
                
                for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    # Draw with custom style
                    self.mp_draw.draw_landmarks(
                        image, 
                        hand_landmarks, 
                        self.mp_hands.HAND_CONNECTIONS,
                        self.landmark_drawing_spec,
                        self.connection_drawing_spec)
                    
                    # Determine if left or right hand
                    handedness = results.multi_handedness[hand_idx].classification[0].label
                    
                    if handedness == "Left":
                        left_hand_present = True
                        left_finger_up, left_pinching = self.detect_gestures(hand_landmarks, handedness)
                        
                        # Left hand controls volume
                        if left_finger_up and not left_pinching:
                            new_volume = min(1.0, self.volume + 0.01)
                            self.volume_history.append(new_volume)
                            audio_update_needed = True
                        elif left_pinching:
                            new_volume = max(0.0, self.volume - 0.01)
                            self.volume_history.append(new_volume)
                            audio_update_needed = True
                        else:
                            self.volume_history.append(self.volume)
                            
                    elif handedness == "Right":
                        right_hand_present = True
                        right_finger_up, right_pinching = self.detect_gestures(hand_landmarks, handedness)
                        
                        # Right hand controls frequency
                        if right_finger_up and not right_pinching:
                            new_frequency = min(self.max_frequency, self.frequency + 2)
                            self.frequency_history.append(new_frequency)
                            audio_update_needed = True
                        elif right_pinching:
                            new_frequency = max(self.min_frequency, self.frequency - 2)
                            self.frequency_history.append(new_frequency)
                            audio_update_needed = True
                        else:
                            self.frequency_history.append(self.frequency)
            
            # Update audio if needed
            if audio_update_needed:
                self.update_audio()
                audio_update_needed = False
            else:
                # Still update frequency display history even when no changes
                self.frequency_display_history.append(self.frequency)
            
            # Draw UI elements
            image = self.draw_ui(image)
            
            # Show the image
            cv2.imshow('Hand Gesture Audio Control', image)
            
            # Exit on 'q' press
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
        
        # Clean up
        self.cap.release()
        cv2.destroyAllWindows()
        pygame.quit()

if __name__ == "__main__":
    controller = HandGestureAudioControl()
    controller.run()