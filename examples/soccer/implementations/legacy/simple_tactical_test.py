"""
Simple tactical analysis test without team classification
Tests jersey assignment and homography transformation
"""

import argparse
import os
import warnings
import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO
from typing import List, Dict, Optional, Tuple

from sports.annotators.soccer import draw_pitch, draw_points_on_pitch
from sports.common.ball import BallTracker, BallAnnotator
from sports.common.view import ViewTransformer
from sports.configs.soccer import SoccerPitchConfiguration

# Import our custom modules
from jersey_assignment import JerseyAssignmentManager

# Suppress repetitive warnings
warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")
warnings.filterwarnings("ignore", category=UserWarning, module="supervision")
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Suppress urllib3 SSL warnings
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Constants
MAX_FRAMES = 100  # Test with 100 frames
BALL_CLASS_ID = 0
PLAYER_CLASS_ID = 1
GOALKEEPER_CLASS_ID = 2
REFEREE_CLASS_ID = 3
BALL_RADAR_CLASS_ID = 4

# Colors for different teams and ball
COLORS = ['#FF1493', '#00BFFF', '#FF6347', '#FFD700', '#32CD32', '#FF69B4', '#FF0000']  # Red for ball

# Soccer pitch configuration
CONFIG = SoccerPitchConfiguration()

def render_radar_with_jerseys(player_detections: sv.Detections, ball_detections: sv.Detections, 
                            keypoints: sv.KeyPoints, manager: JerseyAssignmentManager) -> np.ndarray:
    """Render radar view with player positions and jersey numbers"""
    if len(player_detections) == 0 and len(ball_detections) == 0:
        return draw_pitch(config=CONFIG)
    
    mask = (keypoints.xy[0][:, 0] > 1) & (keypoints.xy[0][:, 1] > 1)
    if not mask.any():
        return draw_pitch(config=CONFIG)
    
    transformer = ViewTransformer(
        source=keypoints.xy[0][mask].astype(np.float32),
        target=np.array(CONFIG.vertices)[mask].astype(np.float32)
    )
    
    radar = draw_pitch(config=CONFIG)
    
    # Draw player positions with jersey numbers
    if len(player_detections) > 0:
        player_xy = player_detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
        transformed_player_xy = transformer.transform_points(points=player_xy)
        
        for i, tracker_id in enumerate(player_detections.tracker_id):
            if tracker_id is None:
                continue
            
            team_id = manager.get_team_id(tracker_id)
            jersey_number = manager.get_jersey_number(tracker_id)
            
            if team_id is not None and jersey_number is not None:
                point = transformed_player_xy[i]
                color = sv.Color.from_hex(COLORS[team_id])
                
                # Draw player position
                radar = draw_points_on_pitch(
                    config=CONFIG, xy=point.reshape(1, -1),
                    face_color=color, radius=20, pitch=radar)
                
                # Draw jersey number
                cv2.putText(radar, str(jersey_number), 
                           (int(point[0]) - 10, int(point[1]) + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Draw ball position
    if len(ball_detections) > 0:
        ball_xy = ball_detections.get_anchors_coordinates(anchor=sv.Position.CENTER)
        transformed_ball_xy = transformer.transform_points(points=ball_xy)
        
        for i in range(len(ball_detections)):
            point = transformed_ball_xy[i]
            radar = draw_points_on_pitch(
                config=CONFIG, xy=point.reshape(1, -1),
                face_color=sv.Color.from_hex(COLORS[BALL_RADAR_CLASS_ID]), 
                radius=10, pitch=radar)
    
    return radar

def simple_tactical_test(source_video_path: str, target_video_path: str, device: str = "cpu"):
    """Run simple tactical test with jersey assignment"""
    
    # Initialize jersey assignment manager
    jersey_manager = JerseyAssignmentManager()
    
    # Load models
    PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PLAYER_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/football-player-detection.pt')
    PITCH_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/football-pitch-detection.pt')
    BALL_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/football-ball-detection.pt')
    
    player_model = YOLO(PLAYER_DETECTION_MODEL_PATH)
    pitch_model = YOLO(PITCH_DETECTION_MODEL_PATH)
    ball_model = YOLO(BALL_DETECTION_MODEL_PATH)
    
    # Initialize trackers
    player_tracker = sv.ByteTrack()
    ball_tracker = BallTracker()
    
    # Video setup
    cap = cv2.VideoCapture(source_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(target_video_path, fourcc, fps, (width, height))
    
    frame_idx = 0
    paused = False
    
    print("Simple Tactical Test Mode")
    print("Controls:")
    print("  SPACE - Pause/Resume")
    print("  A - Toggle jersey assignment mode")
    print("  1-4 - Select team")
    print("  Click player to select")
    print("  0-9 - Enter jersey number")
    print("  ENTER - Assign jersey")
    print("  ESC - Cancel assignment")
    print("  S - Save assignments")
    print("  L - Load assignments")
    print("  Q - Quit")
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_idx += 1
            if frame_idx > MAX_FRAMES:
                break
            
            print(f"Processing frame {frame_idx}/{MAX_FRAMES}")
            
            # Run detection and tracking
            player_result = player_model(frame, verbose=False)[0]
            player_detections = sv.Detections.from_ultralytics(player_result)
            player_detections = player_tracker.update_with_detections(player_detections)
            
            ball_result = ball_model(frame, verbose=False)[0]
            ball_detections = sv.Detections.from_ultralytics(ball_result)
            ball_detections = ball_tracker.update(ball_detections)
            
            pitch_result = pitch_model(frame, verbose=False)[0]
            keypoints = sv.KeyPoints.from_ultralytics(pitch_result)
            
            # Filter detections
            players = player_detections[player_detections.class_id == PLAYER_CLASS_ID]
            goalkeepers = player_detections[player_detections.class_id == GOALKEEPER_CLASS_ID]
            referees = player_detections[player_detections.class_id == REFEREE_CLASS_ID]
            
            # Merge detections
            all_player_detections = sv.Detections.merge([players, goalkeepers, referees])
            
            # Draw interface
            annotated_frame = jersey_manager.draw_assignment_interface(
                frame, all_player_detections, frame_idx)
            
            # Add radar
            try:
                radar = render_radar_with_jerseys(
                    all_player_detections, ball_detections, keypoints, jersey_manager)
                radar = sv.resize_image(radar, (width // 3, height // 3))
                radar_h, radar_w, _ = radar.shape
                rect = sv.Rect(x=width - radar_w - 20, y=20, width=radar_w, height=radar_h)
                annotated_frame = sv.draw_image(annotated_frame, radar, opacity=0.8, rect=rect)
            except Exception as e:
                print(f"Radar rendering failed: {e}")
            
            # Write frame
            out.write(annotated_frame)
        
        # Display frame
        cv2.imshow('Simple Tactical Test', annotated_frame if not paused else frame)
        
        # Handle input
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
        elif jersey_manager.handle_key_press(key, all_player_detections if not paused else sv.Detections.empty()):
            pass  # Jersey assignment handled
        
        # Handle mouse clicks
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                jersey_manager.handle_mouse_click(x, y, all_player_detections if not paused else sv.Detections.empty())
        
        cv2.setMouseCallback('Simple Tactical Test', mouse_callback)
    
    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    # Save assignments
    jersey_manager.save_assignments("simple_test_jersey_assignments.json")
    print(f"✓ Simple tactical test complete! Output saved to: {target_video_path}")
    print(f"✓ Jersey assignments saved to: simple_test_jersey_assignments.json")

def main():
    parser = argparse.ArgumentParser(description="Simple Soccer Tactical Test")
    parser.add_argument("--source_video_path", type=str, required=True, help="Path to source video")
    parser.add_argument("--target_video_path", type=str, required=True, help="Path to output video")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use (cpu/cuda)")
    
    args = parser.parse_args()
    simple_tactical_test(args.source_video_path, args.target_video_path, args.device)

if __name__ == "__main__":
    main()
