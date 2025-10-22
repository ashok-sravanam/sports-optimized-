"""
Enhanced Soccer Tactical Analysis with Jersey Number Assignment and Database Integration
Combines player tracking, jersey assignment, homography transformation, and database storage.
"""

import argparse
import os
import warnings
import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO
from collections import deque
import json
from datetime import datetime
from typing import List, Dict, Optional, Tuple

from sports.annotators.soccer import draw_pitch, draw_points_on_pitch
from sports.common.ball import BallTracker, BallAnnotator
from sports.common.team import TeamClassifier
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
MAX_FRAMES = 50  # Test with 50 frames
BALL_CLASS_ID = 0
PLAYER_CLASS_ID = 1
GOALKEEPER_CLASS_ID = 2
REFEREE_CLASS_ID = 3
BALL_RADAR_CLASS_ID = 4

# Colors for different teams and ball
COLORS = ['#FF1493', '#00BFFF', '#FF6347', '#FFD700', '#32CD32', '#FF69B4', '#FF0000']  # Red for ball

# Soccer pitch configuration
CONFIG = SoccerPitchConfiguration()

# Annotators
BOX_ANNOTATOR = sv.BoxAnnotator()
LABEL_ANNOTATOR = sv.LabelAnnotator()
ELLIPSE_ANNOTATOR = sv.EllipseAnnotator()
ELLIPSE_LABEL_ANNOTATOR = sv.LabelAnnotator()
VERTEX_LABEL_ANNOTATOR = sv.LabelAnnotator()

def get_crops(frame: np.ndarray, detections: sv.Detections) -> List[np.ndarray]:
    """Extract crops from detections for team classification"""
    crops = []
    for i in range(len(detections)):
        x1, y1, x2, y2 = detections.xyxy[i].astype(int)
        crop = frame[y1:y2, x1:x2]
        if crop.size > 0:
            crops.append(crop)
        else:
            crops.append(np.zeros((100, 100, 3), dtype=np.uint8))
    return crops

def resolve_goalkeepers_team_id(goalkeepers_team_id: np.ndarray, players_team_id: np.ndarray) -> np.ndarray:
    """Resolve goalkeeper team IDs based on player team IDs"""
    if len(goalkeepers_team_id) == 0:
        return goalkeepers_team_id
    
    unique_teams = np.unique(players_team_id)
    if len(unique_teams) == 0:
        return goalkeepers_team_id
    
    resolved_goalkeepers_team_id = []
    for goalkeeper_team_id in goalkeepers_team_id:
        if goalkeeper_team_id in unique_teams:
            resolved_goalkeepers_team_id.append(goalkeeper_team_id)
        else:
            resolved_goalkeepers_team_id.append(unique_teams[0])
    
    return np.array(resolved_goalkeepers_team_id)

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

def train_team_classifier(team_classifier: TeamClassifier, video_path: str, max_frames: int = 50):
    """Train team classifier on video data like in clean_analysis.py"""
    print("Collecting player crops for team classification...")
    
    # Load models
    PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PLAYER_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/football-player-detection.pt')
    player_model = YOLO(PLAYER_DETECTION_MODEL_PATH)
    
    # Initialize tracker
    player_tracker = sv.ByteTrack()
    
    # Collect crops from video
    video_info = sv.VideoInfo.from_video_path(video_path)
    frames_generator = sv.get_video_frames_generator(video_path)
    
    crops = []
    frame_count = 0
    
    for frame in frames_generator:
        if frame_count >= max_frames:
            break
            
        # Run detection
        player_result = player_model(frame, verbose=False)[0]
        player_detections = sv.Detections.from_ultralytics(player_result)
        player_detections = player_tracker.update_with_detections(player_detections)
        
        # Filter for players only
        players = player_detections[player_detections.class_id == PLAYER_CLASS_ID]
        
        if len(players) > 0:
            player_crops = get_crops(frame, players)
            crops.extend(player_crops)
        
        frame_count += 1
    
    print(f"Collected {len(crops)} player crops from {frame_count} frames")
    
    if len(crops) > 0:
        print("Training team classifier...")
        team_classifier.fit(crops)
        print("✓ Team classifier trained successfully")
    else:
        print("⚠ No player crops found for training")

def tactical_analysis(source_video_path: str, target_video_path: str, device: str = "cpu", 
                    match_id: int = 1, team_a_id: int = 1, team_b_id: int = 2):
    """Run tactical analysis with jersey assignment and database integration"""
    
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
    
    # Initialize trackers and classifiers
    player_tracker = sv.ByteTrack()
    ball_tracker = BallTracker()
    team_classifier = TeamClassifier()
    
    # Train team classifier first
    train_team_classifier(team_classifier, source_video_path, MAX_FRAMES)
    
    # Video setup
    video_info = sv.VideoInfo.from_video_path(source_video_path)
    frames_generator = sv.get_video_frames_generator(source_video_path)
    
    # Determine frame limit
    total_frames = video_info.total_frames
    if MAX_FRAMES:
        total_frames = min(MAX_FRAMES, total_frames)
        print(f"Processing first {MAX_FRAMES} frames...")
    else:
        print("Processing entire video...")
    
    # Video writer
    with sv.VideoSink(target_video_path, video_info) as sink:
        progress_bar = tqdm(total=total_frames, desc="Processing frames")
        
        for frame_idx, frame in enumerate(frames_generator):
            if MAX_FRAMES and frame_idx >= MAX_FRAMES:
                break
            
            # 1. PLAYER DETECTION
            player_result = player_model(frame, verbose=False)[0]
            player_detections = sv.Detections.from_ultralytics(player_result)
            player_detections = player_tracker.update_with_detections(player_detections)
            
            # Filter for players, goalkeepers, referees
            players = player_detections[player_detections.class_id == PLAYER_CLASS_ID]
            goalkeepers = player_detections[player_detections.class_id == GOALKEEPER_CLASS_ID]
            referees = player_detections[player_detections.class_id == REFEREE_CLASS_ID]
            
            # 2. BALL DETECTION
            ball_result = ball_model(frame, verbose=False)[0]
            ball_detections = sv.Detections.from_ultralytics(ball_result)
            ball_detections = ball_tracker.update(ball_detections)
            
            # 3. PITCH DETECTION
            pitch_result = pitch_model(frame, verbose=False)[0]
            keypoints = sv.KeyPoints.from_ultralytics(pitch_result)
            
            # 4. TEAM CLASSIFICATION (using working approach from clean_analysis.py)
            if len(players) > 0:
                player_crops = get_crops(frame, players)
                players_team_id = team_classifier.predict(player_crops)
            else:
                players_team_id = np.array([])
            
            if len(goalkeepers) > 0:
                goalkeeper_crops = get_crops(frame, goalkeepers)
                goalkeepers_team_id = team_classifier.predict(goalkeeper_crops)
                goalkeepers_team_id = resolve_goalkeepers_team_id(goalkeepers_team_id, players_team_id)
            else:
                goalkeepers_team_id = np.array([])
            
            # 5. JERSEY ASSIGNMENT INTERFACE
            annotated_frame = frame.copy()
            
            # Merge player detections for annotation
            all_player_detections = sv.Detections.merge([players, goalkeepers, referees])
            
            # Draw jersey assignment interface
            annotated_frame = jersey_manager.draw_assignment_interface(
                annotated_frame, all_player_detections, frame_idx)
            
            # 6. RADAR VISUALIZATION WITH JERSEY NUMBERS
            h, w, _ = frame.shape
            try:
                radar = render_radar_with_jerseys(
                    all_player_detections, ball_detections, keypoints, jersey_manager)
                radar = sv.resize_image(radar, (w // 3, h // 3))
                radar_h, radar_w, _ = radar.shape
                
                # Position radar in top-right corner
                rect = sv.Rect(
                    x=w - radar_w - 20,
                    y=20,
                    width=radar_w,
                    height=radar_h
                )
                annotated_frame = sv.draw_image(annotated_frame, radar, opacity=0.8, rect=rect)
            except Exception as e:
                # If radar fails, just draw a simple pitch
                radar = draw_pitch(config=CONFIG)
                radar = sv.resize_image(radar, (w // 3, h // 3))
                radar_h, radar_w, _ = radar.shape
                rect = sv.Rect(
                    x=w - radar_w - 20,
                    y=20,
                    width=radar_w,
                    height=radar_h
                )
                annotated_frame = sv.draw_image(annotated_frame, radar, opacity=0.8, rect=rect)
            
            # 7. ADD ANALYSIS LABELS
            cv2.putText(annotated_frame, "Tactical Analysis with Jersey Assignment", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(annotated_frame, "Press 'A' to assign jersey numbers", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Frame counter
            if MAX_FRAMES:
                cv2.putText(annotated_frame, f"Frame: {frame_idx + 1}/{MAX_FRAMES}", (10, h - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                cv2.putText(annotated_frame, f"Frame: {frame_idx + 1}", (10, h - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Write frame
            sink.write_frame(annotated_frame)
            progress_bar.update(1)
        
        progress_bar.close()
    
    # Save final assignments
    jersey_manager.save_assignments("final_jersey_assignments.json")
    
    print(f"✓ Analysis complete! Output saved to: {target_video_path}")
    print(f"✓ Jersey assignments saved to: final_jersey_assignments.json")

def interactive_analysis(source_video_path: str, target_video_path: str, device: str = "cpu"):
    """Run interactive analysis with real-time jersey assignment"""
    
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
    
    # Initialize trackers and classifiers
    player_tracker = sv.ByteTrack()
    ball_tracker = BallTracker()
    team_classifier = TeamClassifier()
    
    # Train team classifier first
    train_team_classifier(team_classifier, source_video_path, MAX_FRAMES)
    
    # Video setup
    cap = cv2.VideoCapture(source_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(target_video_path, fourcc, fps, 
                         (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                          int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    
    frame_idx = 0
    paused = False
    
    print("Interactive Analysis Mode")
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
            if MAX_FRAMES and frame_idx > MAX_FRAMES:
                break
            
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
            
            # Team classification
            if len(players) > 0:
                player_crops = get_crops(frame, players)
                players_team_id = team_classifier.predict(player_crops)
            else:
                players_team_id = np.array([])
            
            if len(goalkeepers) > 0:
                goalkeeper_crops = get_crops(frame, goalkeepers)
                goalkeepers_team_id = team_classifier.predict(goalkeeper_crops)
                goalkeepers_team_id = resolve_goalkeepers_team_id(goalkeepers_team_id, players_team_id)
            else:
                goalkeepers_team_id = np.array([])
            
            # Merge detections
            all_player_detections = sv.Detections.merge([players, goalkeepers, referees])
            
            # Draw interface
            annotated_frame = jersey_manager.draw_assignment_interface(
                frame, all_player_detections, frame_idx)
            
            # Add radar
            try:
                radar = render_radar_with_jerseys(
                    all_player_detections, ball_detections, keypoints, jersey_manager)
                h, w = frame.shape[:2]
                radar = sv.resize_image(radar, (w // 3, h // 3))
                radar_h, radar_w, _ = radar.shape
                rect = sv.Rect(x=w - radar_w - 20, y=20, width=radar_w, height=radar_h)
                annotated_frame = sv.draw_image(annotated_frame, radar, opacity=0.8, rect=rect)
            except:
                pass
            
            # Write frame
            out.write(annotated_frame)
        
        # Display frame
        cv2.imshow('Tactical Analysis', annotated_frame if not paused else frame)
        
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
        
        cv2.setMouseCallback('Tactical Analysis', mouse_callback)
    
    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    # Save assignments
    jersey_manager.save_assignments("interactive_jersey_assignments.json")
    print(f"✓ Interactive analysis complete! Output saved to: {target_video_path}")
    print(f"✓ Jersey assignments saved to: interactive_jersey_assignments.json")

def main():
    parser = argparse.ArgumentParser(description="Soccer Tactical Analysis with Jersey Assignment")
    parser.add_argument("--source_video_path", type=str, required=True, help="Path to source video")
    parser.add_argument("--target_video_path", type=str, required=True, help="Path to output video")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use (cpu/cuda)")
    parser.add_argument("--mode", type=str, default="batch", choices=["batch", "interactive"], 
                       help="Analysis mode: batch or interactive")
    parser.add_argument("--match_id", type=int, default=1, help="Match ID for database")
    parser.add_argument("--team_a_id", type=int, default=1, help="Team A ID")
    parser.add_argument("--team_b_id", type=int, default=2, help="Team B ID")
    
    args = parser.parse_args()
    
    if args.mode == "interactive":
        interactive_analysis(args.source_video_path, args.target_video_path, args.device)
    else:
        tactical_analysis(args.source_video_path, args.target_video_path, args.device,
                         args.match_id, args.team_a_id, args.team_b_id)

if __name__ == "__main__":
    main()