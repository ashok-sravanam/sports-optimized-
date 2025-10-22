#!/usr/bin/env python3
"""
Split-Screen Soccer Analysis System
Left: Video with player tracking, ball tracking, team classification
Right: Clean tactical board with player positions and jersey numbers
"""

import argparse
import os
import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO
from typing import List, Dict, Optional
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")
warnings.filterwarnings("ignore", category=UserWarning, module="supervision")
warnings.filterwarnings("ignore", category=RuntimeWarning)

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Your imports from sports package
from sports.annotators.soccer import draw_pitch
from sports.common.ball import BallTracker, BallAnnotator
from sports.common.team import TeamClassifier
from sports.common.view import ViewTransformer
from sports.configs.soccer import SoccerPitchConfiguration

# Configuration
PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
PLAYER_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/football-player-detection.pt')
PITCH_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/football-pitch-detection.pt')
BALL_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/football-ball-detection.pt')

# Class IDs
BALL_CLASS_ID = 0
GOALKEEPER_CLASS_ID = 1
PLAYER_CLASS_ID = 2
REFEREE_CLASS_ID = 3

# Team colors (BGR format)
TEAM_1_COLOR = (255, 71, 71)   # Blue
TEAM_2_COLOR = (71, 71, 255)   # Red
BALL_COLOR = (255, 255, 255)    # White

CONFIG = SoccerPitchConfiguration()

class TacticalBoardManager:
    """Manages jersey assignments and tactical board state"""
    def __init__(self):
        self.jersey_assignments = {}  # tracker_id -> jersey_number
    
    def get_jersey_number(self, tracker_id: int) -> Optional[int]:
        return self.jersey_assignments.get(tracker_id)

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

def draw_tactical_board(
    frame_shape: tuple,
    detections: sv.Detections,
    team_ids: np.ndarray,
    ball_detections: sv.Detections,
    keypoints: sv.KeyPoints,
    manager: TacticalBoardManager
) -> np.ndarray:
    """
    Draw clean tactical board with player circles and numbers
    
    Returns:
        np.ndarray: Tactical board image (board_h x board_w x 3)
    """
    h, w = frame_shape[:2]
    
    # Tactical board dimensions (adjust as needed)
    board_w = 1200
    board_h = 800
    
    # Draw base pitch
    pitch = draw_pitch(config=CONFIG)
    tactical_board = cv2.resize(pitch, (board_w, board_h), 
                                interpolation=cv2.INTER_LANCZOS4)
    
    # Check for valid keypoints
    if len(keypoints.xy[0]) == 0:
        return tactical_board  # Return empty pitch
    
    mask = (keypoints.xy[0][:, 0] > 1) & (keypoints.xy[0][:, 1] > 1)
    
    if not mask.any() or mask.sum() < 4:
        return tactical_board  # Return empty pitch
    
    try:
        # Create coordinate transformer
        transformer = ViewTransformer(
            source=keypoints.xy[0][mask].astype(np.float32),
            target=np.array(CONFIG.vertices)[mask].astype(np.float32)
        )
        
        # Transform player positions
        if len(detections) > 0:
            player_xy = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
            transformed_xy = transformer.transform_points(points=player_xy)
            
            # Scale to board dimensions
            scale_x = board_w / CONFIG.length
            scale_y = board_h / CONFIG.width
            
            board_positions = transformed_xy.copy()
            board_positions[:, 0] *= scale_x
            board_positions[:, 1] *= scale_y
            
            # Draw players as circles with numbers
            for i in range(len(detections)):
                x, y = board_positions[i]
                tracker_id = detections.tracker_id[i] if detections.tracker_id is not None else i
                team_id = team_ids[i] if i < len(team_ids) else 0
                
                # Ensure position is within bounds
                x = max(15, min(board_w - 15, x))
                y = max(15, min(board_h - 15, y))
                
                # Choose color based on team
                if team_id == 0:
                    color = TEAM_1_COLOR
                elif team_id == 1:
                    color = TEAM_2_COLOR
                else:
                    color = (200, 200, 200)  # Gray for referees
                
                # Get jersey number or use tracker_id
                jersey_num = manager.get_jersey_number(tracker_id)
                display_num = jersey_num if jersey_num is not None else tracker_id
                
                # Draw circle
                radius = 15
                cv2.circle(tactical_board, (int(x), int(y)), radius, color, -1)
                cv2.circle(tactical_board, (int(x), int(y)), radius, (255, 255, 255), 2)
                
                # Draw number inside circle
                text = str(display_num)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.4
                thickness = 1
                
                text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                text_x = int(x - text_size[0] / 2)
                text_y = int(y + text_size[1] / 2)
                
                cv2.putText(tactical_board, text, (text_x, text_y),
                           font, font_scale, (255, 255, 255), thickness)
        
        # Draw ball
        if ball_detections is not None and len(ball_detections) > 0:
            ball_xy = ball_detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
            transformed_ball = transformer.transform_points(points=ball_xy)
            
            # Scale to board dimensions
            scale_x = board_w / CONFIG.length
            scale_y = board_h / CONFIG.width
            
            ball_board = transformed_ball.copy()
            ball_board[:, 0] *= scale_x
            ball_board[:, 1] *= scale_y
            
            for bx, by in ball_board:
                bx = max(8, min(board_w - 8, bx))
                by = max(8, min(board_h - 8, by))
                cv2.circle(tactical_board, (int(bx), int(by)), 8, BALL_COLOR, -1)
                cv2.circle(tactical_board, (int(bx), int(by)), 8, (0, 0, 0), 2)
    
    except Exception as e:
        print(f"Tactical board error: {e}")
    
    return tactical_board

def train_team_classifier(team_classifier: TeamClassifier, video_path: str, max_frames: int = 50):
    """Train team classifier on video data"""
    print("Collecting player crops for team classification...")
    
    player_model = YOLO(PLAYER_DETECTION_MODEL_PATH)
    player_tracker = sv.ByteTrack()
    
    video_info = sv.VideoInfo.from_video_path(video_path)
    frames_generator = sv.get_video_frames_generator(video_path)
    
    crops = []
    frame_count = 0
    
    for frame in frames_generator:
        if frame_count >= max_frames:
            break
            
        player_result = player_model(frame, verbose=False)[0]
        player_detections = sv.Detections.from_ultralytics(player_result)
        player_detections = player_tracker.update_with_detections(player_detections)
        
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

def process_video(source_path: str, target_path: str, device: str = 'cpu', max_frames: int = 50):
    """Main processing loop"""
    
    # Load models
    print("Loading models...")
    player_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    pitch_model = YOLO(PITCH_DETECTION_MODEL_PATH).to(device=device)
    ball_model = YOLO(BALL_DETECTION_MODEL_PATH).to(device=device)
    
    # Initialize trackers
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    ball_tracker = BallTracker(buffer_size=20)
    ball_annotator = BallAnnotator(radius=6, buffer_size=10)
    
    # Ball detection slicer
    def ball_callback(image_slice: np.ndarray) -> sv.Detections:
        result = ball_model(image_slice, imgsz=640, verbose=False)[0]
        return sv.Detections.from_ultralytics(result)
    
    slicer = sv.InferenceSlicer(callback=ball_callback, slice_wh=(640, 640), 
                               overlap_wh=(0, 0), overlap_ratio_wh=None)
    
    # Team classifier (train on first pass)
    print("Training team classifier...")
    team_classifier = TeamClassifier(device=device)
    train_team_classifier(team_classifier, source_path, max_frames)
    
    # Initialize manager
    manager = TacticalBoardManager()
    
    # Setup video writer
    video_info = sv.VideoInfo.from_video_path(source_path)
    
    # Calculate output dimensions
    video_w, video_h = video_info.width, video_info.height
    board_w, board_h = 1200, 800
    
    # Resize if needed to match heights
    if video_h != board_h:
        aspect = video_w / video_h
        video_h = board_h
        video_w = int(board_h * aspect)
    
    output_w = video_w + board_w
    output_h = max(video_h, board_h)
    
    output_video_info = sv.VideoInfo(
        width=output_w,
        height=output_h,
        fps=video_info.fps,
        total_frames=min(video_info.total_frames, max_frames) if max_frames else video_info.total_frames
    )
    
    # Annotators
    COLORS = ['#FF1493', '#00BFFF', '#FFD700', '#32CD32']
    ellipse_annotator = sv.EllipseAnnotator(
        color=sv.ColorPalette.from_hex(COLORS), thickness=2
    )
    label_annotator = sv.LabelAnnotator(
        color=sv.ColorPalette.from_hex(COLORS),
        text_color=sv.Color.WHITE,
        text_position=sv.Position.BOTTOM_CENTER
    )
    
    print("Processing video...")
    
    frame_generator = sv.get_video_frames_generator(source_path=source_path)
    
    with sv.VideoSink(target_path, output_video_info) as sink:
        frame_count = 0
        for frame in tqdm(frame_generator, total=output_video_info.total_frames):
            if max_frames and frame_count >= max_frames:
                break
                
            # Resize video frame if needed
            if frame.shape[0] != video_h or frame.shape[1] != video_w:
                frame = cv2.resize(frame, (video_w, video_h))
            
            # 1. Player detection & tracking
            player_result = player_model(frame, imgsz=1280, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(player_result)
            detections = tracker.update_with_detections(detections)
            
            # 2. Ball detection & tracking
            ball_detections = slicer(frame).with_nms(threshold=0.1)
            ball_detections = ball_tracker.update(ball_detections)
            
            # 3. Pitch detection (for transformation)
            pitch_result = pitch_model(frame, verbose=False)[0]
            keypoints = sv.KeyPoints.from_ultralytics(pitch_result)
            
            # 4. Team classification
            players = detections[detections.class_id == PLAYER_CLASS_ID]
            goalkeepers = detections[detections.class_id == GOALKEEPER_CLASS_ID]
            referees = detections[detections.class_id == REFEREE_CLASS_ID]
            
            if len(players) > 0:
                player_crops = get_crops(frame, players)
                players_team_id = team_classifier.predict(player_crops)
            else:
                players_team_id = np.array([])
            
            if len(goalkeepers) > 0:
                gk_crops = get_crops(frame, goalkeepers)
                goalkeepers_team_id = team_classifier.predict(gk_crops)
                goalkeepers_team_id = resolve_goalkeepers_team_id(goalkeepers_team_id, players_team_id)
            else:
                goalkeepers_team_id = np.array([])
            
            # Merge detections
            all_detections = sv.Detections.merge([players, goalkeepers, referees])
            team_ids = np.concatenate([
                players_team_id,
                goalkeepers_team_id,
                np.array([2] * len(referees))  # Referees = team 2
            ]) if len(all_detections) > 0 else np.array([])
            
            # 5. Annotate video frame (LEFT SIDE)
            annotated_video = frame.copy()
            
            if len(all_detections) > 0:
                labels = [f"ID:{tid}" for tid in all_detections.tracker_id]
                annotated_video = ellipse_annotator.annotate(annotated_video, all_detections)
                annotated_video = label_annotator.annotate(annotated_video, all_detections, labels)
            
            annotated_video = ball_annotator.annotate(annotated_video, ball_detections)
            
            # Add title
            cv2.putText(annotated_video, "PLAYER TRACKING", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # 6. Draw tactical board (RIGHT SIDE)
            tactical_board = draw_tactical_board(
                frame.shape, all_detections, team_ids, ball_detections, keypoints, manager
            )
            
            # Ensure tactical board matches video height
            if tactical_board.shape[0] != video_h:
                tactical_board = cv2.resize(tactical_board, (board_w, video_h))
            
            # Add title to tactical board
            cv2.putText(tactical_board, "TACTICAL BOARD", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # 7. Combine side-by-side
            output_frame = np.hstack([annotated_video, tactical_board])
            
            # Write output
            sink.write_frame(output_frame)
            frame_count += 1
    
    print(f"✓ Processing complete! Output saved to: {target_path}")

def main():
    parser = argparse.ArgumentParser(description="Split-Screen Soccer Analysis")
    parser.add_argument("--source_video_path", type=str, required=True)
    parser.add_argument("--target_video_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max_frames", type=int, default=50, help="Maximum frames to process")
    
    args = parser.parse_args()
    process_video(args.source_video_path, args.target_video_path, args.device, args.max_frames)

if __name__ == "__main__":
    main()
