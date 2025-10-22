#!/usr/bin/env python3
"""
Clean Soccer Analysis Script
Performs all 6 analysis tasks simultaneously with pitch detection hidden:
1. Ball tracking - Detects and tracks soccer balls
2. Player detection - Identifies players, goalkeepers, and referees  
3. Pitch detection - Detects soccer field boundaries (HIDDEN)
4. Player tracking - Maintains consistent player identification
5. Team classification - Classifies players into teams
6. Radar visualization - Creates overhead view with ball included
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

# Suppress repetitive warnings
warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")
warnings.filterwarnings("ignore", category=UserWarning, module="supervision")
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Suppress urllib3 SSL warnings
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from sports.annotators.soccer import draw_pitch, draw_points_on_pitch
from sports.common.ball import BallTracker, BallAnnotator
from sports.common.team import TeamClassifier
from sports.common.view import ViewTransformer
from sports.configs.soccer import SoccerPitchConfiguration

# Configuration
PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
PLAYER_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/football-player-detection.pt')
PITCH_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/football-pitch-detection.pt')
BALL_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/football-ball-detection.pt')

BALL_CLASS_ID = 0
GOALKEEPER_CLASS_ID = 1
PLAYER_CLASS_ID = 2
REFEREE_CLASS_ID = 3
BALL_RADAR_CLASS_ID = 4  # For radar visualization

STRIDE = 60
CONFIG = SoccerPitchConfiguration()
MAX_FRAMES = 2400  # Process 2400 frames

# Colors for different entities
COLORS = ['#FF1493', '#00BFFF', '#FF6347', '#FFD700', '#32CD32', '#FF69B4', '#FF0000']  # Red for ball

# Annotators (removed pitch detection annotators)
BOX_ANNOTATOR = sv.BoxAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    thickness=2
)

ELLIPSE_ANNOTATOR = sv.EllipseAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    thickness=2
)

BOX_LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    text_color=sv.Color.from_hex('#FFFFFF'),
    text_padding=5,
    text_thickness=1,
)

ELLIPSE_LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    text_color=sv.Color.from_hex('#FFFFFF'),
    text_padding=5,
    text_thickness=1,
    text_position=sv.Position.BOTTOM_CENTER,
)


def get_crops(frame: np.ndarray, detections: sv.Detections) -> list:
    """Extract crops from the frame based on detected bounding boxes."""
    return [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]


def resolve_goalkeepers_team_id(players: sv.Detections, players_team_id: np.array, goalkeepers: sv.Detections) -> np.ndarray:
    """Resolve team IDs for goalkeepers based on proximity to team centroids."""
    goalkeepers_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    team_0_centroid = players_xy[players_team_id == 0].mean(axis=0)
    team_1_centroid = players_xy[players_team_id == 1].mean(axis=0)
    goalkeepers_team_id = []
    for goalkeeper_xy in goalkeepers_xy:
        dist_0 = np.linalg.norm(goalkeeper_xy - team_0_centroid)
        dist_1 = np.linalg.norm(goalkeeper_xy - team_1_centroid)
        goalkeepers_team_id.append(0 if dist_0 < dist_1 else 1)
    return np.array(goalkeepers_team_id)


def render_radar_with_ball(player_detections: sv.Detections, ball_detections: sv.Detections, keypoints: sv.KeyPoints, color_lookup: np.ndarray) -> np.ndarray:
    """Render radar view with player positions and ball."""
    radar = draw_pitch(config=CONFIG)
    
    # Check if we have valid keypoints
    mask = (keypoints.xy[0][:, 0] > 1) & (keypoints.xy[0][:, 1] > 1)
    if not mask.any():
        return radar
        
    transformer = ViewTransformer(
        source=keypoints.xy[0][mask].astype(np.float32),
        target=np.array(CONFIG.vertices)[mask].astype(np.float32)
    )
    
    # Draw player points
    if len(player_detections) > 0:
        player_xy = player_detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
        transformed_player_xy = transformer.transform_points(points=player_xy)
        
        # Draw points for each team/class
        for team_id in range(4):  # 0, 1, 2, 3 (teams)
            team_mask = color_lookup == team_id
            if team_mask.any():
                team_points = transformed_player_xy[team_mask]
                radar = draw_points_on_pitch(
                    config=CONFIG, xy=team_points,
                    face_color=sv.Color.from_hex(COLORS[team_id]), radius=20, pitch=radar)
    
    # Draw ball points
    if ball_detections is not None and len(ball_detections) > 0:
        ball_xy = ball_detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
        transformed_ball_xy = transformer.transform_points(points=ball_xy)
        
        # Draw ball as small red dots
        radar = draw_points_on_pitch(
            config=CONFIG, xy=transformed_ball_xy,
            face_color=sv.Color.from_hex(COLORS[BALL_RADAR_CLASS_ID]), radius=10, pitch=radar)
    
    return radar


def clean_analysis(source_video_path: str, target_video_path: str, device: str) -> None:
    """
    Perform clean soccer analysis combining all 6 modes with pitch detection hidden.
    """
    print("Loading models...")
    
    # Load all models
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    pitch_detection_model = YOLO(PITCH_DETECTION_MODEL_PATH).to(device=device)
    ball_detection_model = YOLO(BALL_DETECTION_MODEL_PATH).to(device=device)
    
    # Initialize trackers and classifiers
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    ball_tracker = BallTracker(buffer_size=20)
    ball_annotator = BallAnnotator(radius=6, buffer_size=10)
    
    # Ball detection callback for slicing
    def ball_callback(image_slice: np.ndarray) -> sv.Detections:
        result = ball_detection_model(image_slice, imgsz=640, verbose=False)[0]
        return sv.Detections.from_ultralytics(result)
    
    slicer = sv.InferenceSlicer(
        callback=ball_callback,
        slice_wh=(640, 640),
        overlap_wh=(0, 0),
        overlap_ratio_wh=None,
    )
    
    print("Collecting player crops for team classification...")
    
    # First pass: collect crops for team classification
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path, stride=STRIDE)
    crops = []
    frame_count = 0
    
    for frame in frame_generator:
        if MAX_FRAMES and frame_count >= MAX_FRAMES:
            break
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        crops += get_crops(frame, detections[detections.class_id == PLAYER_CLASS_ID])
        frame_count += STRIDE
    
    # Train team classifier
    print("Training team classifier...")
    team_classifier = TeamClassifier(device=device)
    team_classifier.fit(crops)
    
    print("Starting clean analysis...")
    
    # Second pass: clean analysis
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    video_info = sv.VideoInfo.from_video_path(source_video_path)
    
    # Limit to MAX_FRAMES if specified
    total_frames = video_info.total_frames if not MAX_FRAMES else min(video_info.total_frames, MAX_FRAMES)
    
    with sv.VideoSink(target_video_path, video_info) as sink:
        frame_idx = 0
        
        for frame in tqdm(frame_generator, total=total_frames, desc="Processing frames"):
            if MAX_FRAMES and frame_idx >= MAX_FRAMES:
                break
                
            # 1. PITCH DETECTION (hidden - analysis only)
            pitch_result = pitch_detection_model(frame, verbose=False)[0]
            keypoints = sv.KeyPoints.from_ultralytics(pitch_result)
            
            # 2. PLAYER DETECTION
            player_result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(player_result)
            
            # 3. PLAYER TRACKING
            detections = tracker.update_with_detections(detections)
            
            # 4. BALL DETECTION
            ball_detections = slicer(frame).with_nms(threshold=0.1)
            ball_detections = ball_tracker.update(ball_detections)
            
            # 5. TEAM CLASSIFICATION
            players = detections[detections.class_id == PLAYER_CLASS_ID]
            crops = get_crops(frame, players)
            players_team_id = team_classifier.predict(crops)
            
            goalkeepers = detections[detections.class_id == GOALKEEPER_CLASS_ID]
            goalkeepers_team_id = resolve_goalkeepers_team_id(players, players_team_id, goalkeepers)
            
            referees = detections[detections.class_id == REFEREE_CLASS_ID]
            
            # Merge player detections (players, goalkeepers, referees)
            all_detections = sv.Detections.merge([players, goalkeepers, referees])
            
            # Create color lookup for player detections
            color_lookup = np.array(
                players_team_id.tolist() +
                goalkeepers_team_id.tolist() +
                [REFEREE_CLASS_ID] * len(referees)
            )
            
            # Handle ball detections separately for radar
            ball_detections_for_radar = None
            if len(ball_detections) > 0:
                # Create tracker IDs for ball detections (use negative IDs to distinguish from players)
                ball_tracker_ids = np.array([-i-1 for i in range(len(ball_detections))])
                ball_detections.tracker_id = ball_tracker_ids
                
                # Create ball detections for radar with proper structure
                ball_detections_for_radar = sv.Detections(
                    xyxy=ball_detections.xyxy,
                    class_id=np.array([BALL_CLASS_ID] * len(ball_detections)),
                    confidence=ball_detections.confidence if hasattr(ball_detections, 'confidence') and ball_detections.confidence is not None else np.ones(len(ball_detections)),
                    tracker_id=ball_tracker_ids
                )
            
            # Create clean annotated frame
            annotated_frame = frame.copy()
            
            # Skip pitch detection annotations (hidden - analysis only)
            # Add pitch detection annotations (hidden - analysis only)
            # annotated_frame = VERTEX_LABEL_ANNOTATOR.annotate(annotated_frame, keypoints, CONFIG.labels)
            # annotated_frame = EDGE_ANNOTATOR.annotate(annotated_frame, keypoints)
            
            # Add player/team annotations
            labels = [f"ID:{tracker_id}" for tracker_id in all_detections.tracker_id]
            annotated_frame = ELLIPSE_ANNOTATOR.annotate(
                annotated_frame, all_detections)
            annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
                annotated_frame, all_detections, labels)
            
            # Add ball tracking annotations
            annotated_frame = ball_annotator.annotate(annotated_frame, ball_detections)
            
            # 6. RADAR VISUALIZATION
            h, w, _ = frame.shape
            try:
                radar = render_radar_with_ball(all_detections, ball_detections_for_radar, keypoints, color_lookup)
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
                rect = sv.Rect(x=w - radar_w - 20, y=20, width=radar_w, height=radar_h)
                annotated_frame = sv.draw_image(annotated_frame, radar, opacity=0.8, rect=rect)
            
            # Add analysis mode labels
            cv2.putText(annotated_frame, "CLEAN SOCCER ANALYSIS", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(annotated_frame, "Ball Tracking | Player Detection | Player Tracking", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(annotated_frame, "Team Classification | Radar View (Ball + Players)", (10, 85), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Add frame counter
            if MAX_FRAMES:
                cv2.putText(annotated_frame, f"Frame: {frame_idx + 1}/{MAX_FRAMES}", (10, h - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                cv2.putText(annotated_frame, f"Frame: {frame_idx + 1}", (10, h - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Write frame to output video
            sink.write_frame(annotated_frame)
            
            # Display frame (optional)
            cv2.imshow("Clean Soccer Analysis", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
                
            frame_idx += 1
    
    cv2.destroyAllWindows()
    print(f"Analysis complete! Output saved to: {target_video_path}")


def main():
    parser = argparse.ArgumentParser(description='Clean Soccer Analysis - All 6 modes with pitch detection hidden')
    parser.add_argument('--source_video_path', type=str, required=True,
                       help='Path to source video file')
    parser.add_argument('--target_video_path', type=str, required=True,
                       help='Path to output video file')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use (cpu, cuda, mps)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("CLEAN SOCCER ANALYSIS")
    print("=" * 60)
    print("Performing all 6 analysis modes simultaneously:")
    print("1. Ball tracking - Detects and tracks soccer balls")
    print("2. Player detection - Identifies players, goalkeepers, referees")
    print("3. Pitch detection - Detects soccer field boundaries (HIDDEN)")
    print("4. Player tracking - Maintains consistent player identification")
    print("5. Team classification - Classifies players into teams")
    print("6. Radar visualization - Creates overhead view")
    print("=" * 60)
    if MAX_FRAMES:
        print(f"Processing first {MAX_FRAMES} frames...")
    else:
        print("Processing entire video...")
    print(f"Source: {args.source_video_path}")
    print(f"Output: {args.target_video_path}")
    print(f"Device: {args.device}")
    print("=" * 60)
    
    clean_analysis(
        source_video_path=args.source_video_path,
        target_video_path=args.target_video_path,
        device=args.device
    )


if __name__ == '__main__':
    main()
