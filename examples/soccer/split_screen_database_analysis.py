#!/usr/bin/env python3
"""
Split-Screen Soccer Analysis with Database Integration
Uses existing database schema with teams, matches, players, formations
"""

import argparse
import os
import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO
from typing import Dict, List
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")
warnings.filterwarnings("ignore", category=UserWarning, module="supervision")
warnings.filterwarnings("ignore", category=RuntimeWarning)

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from sports.annotators.soccer import draw_pitch
from sports.common.ball import BallTracker, BallAnnotator
from sports.common.team import TeamClassifier
from sports.common.view import ViewTransformer
from sports.configs.soccer import SoccerPitchConfiguration

# Import database managers
from soccer_database_manager import SoccerDatabaseManager
from enhanced_jersey_manager import EnhancedJerseyManager

# Configuration
CONFIG = SoccerPitchConfiguration()

# Model paths
PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
PLAYER_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/football-player-detection.pt')
PITCH_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/football-pitch-detection.pt')
BALL_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/football-ball-detection.pt')

# Class IDs
BALL_CLASS_ID = 0
GOALKEEPER_CLASS_ID = 1
PLAYER_CLASS_ID = 2
REFEREE_CLASS_ID = 3

def draw_tactical_board_with_db(
    frame_shape: tuple,
    detections: sv.Detections,
    classified_teams: np.ndarray,
    ball_detections: sv.Detections,
    keypoints: sv.KeyPoints,
    jersey_manager: EnhancedJerseyManager,
    db: SoccerDatabaseManager,
    team_a_color: tuple,
    team_b_color: tuple,
    formation_a: str,
    formation_b: str,
    team_a_name: str,
    team_b_name: str,
    board_w: int,
    board_h: int
) -> np.ndarray:
    """Draw tactical board with database-integrated player info"""
    
    # Draw pitch
    pitch = draw_pitch(config=CONFIG)
    tactical_board = cv2.resize(pitch, (board_w, board_h), 
                                interpolation=cv2.INTER_LANCZOS4)
    
    # Check keypoints
    mask = (keypoints.xy[0][:, 0] > 1) & (keypoints.xy[0][:, 1] > 1)
    
    if not mask.any() or mask.sum() < 4:
        return tactical_board
    
    try:
        transformer = ViewTransformer(
            source=keypoints.xy[0][mask].astype(np.float32),
            target=np.array(CONFIG.vertices)[mask].astype(np.float32)
        )
        
        if len(detections) > 0:
            player_xy = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
            transformed_xy = transformer.transform_points(points=player_xy)
            
            scale_x = board_w / CONFIG.length
            scale_y = board_h / CONFIG.width
            
            board_positions = transformed_xy.copy()
            board_positions[:, 0] *= scale_x
            board_positions[:, 1] *= scale_y
            
            # Draw players
            for i in range(len(detections)):
                x, y = board_positions[i]
                tracker_id = detections.tracker_id[i]
                classified_team = classified_teams[i]
                
                # Get color based on team
                if classified_team == 0:
                    color = team_a_color
                elif classified_team == 1:
                    color = team_b_color
                else:
                    color = (200, 200, 200)  # Gray for referees
                
                # Get jersey number
                jersey_num = jersey_manager.get_jersey(tracker_id)
                
                # Draw circle
                radius = 15
                cv2.circle(tactical_board, (int(x), int(y)), radius, color, -1)
                cv2.circle(tactical_board, (int(x), int(y)), radius, (255, 255, 255), 2)
                
                # Draw number
                text = str(jersey_num)
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
            
            ball_board = transformed_ball.copy()
            ball_board[:, 0] *= scale_x
            ball_board[:, 1] *= scale_y
            
            for bx, by in ball_board:
                # Boundary check
                bx = max(8, min(board_w - 8, bx))
                by = max(8, min(board_h - 8, by))
                cv2.circle(tactical_board, (int(bx), int(by)), 8, (255, 255, 255), -1)
    
    except Exception as e:
        print(f"Tactical board error: {e}")
    
    # Add formations at bottom
    y_offset = board_h - 40
    
    cv2.putText(tactical_board, f"{team_a_name}: {formation_a}", 
                (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, team_a_color, 2)
    
    text_size = cv2.getTextSize(f"{team_b_name}: {formation_b}", 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    cv2.putText(tactical_board, f"{team_b_name}: {formation_b}", 
                (board_w - text_size[0] - 20, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, team_b_color, 2)
    
    # Add title
    cv2.putText(tactical_board, "TACTICAL BOARD", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    return tactical_board

def process_video_with_database(
    source_path: str,
    target_path: str,
    match_id: int,
    device: str = 'cpu',
    db_config: dict = None,
    max_frames: int = None
):
    """
    Process video with full database integration
    
    Args:
        source_path: Input video path
        target_path: Output video path
        match_id: Match ID from database
        device: 'cpu', 'cuda', or 'mps'
        db_config: Database connection config
        max_frames: Maximum frames to process (None for all)
    """
    
    # Ensure target_path is in video_outputs directory
    if not target_path.startswith('video_outputs/'):
        target_path = os.path.join('video_outputs', os.path.basename(target_path))
    
    # Create video_outputs directory if it doesn't exist
    os.makedirs('video_outputs', exist_ok=True)
    
    # Connect to database
    db = SoccerDatabaseManager(db_config)
    db.connect()
    
    try:
        # Setup match context
        db.setup_match(match_id)
        
        # Get team colors from database
        team_a_color_hex = db.get_team_color(db.team_a_id) or '#00BFFF'
        team_b_color_hex = db.get_team_color(db.team_b_id) or '#FF1493'
        
        from supervision import Color
        TEAM_A_COLOR = Color.from_hex(team_a_color_hex).as_bgr()
        TEAM_B_COLOR = Color.from_hex(team_b_color_hex).as_bgr()
        
        # Get team names
        team_a_name = db.get_team_name(db.team_a_id)
        team_b_name = db.get_team_name(db.team_b_id)
        
        # Get formations
        formation_a = db.get_formation(db.team_a_id) or "4-3-3"
        formation_b = db.get_formation(db.team_b_id) or "4-4-2"
        
        print(f"\n{'='*60}")
        print(f"MATCH: {team_a_name} vs {team_b_name}")
        print(f"Formations: {formation_a} vs {formation_b}")
        print(f"Colors: {team_a_color_hex} vs {team_b_color_hex}")
        print(f"{'='*60}\n")
        
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
        
        slicer = sv.InferenceSlicer(callback=ball_callback, slice_wh=(640, 640))
        
        # Train team classifier
        print("Training team classifier...")
        team_classifier = TeamClassifier(device=device)
        
        STRIDE = 60
        frame_generator = sv.get_video_frames_generator(source_path=source_path, stride=STRIDE)
        crops = []
        
        for frame in frame_generator:
            result = player_model(frame, imgsz=1280, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(result)
            players = detections[detections.class_id == PLAYER_CLASS_ID]
            crops.extend([sv.crop_image(frame, xyxy) for xyxy in players.xyxy])
            if len(crops) > 500:
                break
        
        team_classifier.fit(crops)
        
        # Initialize managers
        jersey_manager = EnhancedJerseyManager(db)
        
        # Setup video
        video_info = sv.VideoInfo.from_video_path(source_path)
        
        video_w, video_h = video_info.width, video_info.height
        board_w, board_h = 1200, 800
        
        if video_h != board_h:
            aspect = video_w / video_h
            video_h = board_h
            video_w = int(board_h * aspect)
        
        output_w = video_w + board_w
        output_h = max(video_h, board_h)
        
        # Set total frames
        total_frames = min(video_info.total_frames, max_frames) if max_frames else video_info.total_frames
        
        output_video_info = sv.VideoInfo(
            width=output_w,
            height=output_h,
            fps=video_info.fps,
            total_frames=total_frames
        )
        
        # Annotators
        COLORS = [team_a_color_hex, team_b_color_hex, '#FFD700', '#32CD32']
        ellipse_annotator = sv.EllipseAnnotator(
            color=sv.ColorPalette.from_hex(COLORS), thickness=2
        )
        label_annotator = sv.LabelAnnotator(
            color=sv.ColorPalette.from_hex(COLORS),
            text_color=sv.Color.WHITE,
            text_position=sv.Position.BOTTOM_CENTER
        )
        
        print("Processing video with database storage...")
        
        frame_generator = sv.get_video_frames_generator(source_path=source_path)
        
        # Position batch for database
        position_batch = []
        BATCH_SIZE = 330  # 30 frames * 11 players/team
        
        with sv.VideoSink(target_path, output_video_info) as sink:
            frame_idx = 0
            
            for frame in tqdm(frame_generator, total=total_frames):
                if frame_idx >= total_frames:
                    break
                
                # Resize frame
                if frame.shape[0] != video_h or frame.shape[1] != video_w:
                    frame = cv2.resize(frame, (video_w, video_h))
                
                # Calculate timestamp
                timestamp_sec = frame_idx / video_info.fps
                
                # Player detection & tracking
                player_result = player_model(frame, imgsz=1280, verbose=False)[0]
                detections = sv.Detections.from_ultralytics(player_result)
                detections = tracker.update_with_detections(detections)
                
                # Ball detection & tracking
                ball_detections = slicer(frame).with_nms(threshold=0.1)
                ball_detections = ball_tracker.update(ball_detections)
                
                # Pitch detection
                pitch_result = pitch_model(frame, verbose=False)[0]
                keypoints = sv.KeyPoints.from_ultralytics(pitch_result)
                
                # Team classification
                players = detections[detections.class_id == PLAYER_CLASS_ID]
                goalkeepers = detections[detections.class_id == GOALKEEPER_CLASS_ID]
                referees = detections[detections.class_id == REFEREE_CLASS_ID]
                
                if len(players) > 0:
                    player_crops = [sv.crop_image(frame, xyxy) for xyxy in players.xyxy]
                    players_team_id = team_classifier.predict(player_crops)
                else:
                    players_team_id = np.array([])
                
                if len(goalkeepers) > 0:
                    gk_crops = [sv.crop_image(frame, xyxy) for xyxy in goalkeepers.xyxy]
                    goalkeepers_team_id = team_classifier.predict(gk_crops)
                else:
                    goalkeepers_team_id = np.array([])
                
                # Merge detections
                all_detections = sv.Detections.merge([players, goalkeepers, referees])
                classified_teams = np.concatenate([
                    players_team_id,
                    goalkeepers_team_id,
                    np.array([2] * len(referees))  # Referees
                ]) if len(all_detections) > 0 else np.array([])
                
                # Jersey assignment and database storage
                if len(all_detections) > 0:
                    
                    # Setup coordinate transformer
                    mask = (keypoints.xy[0][:, 0] > 1) & (keypoints.xy[0][:, 1] > 1)
                    transformer = None
                    
                    if mask.any() and mask.sum() >= 4:
                        try:
                            transformer = ViewTransformer(
                                source=keypoints.xy[0][mask].astype(np.float32),
                                target=np.array(CONFIG.vertices)[mask].astype(np.float32)
                            )
                        except:
                            pass
                    
                    # Process each player
                    for i in range(len(all_detections)):
                        tracker_id = all_detections.tracker_id[i]
                        classified_team = classified_teams[i]
                        
                        # Skip referees for jersey assignment
                        if classified_team == 2:
                            continue
                        
                        # Assign jersey
                        jersey_num = jersey_manager.assign_jersey(tracker_id, classified_team)
                        
                        # Get actual team_id
                        team_id = db.get_team_id_from_classification(classified_team)
                        
                        if team_id is None:
                            continue
                        
                        # Get coordinates
                        video_xy = all_detections.get_anchors_coordinates(
                            anchor=sv.Position.BOTTOM_CENTER
                        )[i]
                        
                        # Transform coordinates
                        if transformer:
                            try:
                                pitch_xy = transformer.transform_points(
                                    points=video_xy.reshape(1, -1)
                                )[0]
                                
                                # Add to batch
                                position_batch.append({
                                    'frame_id': frame_idx,
                                    'timestamp': float(timestamp_sec),
                                    'jersey_number': int(jersey_num),
                                    'team_id': int(team_id),
                                    'x': float(pitch_xy[0]),
                                    'y': float(pitch_xy[1]),
                                    'confidence': float(all_detections.confidence[i]) 
                                                 if hasattr(all_detections, 'confidence') else 1.0,
                                    'tracker_id': int(tracker_id)
                                })
                            except:
                                pass
                    
                    # Batch insert to database
                    if len(position_batch) >= BATCH_SIZE:
                        db.insert_tracked_positions_batch(position_batch)
                        position_batch = []
                
                # Annotate video frame (LEFT SIDE)
                annotated_video = frame.copy()
                
                if len(all_detections) > 0:
                    # Show jersey numbers and names
                    labels = []
                    for tid in all_detections.tracker_id:
                        jersey = jersey_manager.get_jersey(tid)
                        name = jersey_manager.get_player_name(tid)
                        labels.append(f"#{jersey}")  # Or f"{name} #{jersey}"
                    
                    annotated_video = ellipse_annotator.annotate(annotated_video, all_detections)
                    annotated_video = label_annotator.annotate(annotated_video, all_detections, labels)
                
                annotated_video = ball_annotator.annotate(annotated_video, ball_detections)
                
                # Add title
                cv2.putText(annotated_video, f"{team_a_name} vs {team_b_name}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # Draw tactical board (RIGHT SIDE)
                tactical_board = draw_tactical_board_with_db(
                    (video_h, video_w),
                    all_detections,
                    classified_teams,
                    ball_detections,
                    keypoints,
                    jersey_manager,
                    db,
                    TEAM_A_COLOR,
                    TEAM_B_COLOR,
                    formation_a,
                    formation_b,
                    team_a_name,
                    team_b_name,
                    board_w,
                    board_h
                )
                
                # Combine side-by-side
                output_frame = np.hstack([annotated_video, tactical_board])
                
                sink.write_frame(output_frame)
                frame_idx += 1
        
        # Insert remaining positions
        if position_batch:
            db.insert_tracked_positions_batch(position_batch)
        
        print(f"\n✓ Processing complete!")
        print(f"✓ Output video: {target_path}")
        print(f"✓ Tracked positions stored in database")
        print(f"✓ Processed {frame_idx} frames")
    
    finally:
        db.disconnect()

def main():
    parser = argparse.ArgumentParser(
        description="Split-Screen Soccer Analysis with Database Integration"
    )
    parser.add_argument("--source_video_path", type=str, required=True,
                       help="Path to input video")
    parser.add_argument("--target_video_path", type=str, required=True,
                       help="Path to output video")
    parser.add_argument("--match_id", type=int, required=True,
                       help="Match ID from database")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device: cpu, cuda, or mps")
    parser.add_argument("--max_frames", type=int, default=None,
                       help="Maximum frames to process (None for all)")
    
    # Database config
    parser.add_argument("--db_host", type=str, default="localhost")
    parser.add_argument("--db_name", type=str, default="soccer_analysis")
    parser.add_argument("--db_user", type=str, default="postgres")
    parser.add_argument("--db_password", type=str, required=True,
                       help="Database password")
    parser.add_argument("--db_port", type=int, default=5432)
    
    args = parser.parse_args()
    
    db_config = {
        'host': args.db_host,
        'database': args.db_name,
        'user': args.db_user,
        'password': args.db_password,
        'port': args.db_port
    }
    
    process_video_with_database(
        source_path=args.source_video_path,
        target_path=args.target_video_path,
        match_id=args.match_id,
        device=args.device,
        db_config=db_config,
        max_frames=args.max_frames
    )

if __name__ == "__main__":
    main()
