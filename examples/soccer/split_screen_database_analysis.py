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
from local_data_exporter import LocalDataExporter

# Mock database classes for testing
class MockCursor:
    def __init__(self):
        self.results = []
    
    def execute(self, query, params=None):
        pass
    
    def fetchall(self):
        return self.results
    
    def fetchone(self):
        return self.results[0] if self.results else None
    
    def close(self):
        pass

class MockConnection:
    def __init__(self):
        self.cursor = MockCursor()
    
    def commit(self):
        pass
    
    def rollback(self):
        pass
    
    def close(self):
        pass

class MockDatabaseManager(SoccerDatabaseManager):
    """Mock database manager for testing without PostgreSQL"""
    
    def __init__(self, db_config: dict):
        self.config = db_config
        self.conn = MockConnection()
        self.cursor = self.conn.cursor
        
        # Mock data
        self.tracker_to_player = {}
        self.jersey_to_player = {}
        self.current_match_id = 1
        self.team_a_id = 1
        self.team_b_id = 2
        
        # Mock teams
        self.teams = {
            1: {'name': 'Real Madrid', 'color': '#FFFFFF', 'formation': '4-3-3'},
            2: {'name': 'Barcelona', 'color': '#A50044', 'formation': '4-4-2'}
        }
        
        # Mock players
        self.players = {
            1: {  # Real Madrid
                1: {'name': 'Courtois', 'position': 'GK'},
                2: {'name': 'Carvajal', 'position': 'RB'},
                3: {'name': 'Militao', 'position': 'CB'},
                4: {'name': 'Alaba', 'position': 'CB'},
                5: {'name': 'Mendy', 'position': 'LB'},
                6: {'name': 'Casemiro', 'position': 'CDM'},
                7: {'name': 'Modric', 'position': 'CM'},
                8: {'name': 'Kroos', 'position': 'CM'},
                9: {'name': 'Vinicius', 'position': 'LW'},
                10: {'name': 'Benzema', 'position': 'ST'},
                11: {'name': 'Asensio', 'position': 'RW'}
            },
            2: {  # Barcelona
                1: {'name': 'ter Stegen', 'position': 'GK'},
                2: {'name': 'Dest', 'position': 'RB'},
                3: {'name': 'Pique', 'position': 'CB'},
                4: {'name': 'Lenglet', 'position': 'CB'},
                5: {'name': 'Alba', 'position': 'LB'},
                6: {'name': 'Busquets', 'position': 'CDM'},
                7: {'name': 'Pedri', 'position': 'CM'},
                8: {'name': 'De Jong', 'position': 'CM'},
                9: {'name': 'Dembele', 'position': 'LW'},
                10: {'name': 'Depay', 'position': 'ST'},
                11: {'name': 'Gavi', 'position': 'RW'}
            }
        }
        
        self.position_data = []
    
    def connect(self):
        print("✓ Using mock database (no PostgreSQL connection required)")
    
    def disconnect(self):
        pass
    
    def setup_match(self, match_id: int):
        self.current_match_id = match_id
        print(f"✓ Mock match setup: {self.teams[1]['name']} vs {self.teams[2]['name']}")
    
    def get_team_id_from_classification(self, classified_team: int) -> int:
        if classified_team == 0:
            return self.team_a_id
        elif classified_team == 1:
            return self.team_b_id
        return None
    
    def assign_tracker_to_jersey(self, tracker_id: int, jersey_number: int, 
                                 classified_team: int) -> dict:
        team_id = self.get_team_id_from_classification(classified_team)
        if team_id is None:
            return None
        
        player_info = self.players[team_id].get(jersey_number, {'name': f'Player {jersey_number}', 'position': 'Unknown'})
        
        assignment = {
            'player_id': f"{team_id}_{jersey_number}",
            'jersey_number': jersey_number,
            'team_id': team_id,
            'name': player_info['name'],
            'position': player_info['position']
        }
        
        self.tracker_to_player[tracker_id] = assignment
        self.jersey_to_player[(team_id, jersey_number)] = f"{team_id}_{jersey_number}"
        
        return assignment
    
    def get_player_info(self, tracker_id: int) -> dict:
        return self.tracker_to_player.get(tracker_id)
    
    def insert_tracked_positions_batch(self, positions: list):
        self.position_data.extend(positions)
        print(f"✓ Mock: Inserted {len(positions)} positions (total: {len(self.position_data)})")
    
    def insert_event(self, event_type: str, team_id: int, timestamp: str,
                    jersey_number: int, details: dict = None):
        print(f"✓ Mock: Event {event_type} for team {team_id}, jersey {jersey_number}")
    
    def insert_substitution(self, team_id: int, jersey_out: int, jersey_in: int, 
                          timestamp: str):
        print(f"✓ Mock: Substitution {jersey_out} -> {jersey_in} for team {team_id}")
    
    def get_formation(self, team_id: int) -> str:
        return self.teams.get(team_id, {}).get('formation', '4-4-2')
    
    def get_team_color(self, team_id: int) -> str:
        return self.teams.get(team_id, {}).get('color', '#FFFFFF')
    
    def get_team_name(self, team_id: int) -> str:
        return self.teams.get(team_id, {}).get('name', f'Team {team_id}')

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

def clip_to_pitch_bounds(
    x: float, 
    y: float, 
    board_w: int, 
    board_h: int,
    margin: int = 20
) -> tuple:
    """
    Clip coordinates to stay within pitch boundaries
    
    Args:
        x, y: Transformed coordinates
        board_w, board_h: Tactical board dimensions
        margin: Pixels from edge (default 20)
    
    Returns:
        (clipped_x, clipped_y)
    """
    min_x = margin
    max_x = board_w - margin
    min_y = margin
    max_y = board_h - margin
    
    clipped_x = max(min_x, min(x, max_x))
    clipped_y = max(min_y, min(y, max_y))
    
    return clipped_x, clipped_y

def is_within_pitch_bounds(
    x: float,
    y: float,
    board_w: int,
    board_h: int,
    margin: int = 20
) -> bool:
    """Check if coordinates are within pitch bounds"""
    return (margin <= x <= board_w - margin and 
            margin <= y <= board_h - margin)

def get_contrasting_text_color(team_color_bgr: tuple) -> tuple:
    """
    Get contrasting text color (black or white) based on team color brightness
    
    Args:
        team_color_bgr: Team color in BGR format
    
    Returns:
        (0, 0, 0) for black or (255, 255, 255) for white
    """
    b, g, r = team_color_bgr
    
    # Calculate perceived brightness (luminance)
    brightness = (0.299 * r + 0.587 * g + 0.114 * b)
    
    # Use black text for bright colors, white for dark colors
    if brightness > 128:
        return (0, 0, 0)  # Black
    else:
        return (255, 255, 255)  # White

def draw_player_circle_auto_contrast(
    tactical_board: np.ndarray,
    x: float, 
    y: float,
    jersey_num: int,
    team_color: tuple,
    radius: int = 15
):
    """Draw player circle with automatic text color contrast"""
    
    # Draw filled circle
    cv2.circle(tactical_board, (int(x), int(y)), radius, team_color, -1)
    
    # Draw border (black for visibility)
    cv2.circle(tactical_board, (int(x), int(y)), radius, (0, 0, 0), 2)
    
    # Get contrasting text color
    text_color = get_contrasting_text_color(team_color)
    
    # Draw text
    text = str(jersey_num)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 2
    
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = int(x - text_size[0] / 2)
    text_y = int(y + text_size[1] / 2)
    
    cv2.putText(tactical_board, text, (text_x, text_y),
                font, font_scale, text_color, thickness)
    
    return tactical_board

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
    """Draw tactical board with boundary clipping and improved visibility"""
    
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
            
            # Draw players with boundary checking
            for i in range(len(detections)):
                x, y = board_positions[i]
                tracker_id = detections.tracker_id[i]
                classified_team = classified_teams[i]
                
                # CLIP TO BOUNDS
                x, y = clip_to_pitch_bounds(x, y, board_w, board_h, margin=30)
                
                # Skip if still somehow out of bounds
                if not is_within_pitch_bounds(x, y, board_w, board_h, margin=25):
                    continue
                
                # Choose color
                if classified_team == 0:
                    color = team_a_color
                elif classified_team == 1:
                    color = team_b_color
                else:
                    color = (200, 200, 200)
                
                jersey_num = jersey_manager.get_jersey(tracker_id)
                
                # Draw with improved visibility
                tactical_board = draw_player_circle_auto_contrast(
                    tactical_board, x, y, jersey_num, color, radius=15
                )
        
        # Draw ball with clipping
        if ball_detections is not None and len(ball_detections) > 0:
            ball_xy = ball_detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
            transformed_ball = transformer.transform_points(points=ball_xy)
            
            ball_board = transformed_ball.copy()
            ball_board[:, 0] *= scale_x
            ball_board[:, 1] *= scale_y
            
            for bx, by in ball_board:
                # CLIP BALL TO BOUNDS
                bx, by = clip_to_pitch_bounds(bx, by, board_w, board_h, margin=30)
                
                if is_within_pitch_bounds(bx, by, board_w, board_h, margin=25):
                    # Draw ball with black border for visibility
                    cv2.circle(tactical_board, (int(bx), int(by)), 10, (255, 255, 255), -1)
                    cv2.circle(tactical_board, (int(bx), int(by)), 10, (0, 0, 0), 2)
    
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

def draw_data_overlay(
    frame: np.ndarray,
    frame_idx: int,
    timestamp: float,
    detections: sv.Detections,
    jersey_manager: EnhancedJerseyManager,
    db: SoccerDatabaseManager,
    exporter: LocalDataExporter
) -> np.ndarray:
    """
    Draw real-time data overlay showing coordinates and stats
    
    Args:
        frame: Video frame
        frame_idx: Current frame number
        timestamp: Current timestamp in seconds
        detections: Player detections
        jersey_manager: Jersey assignment manager
        db: Database manager
        exporter: Local data exporter
    
    Returns:
        Frame with data overlay
    """
    h, w = frame.shape[:2]
    
    # Create semi-transparent overlay panel
    overlay = frame.copy()
    
    # Draw panel background (bottom-left corner)
    panel_h = 200
    panel_w = 350
    panel_x = 10
    panel_y = h - panel_h - 10
    
    cv2.rectangle(overlay, (panel_x, panel_y), 
                 (panel_x + panel_w, panel_y + panel_h),
                 (0, 0, 0), -1)
    
    # Blend with original frame
    alpha = 0.7
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    
    # Draw panel border
    cv2.rectangle(frame, (panel_x, panel_y),
                 (panel_x + panel_w, panel_y + panel_h),
                 (255, 255, 255), 2)
    
    # Title
    cv2.putText(frame, "TRACKING DATA", (panel_x + 10, panel_y + 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Frame info
    y_offset = panel_y + 50
    cv2.putText(frame, f"Frame: {frame_idx}", (panel_x + 10, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    y_offset += 25
    cv2.putText(frame, f"Time: {timestamp:.2f}s", (panel_x + 10, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    y_offset += 25
    cv2.putText(frame, f"Players: {len(detections)}", (panel_x + 10, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Show first 3 players' data
    y_offset += 35
    cv2.putText(frame, "Sample Positions:", (panel_x + 10, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 255, 150), 1)
    
    for i in range(min(3, len(detections))):
        y_offset += 20
        tracker_id = detections.tracker_id[i]
        jersey = jersey_manager.get_jersey(tracker_id)
        player_info = db.get_player_info(tracker_id)
        name = player_info['name'] if player_info else f"P{tracker_id}"
        
        xy = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)[i]
        
        text = f"#{jersey} {name[:8]}: ({int(xy[0])},{int(xy[1])})"
        cv2.putText(frame, text, (panel_x + 15, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    # Export stats
    y_offset += 30
    total_exports = len(exporter.position_buffer)
    cv2.putText(frame, f"Exported: {total_exports} records", 
               (panel_x + 10, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 200, 255), 1)
    
    return frame

def process_video_with_database(
    source_path: str,
    target_path: str,
    match_id: int,
    device: str = 'cpu',
    db_config: dict = None,
    max_frames: int = None,
    export_local: bool = True
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
    
    # Connect to database (use mock if PostgreSQL not available)
    try:
        db = SoccerDatabaseManager(db_config)
        db.connect()
    except Exception as e:
        print(f"PostgreSQL not available ({e}), using mock database")
        db = MockDatabaseManager(db_config)
        db.connect()
    
    # Initialize local exporter
    exporter = LocalDataExporter(output_dir="tracking_data") if export_local else None
    
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
        
        # Position batch for database
        position_batch = []
        BATCH_SIZE = 330  # 30 frames * 11 players/team
        
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
                                
                                # Calculate board coordinates
                                scale_x = board_w / CONFIG.length
                                scale_y = board_h / CONFIG.width
                                board_xy = pitch_xy.copy()
                                board_xy[0] *= scale_x
                                board_xy[1] *= scale_y
                                
                                # Clip to bounds
                                board_xy[0], board_xy[1] = clip_to_pitch_bounds(
                                    board_xy[0], board_xy[1], board_w, board_h, margin=30
                                )
                                
                                # Get player info
                                player_info = db.get_player_info(tracker_id)
                                player_name = player_info['name'] if player_info else f"Player {tracker_id}"
                                
                                confidence = float(all_detections.confidence[i]) \
                                            if hasattr(all_detections, 'confidence') else 1.0
                                
                                # Add to database batch
                                position_batch.append({
                                    'frame_id': frame_idx,
                                    'timestamp': float(timestamp_sec),
                                    'jersey_number': int(jersey_num),
                                    'team_id': int(team_id),
                                    'x': float(pitch_xy[0]),
                                    'y': float(pitch_xy[1]),
                                    'confidence': confidence,
                                    'tracker_id': int(tracker_id)
                                })
                                
                                # Export locally
                                if exporter:
                                    exporter.add_position(
                                        frame_idx, timestamp_sec, tracker_id,
                                        jersey_num, team_id, player_name,
                                        float(video_xy[0]), float(video_xy[1]),
                                        float(pitch_xy[0]), float(pitch_xy[1]),
                                        float(board_xy[0]), float(board_xy[1]),
                                        confidence
                                    )
                            
                            except Exception as e:
                                print(f"Coordinate transform error: {e}")
                    
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
                
                # Add data overlay to video
                if exporter:
                    annotated_video = draw_data_overlay(
                        annotated_video, frame_idx, timestamp_sec,
                        all_detections, jersey_manager, db, exporter
                    )
                
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
        
        # Export all local data
        if exporter:
            exporter.export_all()
            print(f"\n{'='*60}")
            print("LOCAL DATA EXPORTED")
            print(f"{'='*60}")
            print(f"CSV: {exporter.positions_csv}")
            print(f"JSON: {exporter.summary_json}")
            print(f"Report: {exporter.stats_txt}")
            print(f"{'='*60}\n")
        
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
    parser.add_argument("--export_local", action="store_true",
                       help="Export tracking data to local CSV/JSON files")
    
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
        max_frames=args.max_frames,
        export_local=args.export_local
    )

if __name__ == "__main__":
    main()
