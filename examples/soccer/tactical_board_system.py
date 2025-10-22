"""
Tactical Board System - Focus on homography transformation and jersey assignment
Priority: See players on tactical board, assign numbers, LLM-based formation assignment
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
import json

from sports.annotators.soccer import draw_pitch, draw_points_on_pitch
from sports.common.ball import BallTracker, BallAnnotator
from sports.common.team import TeamClassifier
from sports.common.view import ViewTransformer
from sports.configs.soccer import SoccerPitchConfiguration

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")
warnings.filterwarnings("ignore", category=UserWarning, module="supervision")
warnings.filterwarnings("ignore", category=RuntimeWarning)

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Constants
MAX_FRAMES = 50  # Test with 50 frames
BALL_CLASS_ID = 0
PLAYER_CLASS_ID = 1
GOALKEEPER_CLASS_ID = 2
REFEREE_CLASS_ID = 3

# Colors for teams
TEAM_COLORS = ['#FF1493', '#00BFFF', '#FF6347', '#FFD700', '#32CD32', '#FF69B4', '#FF0000']

# Soccer pitch configuration
CONFIG = SoccerPitchConfiguration()

class TacticalBoardManager:
    """Manages tactical board interactions and jersey assignments"""
    
    def __init__(self):
        self.jersey_assignments = {}  # tracker_id -> jersey_number
        self.team_assignments = {}    # tracker_id -> team_id
        self.formation_input = ""
        self.assignment_mode = False
        self.selected_tracker_id = None
        self.current_team_id = 0
        self.jersey_input = ""
        
    def assign_jersey(self, tracker_id: int, jersey_number: int, team_id: int):
        """Assign jersey number to player on tactical board"""
        self.jersey_assignments[tracker_id] = jersey_number
        self.team_assignments[tracker_id] = team_id
        print(f"✓ Assigned jersey #{jersey_number} (Team {team_id}) to tracker {tracker_id}")
    
    def get_jersey_number(self, tracker_id: int) -> Optional[int]:
        """Get jersey number for tracker ID"""
        return self.jersey_assignments.get(tracker_id)
    
    def get_team_id(self, tracker_id: int) -> Optional[int]:
        """Get team ID for tracker ID"""
        return self.team_assignments.get(tracker_id)
    
    def set_formation_input(self, formation_text: str):
        """Set formation input from coach"""
        self.formation_input = formation_text
        print(f"✓ Formation input: {formation_text}")
    
    def save_assignments(self, filepath: str):
        """Save assignments to JSON"""
        data = {
            'jersey_assignments': self.jersey_assignments,
            'team_assignments': self.team_assignments,
            'formation_input': self.formation_input
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"✓ Saved assignments to {filepath}")
    
    def load_assignments(self, filepath: str):
        """Load assignments from JSON"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            self.jersey_assignments = data.get('jersey_assignments', {})
            self.team_assignments = data.get('team_assignments', {})
            self.formation_input = data.get('formation_input', "")
            print(f"✓ Loaded assignments from {filepath}")
        except Exception as e:
            print(f"✗ Failed to load assignments: {e}")

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

def draw_tactical_board(frame: np.ndarray, player_detections: sv.Detections, ball_detections: sv.Detections, 
                       keypoints: sv.KeyPoints, manager: TacticalBoardManager) -> np.ndarray:
    """Draw tactical board with player positions and jersey numbers"""
    h, w = frame.shape[:2]
    
    # Create tactical board (larger, more prominent)
    board_w = w // 2  # Half the screen width
    board_h = h // 2  # Half the screen height
    
    # Position tactical board in center-right
    board_x = w - board_w - 20
    board_y = (h - board_h) // 2
    
    # Draw tactical board background
    tactical_board = np.zeros((board_h, board_w, 3), dtype=np.uint8)
    tactical_board[:] = (40, 40, 40)  # Dark gray background
    
    if len(player_detections) == 0 and len(ball_detections) == 0:
        # Draw empty pitch
        pitch = draw_pitch(config=CONFIG)
        pitch_resized = cv2.resize(pitch, (board_w, board_h))
        tactical_board = pitch_resized
    else:
        try:
            # Check if we have valid keypoints for homography
            mask = (keypoints.xy[0][:, 0] > 1) & (keypoints.xy[0][:, 1] > 1)
            if mask.any() and len(keypoints.xy[0]) >= 4:
                # Create homography transformer
                transformer = ViewTransformer(
                    source=keypoints.xy[0][mask].astype(np.float32),
                    target=np.array(CONFIG.vertices)[mask].astype(np.float32)
                )
                
                # Draw pitch
                pitch = draw_pitch(config=CONFIG)
                pitch_resized = cv2.resize(pitch, (board_w, board_h))
                tactical_board = pitch_resized.copy()
                
                # Transform player positions to tactical board
                if len(player_detections) > 0:
                    player_xy = player_detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
                    transformed_player_xy = transformer.transform_points(points=player_xy)
                    
                    # Scale to tactical board dimensions
                    scale_x = board_w / CONFIG.length
                    scale_y = board_h / CONFIG.width
                    
                    for i, tracker_id in enumerate(player_detections.tracker_id):
                        if tracker_id is None:
                            continue
                        
                        # Get transformed position
                        tx, ty = transformed_player_xy[i]
                        
                        # Scale to tactical board
                        board_x_pos = int(tx * scale_x)
                        board_y_pos = int(ty * scale_y)
                        
                        # Ensure position is within bounds
                        board_x_pos = max(10, min(board_w - 10, board_x_pos))
                        board_y_pos = max(10, min(board_h - 10, board_y_pos))
                        
                        # Get team and jersey info
                        team_id = manager.get_team_id(tracker_id)
                        jersey_number = manager.get_jersey_number(tracker_id)
                        
                        # Choose color
                        if team_id is not None:
                            color = tuple(int(c, 16) for c in [TEAM_COLORS[team_id][i:i+2] for i in (1, 3, 5)])
                        else:
                            color = (128, 128, 128)  # Gray for unassigned
                        
                        # Draw player circle
                        cv2.circle(tactical_board, (board_x_pos, board_y_pos), 15, color, -1)
                        cv2.circle(tactical_board, (board_x_pos, board_y_pos), 15, (255, 255, 255), 2)
                        
                        # Draw jersey number or tracker ID
                        if jersey_number is not None:
                            label = str(jersey_number)
                            font_scale = 0.6
                        else:
                            label = f"ID:{tracker_id}"
                            font_scale = 0.4
                        
                        # Draw label
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
                        label_x = board_x_pos - label_size[0] // 2
                        label_y = board_y_pos + label_size[1] // 2
                        cv2.putText(tactical_board, label, (label_x, label_y), 
                                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)
                        
                        # Highlight selected player
                        if manager.selected_tracker_id == tracker_id:
                            cv2.circle(tactical_board, (board_x_pos, board_y_pos), 20, (0, 255, 255), 3)
                
                # Transform ball position
                if len(ball_detections) > 0:
                    ball_xy = ball_detections.get_anchors_coordinates(anchor=sv.Position.CENTER)
                    transformed_ball_xy = transformer.transform_points(points=ball_xy)
                    
                    for i in range(len(ball_detections)):
                        tx, ty = transformed_ball_xy[i]
                        board_x_pos = int(tx * scale_x)
                        board_y_pos = int(ty * scale_y)
                        board_x_pos = max(10, min(board_w - 10, board_x_pos))
                        board_y_pos = max(10, min(board_h - 10, board_y_pos))
                        
                        # Draw ball
                        cv2.circle(tactical_board, (board_x_pos, board_y_pos), 8, (255, 255, 255), -1)
                        cv2.circle(tactical_board, (board_x_pos, board_y_pos), 8, (0, 0, 0), 2)
                        
            else:
                # No valid keypoints, draw empty pitch
                pitch = draw_pitch(config=CONFIG)
                pitch_resized = cv2.resize(pitch, (board_w, board_h))
                tactical_board = pitch_resized
                
        except Exception as e:
            print(f"Tactical board error: {e}")
            # Fallback to empty pitch
            pitch = draw_pitch(config=CONFIG)
            pitch_resized = cv2.resize(pitch, (board_w, board_h))
            tactical_board = pitch_resized
    
    # Overlay tactical board on main frame
    frame[board_y:board_y+board_h, board_x:board_x+board_w] = tactical_board
    
    # Draw border around tactical board
    cv2.rectangle(frame, (board_x, board_y), (board_x+board_w, board_y+board_h), (255, 255, 255), 2)
    
    return frame

def train_team_classifier(team_classifier: TeamClassifier, video_path: str, max_frames: int = 50):
    """Train team classifier on video data"""
    print("Collecting player crops for team classification...")
    
    PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PLAYER_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/football-player-detection.pt')
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

def tactical_board_analysis(source_video_path: str, target_video_path: str, device: str = "cpu"):
    """Run tactical board analysis with focus on homography and jersey assignment"""
    
    # Initialize tactical board manager
    tactical_manager = TacticalBoardManager()
    
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
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(target_video_path, fourcc, fps, (width, height))
    
    frame_idx = 0
    paused = False
    
    print("=" * 60)
    print("TACTICAL BOARD SYSTEM")
    print("=" * 60)
    print("Controls:")
    print("  SPACE - Pause/Resume")
    print("  A - Toggle jersey assignment mode")
    print("  1-2 - Select team (1 or 2)")
    print("  Click player on tactical board to select")
    print("  0-9 - Enter jersey number")
    print("  ENTER - Assign jersey")
    print("  ESC - Cancel assignment")
    print("  F - Input formation (console)")
    print("  S - Save assignments")
    print("  L - Load assignments")
    print("  Q - Quit")
    print("=" * 60)
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_idx += 1
            if frame_idx > MAX_FRAMES:
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
            
            # Draw tactical board
            annotated_frame = draw_tactical_board(frame, all_player_detections, ball_detections, 
                                                keypoints, tactical_manager)
            
            # Add labels
            cv2.putText(annotated_frame, "TACTICAL BOARD SYSTEM", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            if tactical_manager.assignment_mode:
                cv2.putText(annotated_frame, f"ASSIGNMENT MODE - Team: {tactical_manager.current_team_id}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"Jersey: {tactical_manager.jersey_input}", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Frame counter
            cv2.putText(annotated_frame, f"Frame: {frame_idx}/{MAX_FRAMES}", (10, height - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Write frame
            out.write(annotated_frame)
        
        # Display frame
        cv2.imshow('Tactical Board System', annotated_frame if not paused else frame)
        
        # Handle input
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
        elif key == ord('a'):
            tactical_manager.assignment_mode = not tactical_manager.assignment_mode
            if not tactical_manager.assignment_mode:
                tactical_manager.selected_tracker_id = None
                tactical_manager.jersey_input = ""
            print(f"Assignment mode: {'ON' if tactical_manager.assignment_mode else 'OFF'}")
        elif key >= ord('1') and key <= ord('2'):
            if tactical_manager.assignment_mode:
                tactical_manager.current_team_id = key - ord('1')
                print(f"Selected team: {tactical_manager.current_team_id}")
        elif key >= ord('0') and key <= ord('9'):
            if tactical_manager.assignment_mode and tactical_manager.selected_tracker_id is not None:
                tactical_manager.jersey_input += chr(key)
                print(f"Jersey input: {tactical_manager.jersey_input}")
        elif key == 13:  # Enter
            if tactical_manager.assignment_mode and tactical_manager.selected_tracker_id is not None and tactical_manager.jersey_input:
                try:
                    jersey_number = int(tactical_manager.jersey_input)
                    tactical_manager.assign_jersey(tactical_manager.selected_tracker_id, jersey_number, 
                                                 tactical_manager.current_team_id)
                    tactical_manager.jersey_input = ""
                    tactical_manager.selected_tracker_id = None
                except ValueError:
                    print("Invalid jersey number")
        elif key == 27:  # ESC
            if tactical_manager.assignment_mode:
                tactical_manager.selected_tracker_id = None
                tactical_manager.jersey_input = ""
                print("Assignment cancelled")
        elif key == ord('f'):
            formation_input = input("Enter formation (e.g., '4-4-2 with players 1,2,3,4...'): ")
            tactical_manager.set_formation_input(formation_input)
        elif key == ord('s'):
            tactical_manager.save_assignments("tactical_board_assignments.json")
        elif key == ord('l'):
            tactical_manager.load_assignments("tactical_board_assignments.json")
        
        # Handle mouse clicks on tactical board
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and tactical_manager.assignment_mode:
                # Check if click is on tactical board area
                board_w = width // 2
                board_h = height // 2
                board_x = width - board_w - 20
                board_y = (height - board_h) // 2
                
                if board_x <= x <= board_x + board_w and board_y <= y <= board_y + board_h:
                    # Convert click to tactical board coordinates
                    rel_x = x - board_x
                    rel_y = y - board_y
                    
                    # Find closest player on tactical board
                    min_distance = float('inf')
                    closest_tracker_id = None
                    
                    if len(all_player_detections) > 0:
                        try:
                            mask = (keypoints.xy[0][:, 0] > 1) & (keypoints.xy[0][:, 1] > 1)
                            if mask.any() and len(keypoints.xy[0]) >= 4:
                                transformer = ViewTransformer(
                                    source=keypoints.xy[0][mask].astype(np.float32),
                                    target=np.array(CONFIG.vertices)[mask].astype(np.float32)
                                )
                                
                                player_xy = all_player_detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
                                transformed_player_xy = transformer.transform_points(points=player_xy)
                                
                                scale_x = board_w / CONFIG.length
                                scale_y = board_h / CONFIG.width
                                
                                for i, tracker_id in enumerate(all_player_detections.tracker_id):
                                    if tracker_id is None:
                                        continue
                                    
                                    tx, ty = transformed_player_xy[i]
                                    board_x_pos = tx * scale_x
                                    board_y_pos = ty * scale_y
                                    
                                    distance = np.sqrt((rel_x - board_x_pos)**2 + (rel_y - board_y_pos)**2)
                                    if distance < min_distance and distance < 20:  # Within 20 pixels
                                        min_distance = distance
                                        closest_tracker_id = tracker_id
                                
                                if closest_tracker_id is not None:
                                    tactical_manager.selected_tracker_id = closest_tracker_id
                                    tactical_manager.jersey_input = ""
                                    print(f"Selected player with tracker ID: {closest_tracker_id}")
                        except Exception as e:
                            print(f"Mouse click error: {e}")
        
        cv2.setMouseCallback('Tactical Board System', mouse_callback)
    
    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    # Save final assignments
    tactical_manager.save_assignments("final_tactical_board_assignments.json")
    print(f"✓ Tactical board analysis complete! Output saved to: {target_video_path}")
    print(f"✓ Assignments saved to: final_tactical_board_assignments.json")

def main():
    parser = argparse.ArgumentParser(description="Tactical Board System")
    parser.add_argument("--source_video_path", type=str, required=True, help="Path to source video")
    parser.add_argument("--target_video_path", type=str, required=True, help="Path to output video")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use (cpu/cuda)")
    
    args = parser.parse_args()
    tactical_board_analysis(args.source_video_path, args.target_video_path, args.device)

if __name__ == "__main__":
    main()
