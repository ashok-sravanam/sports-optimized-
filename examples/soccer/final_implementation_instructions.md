# Final Implementation Instructions - Part 5
## Position-Based IDs (1-11), Unified Ball Tracking, and Dynamic Formations

---

## Overview

This document provides the FINAL set of instructions to implement:
1. **Position-based player IDs (1-11)** - Based on spatial location on pitch
2. **Remove player names** - Use only position numbers
3. **Unified tracking table** - Players AND ball in same CSV
4. **Dynamic formations** - User input, not hardcoded

---

## 1. Standard Soccer Position Numbers (1-11)

### FIFA Standard Position Mapping:

```
1  = Goalkeeper (GK)         - Deepest position, central
2  = Right Back (RB)         - Deep, right side of pitch
3  = Left Back (LB)          - Deep, left side of pitch
4  = Center Back Right (CB)  - Deep, center-right
5  = Center Back Left (CB)   - Deep, center-left
6  = Defensive Midfielder (CDM) - Mid-deep, central
7  = Left Winger (LW)        - High up pitch, left side
8  = Central Midfielder Right (CM) - Middle, center-right
9  = Striker (ST/CF)         - Highest position, central
10 = Central Midfielder Left / Attacking Mid (CAM) - Middle, center-left
11 = Right Winger (RW)       - High up pitch, right side
```

### Visual Formation (4-3-3):

```
Attacking End (Opposition Goal)
        ↑
        |
        7 (LW)      9 (ST)      11 (RW)
                    |
                10 (CAM)
        |           |           |
    6 (CDM)     8 (CM)
        |           |           |
    3 (LB)  5 (CB)  4 (CB)  2 (RB)
                    |
                 1 (GK)
        |
        ↓
Defensive End (Own Goal)
```

### Pitch Coordinate System:

```
X-axis: 0m (Own Goal) ────────────> 105m (Opposition Goal)
Y-axis: 0m (Left)     ────────────> 68m (Right)

Example positions (4-3-3 formation):
1 (GK):   (10, 34)   - Near own goal, centered
2 (RB):   (25, 58)   - Defensive, right side
3 (LB):   (25, 10)   - Defensive, left side
4 (CB_R): (22, 40)   - Defensive, center-right
5 (CB_L): (22, 28)   - Defensive, center-left
6 (CDM):  (45, 34)   - Midfield, central
7 (LW):   (80, 12)   - Attack, left wing
8 (CM_R): (50, 45)   - Midfield, center-right
9 (ST):   (90, 34)   - Attack, central striker
10 (CAM): (55, 30)   - Midfield, center-left
11 (RW):  (80, 56)   - Attack, right wing
```

---

## 2. Position-Based Assignment Manager

### Complete Implementation:

```python
import numpy as np
from typing import Dict, Set, Optional

class PositionBasedAssignmentManager:
    """
    Assign position numbers 1-11 based on player's spatial position on pitch.
    Uses pitch coordinates to determine which position each player occupies.
    """
    
    def __init__(self, db_manager):
        """
        Args:
            db_manager: Database manager instance (optional, can be None)
        """
        self.db = db_manager
        
        # tracker_id -> position_number (1-11)
        self.tracker_to_position = {}
        
        # Track which positions are filled per team
        if db_manager:
            self.team_positions_filled = {
                db_manager.team_a_id: set(),
                db_manager.team_b_id: set()
            }
        else:
            # Fallback for mock/testing
            self.team_positions_filled = {
                0: set(),
                1: set()
            }
        
        # Position names for logging
        self.POSITION_NAMES = {
            1: "GK",
            2: "RB",
            3: "LB",
            4: "CB_R",
            5: "CB_L",
            6: "CDM",
            7: "LW",
            8: "CM_R",
            9: "ST",
            10: "CM_L/CAM",
            11: "RW"
        }
        
        # Standard pitch dimensions (FIFA)
        self.PITCH_LENGTH = 105.0  # meters
        self.PITCH_WIDTH = 68.0    # meters
    
    def assign_position(self, tracker_id: int, classified_team: int,
                       pitch_x: float, pitch_y: float) -> int:
        """
        Assign position number (1-11) based on player's location on pitch.
        
        Args:
            tracker_id: ByteTrack tracker ID (can be any integer)
            classified_team: Team classification (0 or 1)
            pitch_x: X coordinate on pitch (0-105m, 0=own goal, 105=opp goal)
            pitch_y: Y coordinate on pitch (0-68m, 0=left, 68=right)
        
        Returns:
            Position number (1-11)
        """
        
        # If already assigned, return existing position
        if tracker_id in self.tracker_to_position:
            return self.tracker_to_position[tracker_id]
        
        # Get actual team_id
        if self.db:
            team_id = self.db.get_team_id_from_classification(classified_team)
        else:
            team_id = classified_team
        
        if team_id is None or team_id not in self.team_positions_filled:
            # Referee or unknown team
            return tracker_id
        
        # Normalize coordinates to 0-1 range
        norm_x = pitch_x / self.PITCH_LENGTH  # 0 = defensive, 1 = attacking
        norm_y = pitch_y / self.PITCH_WIDTH   # 0 = left, 1 = right
        
        # Determine position based on spatial location
        position_num = self._determine_position_from_location(norm_x, norm_y)
        
        # Check if position already filled
        if position_num in self.team_positions_filled[team_id]:
            # Find nearest available position
            position_num = self._find_nearest_available(
                norm_x, norm_y, team_id, position_num
            )
        
        # Assign position
        self.tracker_to_position[tracker_id] = position_num
        self.team_positions_filled[team_id].add(position_num)
        
        # Log assignment
        pos_name = self.POSITION_NAMES.get(position_num, "Unknown")
        print(f"✓ Tracker {tracker_id} → Position #{position_num} ({pos_name}) "
              f"at pitch ({pitch_x:.1f}, {pitch_y:.1f})")
        
        # Register with database if available
        if self.db:
            self.db.assign_tracker_to_jersey(tracker_id, position_num, classified_team)
        
        return position_num
    
    def _determine_position_from_location(self, norm_x: float, norm_y: float) -> int:
        """
        Determine position number based on normalized pitch coordinates.
        
        Coordinate system:
        - norm_x: 0 (own goal) to 1 (opponent goal)
        - norm_y: 0 (left touchline) to 1 (right touchline)
        
        Returns:
            Position number (1-11)
        """
        
        # DEFENSIVE THIRD (norm_x < 0.30)
        if norm_x < 0.18:
            # Goalkeeper zone (very deep)
            return 1  # GK
        
        elif norm_x < 0.30:
            # Defender zone
            if norm_y < 0.20:
                return 3  # Left Back (LB)
            elif norm_y > 0.80:
                return 2  # Right Back (RB)
            elif norm_y < 0.45:
                return 5  # Center Back Left (CB_L)
            else:
                return 4  # Center Back Right (CB_R)
        
        # MIDDLE THIRD (0.30 <= norm_x < 0.60)
        elif norm_x < 0.60:
            # Midfield zone
            if norm_y < 0.30:
                return 10  # Left Central Midfielder (CM_L) or CAM
            elif norm_y > 0.70:
                return 8   # Right Central Midfielder (CM_R)
            else:
                return 6   # Defensive/Central Midfielder (CDM)
        
        # ATTACKING THIRD (norm_x >= 0.60)
        else:
            # Attack zone
            if norm_y < 0.25:
                return 7   # Left Winger (LW)
            elif norm_y > 0.75:
                return 11  # Right Winger (RW)
            else:
                return 9   # Striker (ST)
    
    def _find_nearest_available(self, norm_x: float, norm_y: float,
                                team_id: int, preferred: int) -> int:
        """
        Find nearest available position if preferred is already taken.
        
        Args:
            norm_x, norm_y: Normalized pitch coordinates
            team_id: Team ID
            preferred: Preferred position number that's taken
        
        Returns:
            Available position number
        """
        
        # Get available positions
        all_positions = set(range(1, 12))
        available = all_positions - self.team_positions_filled[team_id]
        
        if not available:
            # All positions filled (shouldn't happen with 11 players max)
            # Use tracker_id or increment
            return preferred + 100
        
        # Define position groups for fallback
        position_groups = {
            1: [1],                    # GK (unique)
            2: [2, 4],                 # RB can fall to CB_R
            3: [3, 5],                 # LB can fall to CB_L
            4: [4, 5, 2],              # CB_R can fall to CB_L or RB
            5: [5, 4, 3],              # CB_L can fall to CB_R or LB
            6: [6, 8, 10],             # CDM can fall to CM
            7: [7, 10, 3],             # LW can fall to CM_L or LB
            8: [8, 6, 10],             # CM_R can fall to CDM or CM_L
            9: [9, 7, 11],             # ST can fall to wingers
            10: [10, 8, 6],            # CM_L can fall to CM_R or CDM
            11: [11, 8, 2]             # RW can fall to CM_R or RB
        }
        
        # Try positions in order of preference
        for fallback in position_groups.get(preferred, [preferred]):
            if fallback in available:
                return fallback
        
        # Last resort: return any available
        return min(available)
    
    def get_position(self, tracker_id: int) -> int:
        """Get position number for tracker_id"""
        return self.tracker_to_position.get(tracker_id, tracker_id)
    
    def get_position_name(self, tracker_id: int) -> str:
        """Get position name (e.g., 'GK', 'ST', 'CM_R')"""
        position = self.get_position(tracker_id)
        return self.POSITION_NAMES.get(position, f"P{position}")
    
    def reset_team_positions(self, team_id: int):
        """Reset position assignments for a team (e.g., after substitution)"""
        if team_id in self.team_positions_filled:
            self.team_positions_filled[team_id].clear()
            print(f"✓ Reset positions for team {team_id}")
```

---

## 3. Unified Tracking Data Exporter

### Single Table for Players AND Ball:

```python
import csv
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

class UnifiedTrackingExporter:
    """
    Export player AND ball tracking data to a single unified CSV table.
    NO player names, only position numbers (1-11).
    """
    
    def __init__(self, output_dir: str = "tracking_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Generate timestamp for this session
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Output files
        self.tracking_csv = self.output_dir / f"tracking_{self.timestamp}.csv"
        self.summary_json = self.output_dir / f"summary_{self.timestamp}.json"
        self.stats_txt = self.output_dir / f"stats_{self.timestamp}.txt"
        
        # Data buffer (holds all tracking records)
        self.tracking_buffer = []
        
        print(f"✓ Unified tracking export initialized: {self.output_dir}")
    
    def add_player(self, frame_idx: int, timestamp: float, tracker_id: int,
                   position: int, team_id: int,
                   video_x: float, video_y: float,
                   pitch_x: float, pitch_y: float,
                   board_x: float, board_y: float,
                   confidence: float):
        """
        Add player tracking record.
        
        Args:
            frame_idx: Frame number
            timestamp: Time in seconds
            tracker_id: ByteTrack ID (for internal tracking)
            position: Position number (1-11)
            team_id: Team database ID
            video_x, video_y: Coordinates in video frame
            pitch_x, pitch_y: Homography-transformed pitch coordinates (meters)
            board_x, board_y: Tactical board display coordinates (pixels)
            confidence: Detection confidence (0-1)
        """
        
        self.tracking_buffer.append({
            'frame': frame_idx,
            'timestamp': round(timestamp, 3),
            'tracker_id': tracker_id,
            'object_type': 'player',
            'position': position,
            'team_id': team_id,
            'video_x': round(video_x, 2),
            'video_y': round(video_y, 2),
            'pitch_x': round(pitch_x, 2),
            'pitch_y': round(pitch_y, 2),
            'board_x': round(board_x, 2),
            'board_y': round(board_y, 2),
            'confidence': round(confidence, 3)
        })
    
    def add_ball(self, frame_idx: int, timestamp: float,
                 video_x: float, video_y: float,
                 pitch_x: float, pitch_y: float,
                 board_x: float, board_y: float,
                 confidence: float):
        """
        Add ball tracking record.
        
        Args:
            frame_idx: Frame number
            timestamp: Time in seconds
            video_x, video_y: Coordinates in video frame
            pitch_x, pitch_y: Homography-transformed pitch coordinates (meters)
            board_x, board_y: Tactical board display coordinates (pixels)
            confidence: Detection confidence (0-1)
        """
        
        self.tracking_buffer.append({
            'frame': frame_idx,
            'timestamp': round(timestamp, 3),
            'tracker_id': -1,  # Special ID for ball
            'object_type': 'ball',
            'position': None,  # Ball has no position number
            'team_id': None,   # Ball has no team
            'video_x': round(video_x, 2),
            'video_y': round(video_y, 2),
            'pitch_x': round(pitch_x, 2),
            'pitch_y': round(pitch_y, 2),
            'board_x': round(board_x, 2),
            'board_y': round(board_y, 2),
            'confidence': round(confidence, 3)
        })
    
    def write_csv(self):
        """Write unified tracking data to CSV file."""
        if not self.tracking_buffer:
            print("⚠ No tracking data to export")
            return
        
        with open(self.tracking_csv, 'w', newline='') as f:
            fieldnames = [
                'frame', 'timestamp', 'tracker_id', 'object_type',
                'position', 'team_id', 'video_x', 'video_y',
                'pitch_x', 'pitch_y', 'board_x', 'board_y', 'confidence'
            ]
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.tracking_buffer)
        
        print(f"✓ Exported {len(self.tracking_buffer)} records to: {self.tracking_csv}")
    
    def write_summary(self):
        """Write summary statistics to JSON."""
        if not self.tracking_buffer:
            return
        
        df = pd.DataFrame(self.tracking_buffer)
        
        summary = {
            'session_timestamp': self.timestamp,
            'total_records': len(df),
            'total_frames': int(df['frame'].max()) if len(df) > 0 else 0,
            'duration_seconds': float(df['timestamp'].max()) if len(df) > 0 else 0.0,
            'players': {},
            'ball': {}
        }
        
        # Player statistics
        player_df = df[df['object_type'] == 'player']
        
        if len(player_df) > 0:
            for tracker_id, group in player_df.groupby('tracker_id'):
                position = int(group.iloc[0]['position'])
                team_id = int(group.iloc[0]['team_id'])
                
                summary['players'][str(tracker_id)] = {
                    'position': position,
                    'team_id': team_id,
                    'frames_tracked': len(group),
                    'avg_pitch_position': {
                        'x': round(float(group['pitch_x'].mean()), 2),
                        'y': round(float(group['pitch_y'].mean()), 2)
                    },
                    'position_range': {
                        'x_min': round(float(group['pitch_x'].min()), 2),
                        'x_max': round(float(group['pitch_x'].max()), 2),
                        'y_min': round(float(group['pitch_y'].min()), 2),
                        'y_max': round(float(group['pitch_y'].max()), 2)
                    },
                    'avg_confidence': round(float(group['confidence'].mean()), 3)
                }
        
        # Ball statistics
        ball_df = df[df['object_type'] == 'ball']
        
        if len(ball_df) > 0:
            summary['ball'] = {
                'frames_tracked': len(ball_df),
                'avg_pitch_position': {
                    'x': round(float(ball_df['pitch_x'].mean()), 2),
                    'y': round(float(ball_df['pitch_y'].mean()), 2)
                },
                'position_range': {
                    'x_min': round(float(ball_df['pitch_x'].min()), 2),
                    'x_max': round(float(ball_df['pitch_x'].max()), 2),
                    'y_min': round(float(ball_df['pitch_y'].min()), 2),
                    'y_max': round(float(ball_df['pitch_y'].max()), 2)
                },
                'avg_confidence': round(float(ball_df['confidence'].mean()), 3)
            }
        
        with open(self.summary_json, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✓ Exported summary to: {self.summary_json}")
    
    def write_stats_report(self):
        """Write human-readable statistics report."""
        if not self.tracking_buffer:
            return
        
        df = pd.DataFrame(self.tracking_buffer)
        
        with open(self.stats_txt, 'w') as f:
            f.write("="*70 + "\n")
            f.write("UNIFIED TRACKING DATA ANALYSIS REPORT\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Session: {self.timestamp}\n")
            f.write(f"Total Records: {len(df)}\n")
            f.write(f"Total Frames: {df['frame'].max()}\n")
            f.write(f"Duration: {df['timestamp'].max():.2f} seconds\n\n")
            
            # Player stats
            player_df = df[df['object_type'] == 'player']
            
            if len(player_df) > 0:
                f.write("-"*70 + "\n")
                f.write("PLAYER STATISTICS\n")
                f.write("-"*70 + "\n\n")
                
                for tracker_id, group in player_df.groupby('tracker_id'):
                    position = int(group.iloc[0]['position'])
                    team_id = int(group.iloc[0]['team_id'])
                    
                    f.write(f"Position #{position} (Team {team_id})\n")
                    f.write(f"  Tracker ID: {tracker_id}\n")
                    f.write(f"  Frames Tracked: {len(group)}\n")
                    f.write(f"  Avg Pitch Position: ({group['pitch_x'].mean():.2f}, "
                           f"{group['pitch_y'].mean():.2f})\n")
                    f.write(f"  Pitch Range: X=[{group['pitch_x'].min():.2f}, "
                           f"{group['pitch_x'].max():.2f}], "
                           f"Y=[{group['pitch_y'].min():.2f}, {group['pitch_y'].max():.2f}]\n")
                    f.write(f"  Avg Confidence: {group['confidence'].mean():.3f}\n\n")
            
            # Ball stats
            ball_df = df[df['object_type'] == 'ball']
            
            if len(ball_df) > 0:
                f.write("-"*70 + "\n")
                f.write("BALL STATISTICS\n")
                f.write("-"*70 + "\n\n")
                
                f.write(f"Frames Tracked: {len(ball_df)}\n")
                f.write(f"Avg Pitch Position: ({ball_df['pitch_x'].mean():.2f}, "
                       f"{ball_df['pitch_y'].mean():.2f})\n")
                f.write(f"Pitch Range: X=[{ball_df['pitch_x'].min():.2f}, "
                       f"{ball_df['pitch_x'].max():.2f}], "
                       f"Y=[{ball_df['pitch_y'].min():.2f}, {ball_df['pitch_y'].max():.2f}]\n")
                f.write(f"Avg Confidence: {ball_df['confidence'].mean():.3f}\n")
        
        print(f"✓ Exported stats report to: {self.stats_txt}")
    
    def export_all(self):
        """Export all data files."""
        self.write_csv()
        self.write_summary()
        self.write_stats_report()
        
        print(f"\n{'='*70}")
        print("EXPORT COMPLETE")
        print(f"{'='*70}")
        print(f"CSV:     {self.tracking_csv}")
        print(f"JSON:    {self.summary_json}")
        print(f"Report:  {self.stats_txt}")
        print(f"{'='*70}\n")
```

---

## 4. Dynamic Formation Manager

### User Input Formations (Not Hardcoded):

```python
class DynamicFormationManager:
    """
    Manage formations with user input or command-line arguments.
    NO hardcoded formations.
    """
    
    def __init__(self):
        self.formations = {
            0: None,  # Team A formation
            1: None   # Team B formation
        }
        self.team_names = {
            0: "Team A",
            1: "Team B"
        }
    
    def set_formation(self, team_idx: int, formation: str, team_name: str = None):
        """
        Set formation for a team.
        
        Args:
            team_idx: 0 or 1
            formation: Formation string (e.g., "4-3-3", "4-4-2", "3-5-2")
            team_name: Optional team name
        """
        self.formations[team_idx] = formation
        
        if team_name:
            self.team_names[team_idx] = team_name
        
        print(f"✓ {self.team_names[team_idx]} formation set: {formation}")
    
    def get_formation(self, team_idx: int) -> str:
        """Get formation for team"""
        return self.formations.get(team_idx, "Unknown")
    
    def get_team_name(self, team_idx: int) -> str:
        """Get team name"""
        return self.team_names.get(team_idx, f"Team {team_idx}")
    
    def prompt_interactive(self):
        """
        Interactive prompt for formations.
        Call this if no formations provided via command-line.
        """
        print("\n" + "="*70)
        print("FORMATION INPUT (or press Enter for defaults)")
        print("="*70)
        
        # Team A
        team_a_name = input("Team A name [Team A]: ").strip()
        if not team_a_name:
            team_a_name = "Team A"
        
        team_a_formation = input(f"{team_a_name} formation [4-3-3]: ").strip()
        if not team_a_formation:
            team_a_formation = "4-3-3"
        
        # Team B
        team_b_name = input("Team B name [Team B]: ").strip()
        if not team_b_name:
            team_b_name = "Team B"
        
        team_b_formation = input(f"{team_b_name} formation [4-4-2]: ").strip()
        if not team_b_formation:
            team_b_formation = "4-4-2"
        
        # Set formations
        self.set_formation(0, team_a_formation, team_a_name)
        self.set_formation(1, team_b_formation, team_b_name)
        
        print("="*70 + "\n")
```

---

## 5. Main Processing Loop Integration

### Complete Implementation:

```python
#!/usr/bin/env python3
"""
Soccer Analysis - Position-Based Tracking (1-11) with Unified Ball Data
"""

import argparse
import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO

from sports.annotators.soccer import draw_pitch
from sports.common.ball import BallTracker, BallAnnotator
from sports.common.team import TeamClassifier
from sports.common.view import ViewTransformer
from sports.configs.soccer import SoccerPitchConfiguration

# Import custom classes (from code above)
from position_assignment import PositionBasedAssignmentManager
from unified_exporter import UnifiedTrackingExporter
from formation_manager import DynamicFormationManager

CONFIG = SoccerPitchConfiguration()

def process_video_final(
    source_path: str,
    target_path: str,
    device: str = 'cpu',
    team_a_formation: str = None,
    team_b_formation: str = None,
    team_a_name: str = None,
    team_b_name: str = None
):
    """
    Final video processing with:
    - Position-based player IDs (1-11)
    - Unified player+ball tracking
    - Dynamic formations
    - No player names
    """
    
    # Setup formations
    formation_mgr = DynamicFormationManager()
    
    if team_a_formation and team_b_formation:
        formation_mgr.set_formation(0, team_a_formation, team_a_name)
        formation_mgr.set_formation(1, team_b_formation, team_b_name)
    else:
        formation_mgr.prompt_interactive()
    
    # Get team names and formations
    team_a_name = formation_mgr.get_team_name(0)
    team_b_name = formation_mgr.get_team_name(1)
    formation_a = formation_mgr.get_formation(0)
    formation_b = formation_mgr.get_formation(1)
    
    print(f"\n{'='*70}")
    print(f"MATCH: {team_a_name} ({formation_a}) vs {team_b_name} ({formation_b})")
    print(f"{'='*70}\n")
    
    # Load models
    print("Loading models...")
    PLAYER_MODEL_PATH = 'data/football-player-detection.pt'
    PITCH_MODEL_PATH = 'data/football-pitch-detection.pt'
    BALL_MODEL_PATH = 'data/football-ball-detection.pt'
    
    player_model = YOLO(PLAYER_MODEL_PATH).to(device=device)
    pitch_model = YOLO(PITCH_MODEL_PATH).to(device=device)
    ball_model = YOLO(BALL_MODEL_PATH).to(device=device)
    
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
    PLAYER_CLASS_ID = 2
    
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
    position_mgr = PositionBasedAssignmentManager(db_manager=None)  # No DB for now
    exporter = UnifiedTrackingExporter(output_dir="tracking_data")
    
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
    
    output_video_info = sv.VideoInfo(
        width=output_w,
        height=output_h,
        fps=video_info.fps,
        total_frames=video_info.total_frames
    )
    
    # Colors
    TEAM_A_COLOR = (255, 100, 100)  # Light blue
    TEAM_B_COLOR = (100, 100, 255)  # Light red
    
    # Annotators
    COLORS = ['#6495ED', '#FF6B6B', '#FFD700']
    ellipse_annotator = sv.EllipseAnnotator(
        color=sv.ColorPalette.from_hex(COLORS), thickness=2
    )
    label_annotator = sv.LabelAnnotator(
        color=sv.ColorPalette.from_hex(COLORS),
        text_color=sv.Color.BLACK,
        text_position=sv.Position.BOTTOM_CENTER
    )
    
    print("Processing video...")
    
    frame_generator = sv.get_video_frames_generator(source_path=source_path)
    
    GOALKEEPER_CLASS_ID = 1
    REFEREE_CLASS_ID = 3
    
    with sv.VideoSink(target_path, output_video_info) as sink:
        frame_idx = 0
        
        for frame in tqdm(frame_generator, total=video_info.total_frames):
            
            # Resize frame
            if frame.shape[0] != video_h or frame.shape[1] != video_w:
                frame = cv2.resize(frame, (video_w, video_h))
            
            timestamp_sec = frame_idx / video_info.fps
            
            # Detection & tracking
            player_result = player_model(frame, imgsz=1280, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(player_result)
            detections = tracker.update_with_detections(detections)
            
            ball_detections = slicer(frame).with_nms(threshold=0.1)
            ball_detections = ball_tracker.update(ball_detections)
            
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
                np.array([2] * len(referees))
            ]) if len(all_detections) > 0 else np.array([])
            
            # Coordinate transformation
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
            
            scale_x = board_w / CONFIG.length
            scale_y = board_h / CONFIG.width
            
            # Process players - POSITION ASSIGNMENT
            for i in range(len(all_detections)):
                tracker_id = all_detections.tracker_id[i]
                classified_team = classified_teams[i]
                
                if classified_team == 2:  # Skip referees
                    continue
                
                video_xy = all_detections.get_anchors_coordinates(
                    anchor=sv.Position.BOTTOM_CENTER
                )[i]
                
                if transformer:
                    try:
                        pitch_xy = transformer.transform_points(
                            points=video_xy.reshape(1, -1)
                        )[0]
                        
                        # ASSIGN POSITION BASED ON LOCATION (1-11)
                        position_num = position_mgr.assign_position(
                            tracker_id, classified_team,
                            pitch_x=float(pitch_xy[0]),
                            pitch_y=float(pitch_xy[1])
                        )
                        
                        # Calculate board coordinates
                        board_xy = pitch_xy.copy()
                        board_xy[0] *= scale_x
                        board_xy[1] *= scale_y
                        
                        # Clip to bounds
                        board_xy[0] = np.clip(board_xy[0], 30, board_w - 30)
                        board_xy[1] = np.clip(board_xy[1], 30, board_h - 30)
                        
                        confidence = float(all_detections.confidence[i]) \
                                    if hasattr(all_detections, 'confidence') else 1.0
                        
                        # Export player data
                        exporter.add_player(
                            frame_idx, timestamp_sec, tracker_id,
                            position_num, classified_team,
                            float(video_xy[0]), float(video_xy[1]),
                            float(pitch_xy[0]), float(pitch_xy[1]),
                            float(board_xy[0]), float(board_xy[1]),
                            confidence
                        )
                    
                    except Exception as e:
                        print(f"Position assignment error: {e}")
            
            # Process ball
            if ball_detections is not None and len(ball_detections) > 0 and transformer:
                try:
                    ball_xy = ball_detections.get_anchors_coordinates(
                        anchor=sv.Position.BOTTOM_CENTER
                    )[0]
                    
                    ball_pitch = transformer.transform_points(
                        points=ball_xy.reshape(1, -1)
                    )[0]
                    
                    ball_board = ball_pitch.copy()
                    ball_board[0] *= scale_x
                    ball_board[1] *= scale_y
                    
                    # Clip ball to bounds
                    ball_board[0] = np.clip(ball_board[0], 30, board_w - 30)
                    ball_board[1] = np.clip(ball_board[1], 30, board_h - 30)
                    
                    ball_confidence = float(ball_detections.confidence[0]) \
                                     if hasattr(ball_detections, 'confidence') else 1.0
                    
                    # Export ball data
                    exporter.add_ball(
                        frame_idx, timestamp_sec,
                        float(ball_xy[0]), float(ball_xy[1]),
                        float(ball_pitch[0]), float(ball_pitch[1]),
                        float(ball_board[0]), float(ball_board[1]),
                        ball_confidence
                    )
                
                except Exception as e:
                    print(f"Ball tracking error: {e}")
            
            # Annotate video (LEFT SIDE)
            annotated_video = frame.copy()
            
            if len(all_detections) > 0:
                # Show ONLY position numbers (no names)
                labels = []
                for tid in all_detections.tracker_id:
                    position = position_mgr.get_position(tid)
                    labels.append(f"#{position}")
                
                annotated_video = ellipse_annotator.annotate(annotated_video, all_detections)
                annotated_video = label_annotator.annotate(annotated_video, all_detections, labels)
            
            annotated_video = ball_annotator.annotate(annotated_video, ball_detections)
            
            # Add title
            cv2.putText(annotated_video, f"{team_a_name} vs {team_b_name}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Draw tactical board (RIGHT SIDE)
            tactical_board = draw_tactical_board_final(
                all_detections, classified_teams, ball_detections,
                keypoints, position_mgr, transformer,
                TEAM_A_COLOR, TEAM_B_COLOR,
                formation_a, formation_b,
                team_a_name, team_b_name,
                board_w, board_h, scale_x, scale_y
            )
            
            # Combine side-by-side
            output_frame = np.hstack([annotated_video, tactical_board])
            
            sink.write_frame(output_frame)
            frame_idx += 1
    
    # Export all data
    exporter.export_all()
    
    print(f"\n✓ Video processing complete: {target_path}")


def draw_tactical_board_final(
    detections, classified_teams, ball_detections,
    keypoints, position_mgr, transformer,
    team_a_color, team_b_color,
    formation_a, formation_b,
    team_a_name, team_b_name,
    board_w, board_h, scale_x, scale_y
):
    """Draw tactical board with position-based numbering"""
    
    pitch = draw_pitch(config=CONFIG)
    tactical_board = cv2.resize(pitch, (board_w, board_h), 
                                interpolation=cv2.INTER_LANCZOS4)
    
    if transformer and len(detections) > 0:
        try:
            player_xy = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
            transformed_xy = transformer.transform_points(points=player_xy)
            
            board_positions = transformed_xy.copy()
            board_positions[:, 0] *= scale_x
            board_positions[:, 1] *= scale_y
            
            # Draw players
            for i in range(len(detections)):
                x, y = board_positions[i]
                
                # Clip to bounds
                x = np.clip(x, 30, board_w - 30)
                y = np.clip(y, 30, board_h - 30)
                
                tracker_id = detections.tracker_id[i]
                classified_team = classified_teams[i]
                
                # Get color
                if classified_team == 0:
                    color = team_a_color
                elif classified_team == 1:
                    color = team_b_color
                else:
                    color = (200, 200, 200)
                
                # Get position number
                position = position_mgr.get_position(tracker_id)
                
                # Draw circle with BLACK border
                radius = 15
                cv2.circle(tactical_board, (int(x), int(y)), radius, color, -1)
                cv2.circle(tactical_board, (int(x), int(y)), radius, (0, 0, 0), 2)
                
                # Draw position number with BLACK text
                text = str(position)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 2
                
                text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                text_x = int(x - text_size[0] / 2)
                text_y = int(y + text_size[1] / 2)
                
                cv2.putText(tactical_board, text, (text_x, text_y),
                           font, font_scale, (0, 0, 0), thickness)
            
            # Draw ball
            if ball_detections is not None and len(ball_detections) > 0:
                ball_xy = ball_detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
                transformed_ball = transformer.transform_points(points=ball_xy)
                
                ball_board = transformed_ball.copy()
                ball_board[:, 0] *= scale_x
                ball_board[:, 1] *= scale_y
                
                for bx, by in ball_board:
                    bx = np.clip(bx, 30, board_w - 30)
                    by = np.clip(by, 30, board_h - 30)
                    
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


def main():
    parser = argparse.ArgumentParser(
        description="Soccer Analysis - Position-Based Tracking"
    )
    parser.add_argument("--source_video_path", type=str, required=True)
    parser.add_argument("--target_video_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    
    # Formation arguments (optional)
    parser.add_argument("--team_a_formation", type=str, default=None,
                       help="Team A formation (e.g., 4-3-3)")
    parser.add_argument("--team_b_formation", type=str, default=None,
                       help="Team B formation (e.g., 4-4-2)")
    parser.add_argument("--team_a_name", type=str, default=None,
                       help="Team A name")
    parser.add_argument("--team_b_name", type=str, default=None,
                       help="Team B name")
    
    args = parser.parse_args()
    
    process_video_final(
        source_path=args.source_video_path,
        target_path=args.target_video_path,
        device=args.device,
        team_a_formation=args.team_a_formation,
        team_b_formation=args.team_b_formation,
        team_a_name=args.team_a_name,
        team_b_name=args.team_b_name
    )


if __name__ == "__main__":
    main()
```

---

## 6. Usage

### With Command-Line Args:

```bash
python soccer_analysis_final.py \
    --source_video_path input.mp4 \
    --target_video_path output.mp4 \
    --device cpu \
    --team_a_name "Real Madrid" \
    --team_a_formation "4-3-3" \
    --team_b_name "Barcelona" \
    --team_b_formation "4-4-2"
```

### Interactive Prompt:

```bash
python soccer_analysis_final.py \
    --source_video_path input.mp4 \
    --target_video_path output.mp4 \
    --device cpu

# Will prompt:
# Team A name [Team A]: Real Madrid
# Real Madrid formation [4-3-3]: 4-3-3
# Team B name [Team B]: Barcelona
# Barcelona formation [4-4-2]: 3-5-2
```

---

## 7. Expected CSV Output

```csv
frame,timestamp,tracker_id,object_type,position,team_id,video_x,video_y,pitch_x,pitch_y,board_x,board_y,confidence
0,0.033,15,player,1,0,450.2,620.5,12.5,34.0,150.0,435.2,0.920
0,0.033,23,player,2,0,1100.5,580.3,85.2,48.5,1022.4,621.4,0.915
0,0.033,8,player,3,0,320.4,560.1,25.3,12.4,303.6,158.8,0.910
0,0.033,12,player,9,0,1250.8,340.2,95.8,32.1,1149.6,411.4,0.905
0,0.033,-1,ball,NULL,NULL,680.3,420.5,55.2,30.8,662.4,394.5,0.950
1,0.067,15,player,1,0,451.1,621.2,12.6,34.1,151.2,436.3,0.918
1,0.067,23,player,2,0,1102.3,581.5,85.4,48.7,1024.8,623.8,0.913
...
```

---

## 8. Summary Checklist

✅ **Position-based IDs (1-11)** - Assigned by spatial location  
✅ **No player names** - Only position numbers  
✅ **Unified table** - Players and ball in same CSV  
✅ **Dynamic formations** - User input, not hardcoded  
✅ **Black text on circles** - Readable on all colors  
✅ **Boundary clipping** - No objects outside pitch  
✅ **Standard soccer positions** - GK=1, RB=2, LB=3, etc.  
✅ **Ball tracking** - tracker_id=-1, object_type='ball'  
✅ **Export statistics** - CSV + JSON + TXT reports

This is the FINAL implementation for the agent to build!
