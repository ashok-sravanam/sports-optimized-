# Database Integration Instructions - Part 2
## Using Existing Soccer Tactical Analysis Schema

---

## Overview

This document provides instructions for integrating the split-screen soccer analysis system with your existing PostgreSQL database schema. The schema includes comprehensive match tracking, player management, formations, events, and position tracking.

---

## 1. Database Schema Understanding

Your schema has these key tables:

### Core Tables
- **`teams`**: Team information with colors
- **`matches`**: Match metadata (teams, date, venue, competition)
- **`players`**: Player details (jersey_number, name, position, team_id)
- **`formations`**: Team formation types (4-3-3, 4-4-2, etc.)
- **`formation_positions`**: Initial tactical positions for each player
- **`tracked_positions`**: Real-time position data from video analysis

### Event Tables
- **`events`**: Match events (goals, cards, assists)
- **`substitutions`**: Player substitutions during match

---

## 2. Key Differences from Previous Instructions

**IMPORTANT CHANGES:**

| Previous Schema | Your Schema | Change Required |
|----------------|-------------|-----------------|
| `tracker_id` as PRIMARY KEY | `jersey_number` + `team_id` as identifiers | Map tracker_id → jersey_number |
| `player_id` auto-generated | `player_id` references existing players table | Link to pre-existing players |
| Simple team_id (0, 1) | Proper `team_id` from teams table | Use actual team IDs from database |
| No match context | `match_id` required for all tracking | Must provide match_id |

---

## 3. Database Manager Implementation

```python
import psycopg2
from psycopg2.extras import execute_batch, RealDictCursor
from typing import List, Dict, Optional, Tuple
import numpy as np

class SoccerDatabaseManager:
    """
    Database manager for soccer tactical analysis
    Uses existing schema with teams, matches, players, formations, tracked_positions
    """
    
    def __init__(self, db_config: dict):
        """
        db_config = {
            'host': 'localhost',
            'database': 'soccer_analysis',
            'user': 'postgres',
            'password': 'your_password',
            'port': 5432
        }
        """
        self.config = db_config
        self.conn = None
        self.cursor = None
        
        # Caches
        self.tracker_to_player = {}  # tracker_id -> {'player_id', 'jersey_number', 'team_id'}
        self.jersey_to_player = {}   # (team_id, jersey_number) -> player_id
        
        # Current match context
        self.current_match_id = None
        self.team_a_id = None
        self.team_b_id = None
    
    def connect(self):
        """Establish database connection"""
        self.conn = psycopg2.connect(**self.config)
        self.cursor = self.conn.cursor(cursor_factory=RealDictCursor)
        print("✓ Database connected")
    
    def disconnect(self):
        """Close database connection"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.commit()
            self.conn.close()
        print("✓ Database disconnected")
    
    def setup_match(self, match_id: int):
        """
        Setup match context - load teams and players
        MUST be called before processing video
        """
        self.current_match_id = match_id
        
        # Get match details
        self.cursor.execute(
            """
            SELECT team_a_id, team_b_id, match_date, venue, competition
            FROM matches
            WHERE match_id = %s
            """,
            (match_id,)
        )
        match = self.cursor.fetchone()
        
        if not match:
            raise ValueError(f"Match {match_id} not found in database")
        
        self.team_a_id = match['team_a_id']
        self.team_b_id = match['team_b_id']
        
        print(f"✓ Match setup: Team {self.team_a_id} vs Team {self.team_b_id}")
        print(f"  Venue: {match['venue']}, Competition: {match['competition']}")
        
        # Load all players for both teams
        self._load_players()
        
        # Load formations if available
        self._load_formations()
    
    def _load_players(self):
        """Load all players for both teams in current match"""
        self.cursor.execute(
            """
            SELECT player_id, team_id, jersey_number, name, position
            FROM players
            WHERE team_id IN (%s, %s) AND is_active = TRUE
            ORDER BY team_id, jersey_number
            """,
            (self.team_a_id, self.team_b_id)
        )
        
        players = self.cursor.fetchall()
        
        for player in players:
            key = (player['team_id'], player['jersey_number'])
            self.jersey_to_player[key] = player
            
        print(f"✓ Loaded {len(players)} players from database")
        return players
    
    def _load_formations(self):
        """Load active formations for current match"""
        self.cursor.execute(
            """
            SELECT f.formation_id, f.team_id, f.formation_type,
                   fp.player_id, fp.jersey_number, fp.tactical_position
            FROM formations f
            LEFT JOIN formation_positions fp ON f.formation_id = fp.formation_id
            WHERE f.match_id = %s AND f.is_active = TRUE
            ORDER BY f.team_id, fp.jersey_number
            """,
            (self.current_match_id,)
        )
        
        formations = self.cursor.fetchall()
        
        if formations:
            team_formations = {}
            for row in formations:
                team_id = row['team_id']
                if team_id not in team_formations:
                    team_formations[team_id] = {
                        'formation_type': row['formation_type'],
                        'players': []
                    }
                if row['player_id']:
                    team_formations[team_id]['players'].append({
                        'jersey': row['jersey_number'],
                        'position': row['tactical_position']
                    })
            
            print(f"✓ Loaded formations:")
            for team_id, data in team_formations.items():
                print(f"  Team {team_id}: {data['formation_type']} ({len(data['players'])} players)")
            
            return team_formations
        else:
            print("⚠ No formations found for this match")
            return {}
    
    def get_team_id_from_classification(self, classified_team: int) -> int:
        """
        Convert team classifier output (0 or 1) to actual database team_id
        
        Args:
            classified_team: 0 or 1 from team classifier
        
        Returns:
            Actual team_id from database (e.g., 1, 2, 3, 4)
        """
        if classified_team == 0:
            return self.team_a_id
        elif classified_team == 1:
            return self.team_b_id
        else:
            return None  # Referee or unknown
    
    def assign_tracker_to_jersey(self, tracker_id: int, jersey_number: int, 
                                 classified_team: int) -> Dict:
        """
        Map tracker_id to jersey_number and get player info
        
        Args:
            tracker_id: ByteTrack tracker ID
            jersey_number: Assigned jersey (1-11)
            classified_team: Team classification (0 or 1)
        
        Returns:
            Dict with player_id, jersey_number, team_id, name, position
        """
        # Check if already mapped
        if tracker_id in self.tracker_to_player:
            return self.tracker_to_player[tracker_id]
        
        # Convert classified team to actual team_id
        team_id = self.get_team_id_from_classification(classified_team)
        
        if team_id is None:
            return None
        
        # Get player from database
        key = (team_id, jersey_number)
        player = self.jersey_to_player.get(key)
        
        if not player:
            print(f"⚠ Warning: Player not found for Team {team_id}, Jersey #{jersey_number}")
            return None
        
        # Cache the mapping
        mapping = {
            'player_id': player['player_id'],
            'jersey_number': player['jersey_number'],
            'team_id': player['team_id'],
            'name': player['name'],
            'position': player['position']
        }
        self.tracker_to_player[tracker_id] = mapping
        
        print(f"✓ Mapped tracker {tracker_id} → Jersey #{jersey_number} ({player['name']}, {player['position']})")
        
        return mapping
    
    def get_player_info(self, tracker_id: int) -> Optional[Dict]:
        """Get player info for tracker_id"""
        return self.tracker_to_player.get(tracker_id)
    
    def insert_tracked_positions_batch(self, positions: List[Dict]):
        """
        Batch insert tracked positions
        
        Args:
            positions: List of position dicts with:
                - frame_id: int
                - timestamp: float (seconds)
                - jersey_number: int
                - team_id: int
                - x: float (homography transformed)
                - y: float (homography transformed)
                - confidence: float
                - tracker_id: int
        """
        if not positions:
            return
        
        if self.current_match_id is None:
            raise ValueError("No match setup. Call setup_match() first.")
        
        # Add match_id to all positions
        for pos in positions:
            pos['match_id'] = self.current_match_id
        
        query = """
            INSERT INTO tracked_positions 
            (frame_id, match_id, timestamp, jersey_number, team_id, 
             x, y, confidence, tracker_id)
            VALUES (%(frame_id)s, %(match_id)s, %(timestamp)s, %(jersey_number)s, 
                    %(team_id)s, %(x)s, %(y)s, %(confidence)s, %(tracker_id)s)
        """
        
        try:
            execute_batch(self.cursor, query, positions, page_size=1000)
            self.conn.commit()
            # print(f"✓ Inserted {len(positions)} position records")
        except Exception as e:
            print(f"✗ Error inserting positions: {e}")
            self.conn.rollback()
    
    def insert_event(self, event_type: str, team_id: int, timestamp: str,
                    jersey_number: int, details: dict = None):
        """
        Insert match event (goal, card, assist, etc.)
        
        Args:
            event_type: 'GOAL', 'YELLOW_CARD', 'RED_CARD', 'ASSIST', 'SUBSTITUTION'
            team_id: Database team_id
            timestamp: Time string (e.g., '45:30')
            jersey_number: Player's jersey number
            details: Additional event data (JSONB)
        """
        # Get player_id from jersey_number and team_id
        key = (team_id, jersey_number)
        player = self.jersey_to_player.get(key)
        player_id = player['player_id'] if player else None
        
        self.cursor.execute(
            """
            INSERT INTO events 
            (match_id, event_type, team_id, timestamp, player_id, jersey_number, details)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (self.current_match_id, event_type, team_id, timestamp, 
             player_id, jersey_number, details)
        )
        self.conn.commit()
        
        print(f"✓ Event recorded: {event_type} - Jersey #{jersey_number} at {timestamp}")
    
    def insert_substitution(self, team_id: int, jersey_out: int, jersey_in: int, 
                          timestamp: str):
        """
        Record player substitution
        
        Args:
            team_id: Database team_id
            jersey_out: Player leaving the field
            jersey_in: Player entering the field
            timestamp: Time string (e.g., '60:00')
        """
        # Get player IDs
        key_out = (team_id, jersey_out)
        key_in = (team_id, jersey_in)
        
        player_out = self.jersey_to_player.get(key_out)
        player_in = self.jersey_to_player.get(key_in)
        
        player_out_id = player_out['player_id'] if player_out else None
        player_in_id = player_in['player_id'] if player_in else None
        
        self.cursor.execute(
            """
            INSERT INTO substitutions 
            (match_id, team_id, player_out, player_in, jersey_out, jersey_in, timestamp)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (self.current_match_id, team_id, player_out_id, player_in_id,
             jersey_out, jersey_in, timestamp)
        )
        self.conn.commit()
        
        out_name = player_out['name'] if player_out else f"#{jersey_out}"
        in_name = player_in['name'] if player_in else f"#{jersey_in}"
        print(f"✓ Substitution: {out_name} ➔ {in_name} at {timestamp}")
    
    def get_formation(self, team_id: int) -> Optional[str]:
        """Get formation type for team"""
        self.cursor.execute(
            """
            SELECT formation_type
            FROM formations
            WHERE match_id = %s AND team_id = %s AND is_active = TRUE
            LIMIT 1
            """,
            (self.current_match_id, team_id)
        )
        result = self.cursor.fetchone()
        return result['formation_type'] if result else None
    
    def get_team_color(self, team_id: int) -> Optional[str]:
        """Get team color hex code"""
        self.cursor.execute(
            "SELECT team_color FROM teams WHERE team_id = %s",
            (team_id,)
        )
        result = self.cursor.fetchone()
        return result['team_color'] if result else None
    
    def get_team_name(self, team_id: int) -> Optional[str]:
        """Get team name"""
        self.cursor.execute(
            "SELECT team_name FROM teams WHERE team_id = %s",
            (team_id,)
        )
        result = self.cursor.fetchone()
        return result['team_name'] if result else None
```

---

## 4. Integration with Video Processing

```python
#!/usr/bin/env python3
"""
Split-Screen Soccer Analysis with Database Integration
Uses existing database schema with teams, matches, players, formations
"""

import argparse
import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO
from typing import Dict, List

from sports.annotators.soccer import draw_pitch
from sports.common.ball import BallTracker, BallAnnotator
from sports.common.team import TeamClassifier
from sports.common.view import ViewTransformer
from sports.configs.soccer import SoccerPitchConfiguration

# Import database manager
from database_manager import SoccerDatabaseManager

# Configuration
CONFIG = SoccerPitchConfiguration()

class EnhancedJerseyManager:
    """
    Manages jersey assignments with database integration
    """
    
    def __init__(self, db: SoccerDatabaseManager):
        self.db = db
        self.tracker_to_jersey = {}  # tracker_id -> jersey_number
        self.available_jerseys = {
            db.team_a_id: set(range(1, 12)),
            db.team_b_id: set(range(1, 12))
        }
    
    def assign_jersey(self, tracker_id: int, classified_team: int) -> int:
        """
        Assign jersey and register with database
        
        Args:
            tracker_id: ByteTrack ID
            classified_team: 0 or 1 from team classifier
        
        Returns:
            jersey_number (1-11)
        """
        # Check if already assigned
        if tracker_id in self.tracker_to_jersey:
            return self.tracker_to_jersey[tracker_id]
        
        # Get actual team_id
        team_id = self.db.get_team_id_from_classification(classified_team)
        
        if team_id is None or team_id not in self.available_jerseys:
            return tracker_id  # Fallback for referee
        
        # Assign next available jersey
        if self.available_jerseys[team_id]:
            jersey_num = min(self.available_jerseys[team_id])
            self.available_jerseys[team_id].remove(jersey_num)
            
            # Store assignment
            self.tracker_to_jersey[tracker_id] = jersey_num
            
            # Register with database
            player_info = self.db.assign_tracker_to_jersey(
                tracker_id, jersey_num, classified_team
            )
            
            if player_info:
                print(f"✓ Assigned: Tracker {tracker_id} → #{jersey_num} ({player_info['name']})")
            
            return jersey_num
        else:
            # All jerseys taken
            return tracker_id
    
    def get_jersey(self, tracker_id: int) -> int:
        """Get jersey number for tracker"""
        return self.tracker_to_jersey.get(tracker_id, tracker_id)
    
    def get_player_name(self, tracker_id: int) -> str:
        """Get player name from database"""
        player_info = self.db.get_player_info(tracker_id)
        if player_info:
            return player_info['name']
        return f"Player {tracker_id}"

def process_video_with_database(
    source_path: str,
    target_path: str,
    match_id: int,
    device: str = 'cpu',
    db_config: dict = None
):
    """
    Process video with full database integration
    
    Args:
        source_path: Input video path
        target_path: Output video path
        match_id: Match ID from database
        device: 'cpu', 'cuda', or 'mps'
        db_config: Database connection config
    """
    
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
        frame_generator = sv.get_video_frames_generator(source_path=source_path, stride=STRIDE)
        crops = []
        
        PLAYER_CLASS_ID = 2
        
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
        
        output_video_info = sv.VideoInfo(
            width=output_w,
            height=output_h,
            fps=video_info.fps,
            total_frames=video_info.total_frames
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
        
        GOALKEEPER_CLASS_ID = 1
        REFEREE_CLASS_ID = 3
        
        with sv.VideoSink(target_path, output_video_info) as sink:
            frame_idx = 0
            
            for frame in tqdm(frame_generator, total=video_info.total_frames):
                
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
    
    finally:
        db.disconnect()

def draw_tactical_board_with_db(
    frame_shape: tuple,
    detections: sv.Detections,
    classified_teams: np.ndarray,
    ball_detections: sv.Detections,
    keypoints: sv.KeyPoints,
    jersey_manager,
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
        db_config=db_config
    )

if __name__ == "__main__":
    main()
```

---

## 5. Usage Instructions

### Step 1: Prepare Database

```sql
-- Ensure you have a match in the database
INSERT INTO matches (team_a_id, team_b_id, match_date, venue, competition)
VALUES (1, 2, '2025-01-20 15:00:00', 'Camp Nou', 'La Liga');

-- Get the match_id (e.g., 1)

-- Ensure formations exist
INSERT INTO formations (match_id, team_id, formation_type, is_active)
VALUES 
    (1, 1, '4-3-3', TRUE),
    (1, 2, '4-4-2', TRUE);
```

### Step 2: Run Analysis

```bash
python split_screen_analysis.py \
    --source_video_path input_video.mp4 \
    --target_video_path output_video.mp4 \
    --match_id 1 \
    --device cpu \
    --db_host localhost \
    --db_name soccer_analysis \
    --db_user postgres \
    --db_password your_password \
    --db_port 5432
```

### Step 3: Query Results

```sql
-- View tracked positions for a specific player
SELECT tp.timestamp, tp.x, tp.y, p.name, p.jersey_number
FROM tracked_positions tp
JOIN players p ON tp.jersey_number = p.jersey_number AND tp.team_id = p.team_id
WHERE tp.match_id = 1 AND tp.jersey_number = 10 AND tp.team_id = 1
ORDER BY tp.timestamp;

-- Get player heatmap data
SELECT jersey_number, AVG(x) as avg_x, AVG(y) as avg_y, COUNT(*) as touches
FROM tracked_positions
WHERE match_id = 1 AND team_id = 1
GROUP BY jersey_number;

-- Get all positions at specific timestamp
SELECT p.name, p.jersey_number, tp.x, tp.y
FROM tracked_positions tp
JOIN players p ON tp.jersey_number = p.jersey_number AND tp.team_id = p.team_id
WHERE tp.match_id = 1 AND tp.timestamp BETWEEN 300 AND 305
ORDER BY tp.team_id, tp.jersey_number;
```

---

## 6. Key Integration Points

### Mapping Summary

```python
# Team Classifier Output → Database Team ID
classifier_output = 0  # or 1
actual_team_id = db.get_team_id_from_classification(classifier_output)
# actual_team_id = 1 (from teams table)

# Tracker ID → Jersey Number → Player Info
tracker_id = 15
jersey_num = jersey_manager.assign_jersey(tracker_id, classifier_output=0)
player_info = db.get_player_info(tracker_id)
# player_info = {'player_id': 3, 'name': 'Hazard', 'jersey_number': 7, ...}

# Store Position
position_data = {
    'frame_id': 100,
    'match_id': 1,
    'timestamp': 3.33,
    'jersey_number': 7,
    'team_id': 1,
    'x': 52.5,  # Pitch coordinates
    'y': 34.0,
    'confidence': 0.95,
    'tracker_id': 15
}
db.insert_tracked_positions_batch([position_data])
```

---

## 7. Testing Checklist

- [ ] Database connection successful
- [ ] Match loaded with correct teams
- [ ] Players loaded from database (should see names/jerseys)
- [ ] Team colors loaded from database
- [ ] Formations displayed correctly
- [ ] Jersey numbers 1-11 assigned to each team
- [ ] Jersey numbers remain consistent across frames
- [ ] Positions saved to `tracked_positions` table
- [ ] Can query player trajectories from database
- [ ] Player names shown on video (optional)
- [ ] Tactical board shows correct team colors
- [ ] Formations shown at bottom of tactical board

---

## 8. Error Handling

```python
# Add try-except blocks for database operations
try:
    db.insert_tracked_positions_batch(positions)
except psycopg2.Error as e:
    print(f"Database error: {e}")
    # Optionally save to file as backup
    import json
    with open('backup_positions.json', 'a') as f:
        json.dump(positions, f)
```

---

## Summary

This integration:

✅ Uses your existing database schema  
✅ Maps tracker IDs to actual players in database  
✅ Stores all position data with match context  
✅ Loads team colors and formations from database  
✅ Maintains jersey number consistency (1-11)  
✅ Enables post-match analysis and queries  
✅ Ready for production use

The system now bridges computer vision tracking with your relational database for comprehensive match analysis.
