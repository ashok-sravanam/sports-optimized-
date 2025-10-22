# Bug Fixes & Local Data Visualization - Part 3
## Instructions for UI Improvements and Data Analysis

---

## Issues to Fix

### Issue 1: White Border Making Numbers Illegible
**Problem:** White circle border (stroke) is interfering with white text inside the circle, making jersey numbers hard to read.

**Solution:** Use contrasting text color based on team color, or use black text instead of white.

### Issue 2: Players/Ball Outside Pitch Boundaries
**Problem:** Detection shows players and ball outside the tactical board pitch area.

**Solution:** Add boundary validation and clipping to keep all objects within pitch bounds.

### Issue 3: No Local Data Inspection
**Problem:** Cannot verify if position data is accurate without viewing the database.

**Solution:** Create local CSV/JSON files and add data overlay on video showing coordinates and stats.

---

## 1. Fix Circle Text Readability

### Problem Analysis
```
Current (BAD):
┌─────────────┐
│  ●  10      │  White circle + white border + white text = can't read "10"
└─────────────┘
```

### Solution: Use Black Text with Better Visibility

```python
def draw_player_circle_fixed(
    tactical_board: np.ndarray,
    x: float, 
    y: float,
    jersey_num: int,
    team_color: tuple,  # BGR format
    radius: int = 15
):
    """
    Draw player circle with improved text visibility
    
    Args:
        tactical_board: Board image
        x, y: Circle position
        jersey_num: Jersey number to display
        team_color: Team color in BGR format
        radius: Circle radius (default 15)
    """
    
    # Draw filled circle (team color)
    cv2.circle(tactical_board, (int(x), int(y)), radius, team_color, -1)
    
    # Draw BLACK border instead of white (better contrast)
    cv2.circle(tactical_board, (int(x), int(y)), radius, (0, 0, 0), 2)
    
    # Prepare text
    text = str(jersey_num)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5  # Slightly larger
    thickness = 2      # Thicker for readability
    
    # Calculate text size for centering
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = int(x - text_size[0] / 2)
    text_y = int(y + text_size[1] / 2)
    
    # Use BLACK text (works on all team colors)
    cv2.putText(tactical_board, text, (text_x, text_y),
                font, font_scale, (0, 0, 0), thickness)
    
    return tactical_board

# Alternative: Use team-aware text color
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

# Updated draw function with auto-contrast
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
```

### Update in main drawing function:

```python
# In draw_tactical_board_with_db function:

# OLD CODE (remove this):
# cv2.circle(tactical_board, (int(x), int(y)), radius, color, -1)
# cv2.circle(tactical_board, (int(x), int(y)), radius, (255, 255, 255), 2)
# cv2.putText(tactical_board, text, (text_x, text_y),
#            font, font_scale, (255, 255, 255), thickness)

# NEW CODE (use this):
tactical_board = draw_player_circle_auto_contrast(
    tactical_board, x, y, jersey_num, color, radius=15
)
```

---

## 2. Fix Players/Ball Outside Pitch Boundaries

### Problem
Homography transformation can place objects outside the tactical board when:
- Pitch detection is inaccurate
- Players are at extreme edges
- Transformation matrix is unstable

### Solution: Boundary Clipping

```python
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

# Update drawing code:
def draw_tactical_board_with_db_fixed(
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
    """Draw tactical board with boundary clipping"""
    
    pitch = draw_pitch(config=CONFIG)
    tactical_board = cv2.resize(pitch, (board_w, board_h), 
                                interpolation=cv2.INTER_LANCZOS4)
    
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
    cv2.putText(tactical_board, "TACTICAL BOARD (MOCK DB)", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    return tactical_board
```

---

## 3. Local Data Export & Visualization

### 3.1 Export to CSV

```python
import csv
import json
from datetime import datetime
from pathlib import Path

class LocalDataExporter:
    """Export tracking data to local files for analysis"""
    
    def __init__(self, output_dir: str = "tracking_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Generate timestamp for this run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # File paths
        self.positions_csv = self.output_dir / f"positions_{self.timestamp}.csv"
        self.summary_json = self.output_dir / f"summary_{self.timestamp}.json"
        self.stats_txt = self.output_dir / f"stats_{self.timestamp}.txt"
        
        # Data buffers
        self.position_buffer = []
        self.player_stats = {}  # tracker_id -> stats
        
        print(f"✓ Local data export initialized: {self.output_dir}")
    
    def add_position(self, frame_idx: int, timestamp: float, tracker_id: int,
                    jersey_num: int, team_id: int, player_name: str,
                    video_x: float, video_y: float,
                    pitch_x: float, pitch_y: float,
                    board_x: float, board_y: float,
                    confidence: float):
        """Add position data point"""
        
        self.position_buffer.append({
            'frame': frame_idx,
            'timestamp': timestamp,
            'tracker_id': tracker_id,
            'jersey': jersey_num,
            'team_id': team_id,
            'player_name': player_name,
            'video_x': video_x,
            'video_y': video_y,
            'pitch_x': pitch_x,
            'pitch_y': pitch_y,
            'board_x': board_x,
            'board_y': board_y,
            'confidence': confidence
        })
        
        # Update stats
        if tracker_id not in self.player_stats:
            self.player_stats[tracker_id] = {
                'jersey': jersey_num,
                'name': player_name,
                'team_id': team_id,
                'total_frames': 0,
                'avg_x': 0,
                'avg_y': 0,
                'positions': []
            }
        
        stats = self.player_stats[tracker_id]
        stats['total_frames'] += 1
        stats['positions'].append((pitch_x, pitch_y))
    
    def write_csv(self):
        """Write positions to CSV file"""
        if not self.position_buffer:
            return
        
        with open(self.positions_csv, 'w', newline='') as f:
            fieldnames = ['frame', 'timestamp', 'tracker_id', 'jersey', 'team_id',
                         'player_name', 'video_x', 'video_y', 'pitch_x', 'pitch_y',
                         'board_x', 'board_y', 'confidence']
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.position_buffer)
        
        print(f"✓ Exported {len(self.position_buffer)} positions to CSV: {self.positions_csv}")
    
    def write_summary(self):
        """Write summary statistics to JSON"""
        
        # Calculate stats
        summary = {
            'timestamp': self.timestamp,
            'total_positions': len(self.position_buffer),
            'total_frames': max([p['frame'] for p in self.position_buffer]) if self.position_buffer else 0,
            'players': {}
        }
        
        for tracker_id, stats in self.player_stats.items():
            positions = np.array(stats['positions'])
            
            player_summary = {
                'jersey': stats['jersey'],
                'name': stats['name'],
                'team_id': stats['team_id'],
                'frames_tracked': stats['total_frames'],
                'avg_position': {
                    'x': float(np.mean(positions[:, 0])),
                    'y': float(np.mean(positions[:, 1]))
                },
                'position_variance': {
                    'x': float(np.var(positions[:, 0])),
                    'y': float(np.var(positions[:, 1]))
                }
            }
            
            summary['players'][str(tracker_id)] = player_summary
        
        with open(self.summary_json, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✓ Exported summary to JSON: {self.summary_json}")
    
    def write_stats_report(self):
        """Write human-readable stats report"""
        
        with open(self.stats_txt, 'w') as f:
            f.write("="*60 + "\n")
            f.write("TRACKING DATA ANALYSIS REPORT\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Generated: {self.timestamp}\n")
            f.write(f"Total Positions Tracked: {len(self.position_buffer)}\n")
            f.write(f"Total Players: {len(self.player_stats)}\n\n")
            
            f.write("-"*60 + "\n")
            f.write("PLAYER STATISTICS\n")
            f.write("-"*60 + "\n\n")
            
            for tracker_id, stats in sorted(self.player_stats.items(), 
                                           key=lambda x: x[1]['jersey']):
                positions = np.array(stats['positions'])
                
                f.write(f"Player: {stats['name']} (Jersey #{stats['jersey']})\n")
                f.write(f"  Team ID: {stats['team_id']}\n")
                f.write(f"  Frames Tracked: {stats['total_frames']}\n")
                f.write(f"  Average Position: ({np.mean(positions[:, 0]):.2f}, "
                       f"{np.mean(positions[:, 1]):.2f})\n")
                f.write(f"  Position Range: X=[{np.min(positions[:, 0]):.2f}, "
                       f"{np.max(positions[:, 0]):.2f}], "
                       f"Y=[{np.min(positions[:, 1]):.2f}, {np.max(positions[:, 1]):.2f}]\n")
                f.write("\n")
        
        print(f"✓ Exported stats report: {self.stats_txt}")
    
    def export_all(self):
        """Export all data files"""
        self.write_csv()
        self.write_summary()
        self.write_stats_report()
```

### 3.2 Add Data Overlay on Video

```python
def draw_data_overlay(
    frame: np.ndarray,
    frame_idx: int,
    timestamp: float,
    detections: sv.Detections,
    jersey_manager,
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
```

### 3.3 Integration in Main Loop

```python
def process_video_with_local_export(
    source_path: str,
    target_path: str,
    match_id: int,
    device: str = 'cpu',
    db_config: dict = None,
    export_local: bool = True
):
    """Process video with local data export"""
    
    # Initialize database
    db = SoccerDatabaseManager(db_config)
    db.connect()
    
    # Initialize local exporter
    exporter = LocalDataExporter(output_dir="tracking_data") if export_local else None
    
    try:
        db.setup_match(match_id)
        
        # ... (model loading, setup, etc.) ...
        
        jersey_manager = EnhancedJerseyManager(db)
        
        # ... (video setup) ...
        
        position_batch = []
        frame_idx = 0
        
        for frame in video:
            timestamp_sec = frame_idx / video_info.fps
            
            # ... (detection, tracking, classification) ...
            
            # Process each player
            for i in range(len(all_detections)):
                tracker_id = all_detections.tracker_id[i]
                classified_team = classified_teams[i]
                
                if classified_team == 2:  # Skip referees
                    continue
                
                jersey_num = jersey_manager.assign_jersey(tracker_id, classified_team)
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
            
            # ... (annotate video, draw tactical board) ...
            
            # Add data overlay to video
            if exporter:
                annotated_video = draw_data_overlay(
                    annotated_video, frame_idx, timestamp_sec,
                    all_detections, jersey_manager, db, exporter
                )
            
            # ... (combine frames, write output) ...
            
            frame_idx += 1
        
        # Final database insert
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
    
    finally:
        db.disconnect()
```

---

## 4. Quick Verification Script

Create a separate script to analyze the exported data:

```python
#!/usr/bin/env python3
"""
analyze_tracking_data.py
Quick script to analyze exported tracking data
"""

import pandas as pd
import json
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_latest_tracking_data(data_dir: str = "tracking_data"):
    """Analyze the most recent tracking data export"""
    
    data_path = Path(data_dir)
    
    # Find latest CSV
    csv_files = list(data_path.glob("positions_*.csv"))
    if not csv_files:
        print("No tracking data found!")
        return
    
    latest_csv = max(csv_files, key=lambda p: p.stat().st_mtime)
    print(f"Analyzing: {latest_csv}")
    
    # Load data
    df = pd.read_csv(latest_csv)
    
    print(f"\n{'='*60}")
    print("TRACKING DATA ANALYSIS")
    print(f"{'='*60}\n")
    
    # Basic stats
    print(f"Total records: {len(df)}")
    print(f"Total frames: {df['frame'].max()}")
    print(f"Duration: {df['timestamp'].max():.2f} seconds")
    print(f"Unique players: {df['tracker_id'].nunique()}")
    print(f"Teams: {df['team_id'].unique()}")
    
    # Per-player stats
    print(f"\n{'-'*60}")
    print("PER-PLAYER STATISTICS")
    print(f"{'-'*60}\n")
    
    for _, player_df in df.groupby('tracker_id'):
        player_name = player_df.iloc[0]['player_name']
        jersey = player_df.iloc[0]['jersey']
        team = player_df.iloc[0]['team_id']
        
        print(f"Player: {player_name} (Jersey #{jersey}, Team {team})")
        print(f"  Frames tracked: {len(player_df)}")
        print(f"  Avg position: ({player_df['pitch_x'].mean():.2f}, {player_df['pitch_y'].mean():.2f})")
        print(f"  Avg confidence: {player_df['confidence'].mean():.3f}")
        print()
    
    # Plot heatmaps
    print("Generating visualizations...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Team 1
    team1_df = df[df['team_id'] == df['team_id'].unique()[0]]
    axes[0].scatter(team1_df['pitch_x'], team1_df['pitch_y'], 
                   c='blue', alpha=0.5, s=10)
    axes[0].set_title(f"Team {team1_df['team_id'].iloc[0]} Heatmap")
    axes[0].set_xlabel("Pitch X")
    axes[0].set_ylabel("Pitch Y")
    axes[0].grid(True)
    
    # Team 2
    if len(df['team_id'].unique()) > 1:
        team2_df = df[df['team_id'] == df['team_id'].unique()[1]]
        axes[1].scatter(team2_df['pitch_x'], team2_df['pitch_y'], 
                       c='red', alpha=0.5, s=10)
        axes[1].set_title(f"Team {team2_df['team_id'].iloc[0]} Heatmap")
        axes[1].set_xlabel("Pitch X")
        axes[1].set_ylabel("Pitch Y")
        axes[1].grid(True)
    
    plt.tight_layout()
    
    output_plot = data_path / f"heatmap_{latest_csv.stem}.png"
    plt.savefig(output_plot, dpi=150)
    print(f"✓ Heatmap saved: {output_plot}")
    
    plt.show()

if __name__ == "__main__":
    analyze_latest_tracking_data()
```

---

## 5. Complete Update Checklist

### Visual Fixes
- [ ] Change circle border from white to black
- [ ] Use black text or auto-contrast text for jersey numbers
- [ ] Increase font scale to 0.5 and thickness to 2
- [ ] Add boundary clipping for all player positions
- [ ] Add boundary clipping for ball position
- [ ] Set margin to 30 pixels from edge

### Local Data Export
- [ ] Create LocalDataExporter class
- [ ] Export positions to CSV file
- [ ] Export summary to JSON file
- [ ] Export stats report to TXT file
- [ ] Add data overlay panel on video (bottom-left)
- [ ] Show frame number, timestamp, player count
- [ ] Show sample positions for first 3 players
- [ ] Show total exported records counter

### Data Verification
- [ ] Create analyze_tracking_data.py script
- [ ] Verify CSV contains all expected columns
- [ ] Check player jersey consistency
- [ ] Verify coordinates are within valid ranges
- [ ] Generate heatmap visualizations
- [ ] Compare video output with CSV data

---

## 6. Usage

### Run with local export enabled:

```bash
python split_screen_analysis.py \
    --source_video_path input.mp4 \
    --target_video_path output.mp4 \
    --match_id 1 \
    --device cpu \
    --db_host localhost \
    --db_name soccer_analysis \
    --db_user postgres \
    --db_password your_password \
    --export_local
```

### Analyze exported data:

```bash
python analyze_tracking_data.py
```

This will:
1. Find the latest CSV export
2. Print detailed statistics
3. Generate heatmap visualizations
4. Save plots to PNG files

---

## 7. Expected Output Files

```
tracking_data/
├── positions_20250121_143022.csv          # All position data
├── summary_20250121_143022.json           # Summary statistics
├── stats_20250121_143022.txt              # Human-readable report
└── heatmap_positions_20250121_143022.png  # Visualization
```

### CSV Format:
```csv
frame,timestamp,tracker_id,jersey,team_id,player_name,video_x,video_y,pitch_x,pitch_y,board_x,board_y,confidence
0,0.00,1,7,1,Hazard,850.5,420.3,45.2,30.1,540.2,320.8,0.95
0,0.00,2,10,1,Modric,920.1,380.7,52.3,28.5,628.3,304.5,0.92
...
```

---

## Summary of Changes

✅ **Fixed white text visibility** - Black border + contrasting text  
✅ **Fixed out-of-bounds players** - Boundary clipping with 30px margin  
✅ **Added local CSV export** - All positions saved locally  
✅ **Added JSON summary** - Statistics and aggregations  
✅ **Added TXT report** - Human-readable analysis  
✅ **Added video overlay** - Real-time data panel  
✅ **Added analysis script** - Quick verification tool  
✅ **Added heatmap generation** - Visual position analysis

Now you can verify tracking accuracy by inspecting local files before committing to the database!
