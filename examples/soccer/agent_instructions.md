# Agent Instructions: Split-Screen Soccer Analysis System

## Objective
Create a NEW Python file that combines player detection/tracking with a clean split-screen tactical board interface. The system should show video analysis on the left and a large, high-quality tactical board on the right.

---

## Core Requirements

### 1. SPLIT-SCREEN LAYOUT (Not Overlay)
**Critical:** Display video and tactical board side-by-side, NOT overlaid

```
┌─────────────────────┬─────────────────────┐
│                     │                     │
│   VIDEO FEED        │   TACTICAL BOARD    │
│   (Left Side)       │   (Right Side)      │
│   - Player tracking │   - Clean pitch     │
│   - Ball tracking   │   - Player circles  │
│   - Team colors     │   - Jersey numbers  │
│                     │                     │
└─────────────────────┴─────────────────────┘
```

**Implementation:**
- Create output frame: `width = video_width + tactical_board_width`
- Video goes in `frame[:, 0:video_width]`
- Tactical board goes in `frame[:, video_width:]`
- Use `np.hstack()` or manual array slicing

---

### 2. VIDEO SIDE (Left) - Based on clean_analysis.py

**Features to Include:**
- Ball tracking with trail visualization
- Player detection with bounding boxes/ellipses
- Player tracking IDs (format: `ID:X` without "D:" prefix)
- Team classification (2 team colors)
- Goalkeeper detection
- Referee detection (optional, can use neutral color)

**Features to Remove:**
- Remove the small radar overlay in top-right corner
- Remove pitch keypoint visualization
- Keep annotations clean and minimal

**Code Structure:**
```python
# 1. Detect players
player_result = player_model(frame, imgsz=1280, verbose=False)[0]
detections = sv.Detections.from_ultralytics(player_result)

# 2. Track players
detections = tracker.update_with_detections(detections)

# 3. Detect & track ball
ball_detections = slicer(frame).with_nms(threshold=0.1)
ball_detections = ball_tracker.update(ball_detections)

# 4. Classify teams
players = detections[detections.class_id == PLAYER_CLASS_ID]
crops = get_crops(frame, players)
players_team_id = team_classifier.predict(crops)

# 5. Annotate video
annotated_frame = ELLIPSE_ANNOTATOR.annotate(frame, detections)
annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(annotated_frame, detections, labels)
annotated_frame = ball_annotator.annotate(annotated_frame, ball_detections)
```

---

### 3. TACTICAL BOARD SIDE (Right) - Enhanced from tactical_board_system.py

#### 3a. Board Specifications

**Size:**
- Large canvas: minimum 800x600 pixels (or half of video width/height)
- Scale to be spacious and clear
- High resolution: use `cv2.resize()` with `cv2.INTER_LANCZOS4` for quality

**Pitch Drawing:**
```python
from sports.annotators.soccer import draw_pitch
from sports.configs.soccer import SoccerPitchConfiguration

CONFIG = SoccerPitchConfiguration()
pitch = draw_pitch(config=CONFIG)

# Resize to tactical board dimensions (high quality)
board_h, board_w = 800, 1200  # Or dynamically calculate
tactical_board = cv2.resize(pitch, (board_w, board_h), interpolation=cv2.INTER_LANCZOS4)
```

**Avoid Pixelation:**
- Use proper interpolation methods
- Don't stretch excessively
- Maintain pitch aspect ratio (~1.5:1)

#### 3b. Player Representation - CRITICAL CHANGES

**Current Problem (from screenshots):**
- Image 1: White circles with "D:17", "D:21" labels are cluttered
- Image 2: Orange/yellow boxes with "ID:X" are too large

**Required Solution:**
```
BEFORE (Wrong):
┌─────────────┐
│   D:17      │  <- Text box, too large
└─────────────┘

AFTER (Correct):
  ●17   <- Small circle with number inside
```

**Implementation:**
```python
# Draw small circles for each player
for i, (x, y) in enumerate(transformed_player_positions):
    tracker_id = player_detections.tracker_id[i]
    team_id = team_ids[i]  # 0 or 1
    
    # Get team color
    color = TEAM_1_COLOR if team_id == 0 else TEAM_2_COLOR
    
    # Get jersey number (or use tracker_id if not assigned)
    jersey_num = manager.get_jersey_number(tracker_id) or tracker_id
    
    # Draw filled circle
    radius = 15  # Small but visible
    cv2.circle(tactical_board, (int(x), int(y)), radius, color, -1)  # Filled
    cv2.circle(tactical_board, (int(x), int(y)), radius, (255, 255, 255), 2)  # White border
    
    # Draw number INSIDE circle (small font)
    text = str(jersey_num)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    thickness = 1
    
    # Center text in circle
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = int(x - text_size[0] / 2)
    text_y = int(y + text_size[1] / 2)
    
    cv2.putText(tactical_board, text, (text_x, text_y), 
                font, font_scale, (255, 255, 255), thickness)
```

**Key Points:**
- **Small circles:** radius 12-15 pixels
- **Two colors only:** Blue for Team 1, Red for Team 2 (example)
- **No boxes, no "D:", no "ID:" prefix**
- **Number inside circle:** white text, small font
- **White border:** makes circles visible on green pitch

#### 3c. Color Scheme

```python
# Define team colors (choose contrasting pairs)
TEAM_1_COLOR = (255, 71, 71)   # Blue (BGR format)
TEAM_2_COLOR = (71, 71, 255)   # Red (BGR format)
GOALKEEPER_COLOR = (71, 255, 71) # Green (optional)
REFEREE_COLOR = (200, 200, 200) # Gray (optional)
BALL_COLOR = (255, 255, 255)    # White

# Or from hex
from supervision import Color
TEAM_1_COLOR = Color.from_hex('#00BFFF').as_bgr()  # Light blue
TEAM_2_COLOR = Color.from_hex('#FF1493').as_bgr()  # Pink/Magenta
```

---

### 4. COORDINATE TRANSFORMATION

**Use ViewTransformer for accurate mapping:**

```python
from sports.common.view import ViewTransformer

# Get pitch keypoints from detection
pitch_result = pitch_model(frame, verbose=False)[0]
keypoints = sv.KeyPoints.from_ultralytics(pitch_result)

# Check for valid keypoints
mask = (keypoints.xy[0][:, 0] > 1) & (keypoints.xy[0][:, 1] > 1)

if mask.any() and mask.sum() >= 4:
    # Create transformer
    transformer = ViewTransformer(
        source=keypoints.xy[0][mask].astype(np.float32),
        target=np.array(CONFIG.vertices)[mask].astype(np.float32)
    )
    
    # Transform player positions from video to pitch coordinates
    player_xy = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
    transformed_xy = transformer.transform_points(points=player_xy)
    
    # Scale to tactical board dimensions
    scale_x = board_w / CONFIG.length  # CONFIG.length ≈ 105m
    scale_y = board_h / CONFIG.width   # CONFIG.width ≈ 68m
    
    board_positions = transformed_xy.copy()
    board_positions[:, 0] *= scale_x
    board_positions[:, 1] *= scale_y
    
    # Now draw circles at board_positions
```

---

### 5. FILE STRUCTURE

**Create: `split_screen_soccer_analysis.py`**

```python
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

# Your imports from sports package
from sports.annotators.soccer import draw_pitch
from sports.common.ball import BallTracker, BallAnnotator
from sports.common.team import TeamClassifier
from sports.common.view import ViewTransformer
from sports.configs.soccer import SoccerPitchConfiguration

# Configuration
PLAYER_DETECTION_MODEL_PATH = 'data/football-player-detection.pt'
PITCH_DETECTION_MODEL_PATH = 'data/football-pitch-detection.pt'
BALL_DETECTION_MODEL_PATH = 'data/football-ball-detection.pt'

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
                tracker_id = detections.tracker_id[i]
                team_id = team_ids[i]
                
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
            
            ball_board = transformed_ball.copy()
            ball_board[:, 0] *= scale_x
            ball_board[:, 1] *= scale_y
            
            for bx, by in ball_board:
                cv2.circle(tactical_board, (int(bx), int(by)), 8, BALL_COLOR, -1)
    
    except Exception as e:
        print(f"Tactical board error: {e}")
    
    return tactical_board

def process_video(source_path: str, target_path: str, device: str = 'cpu'):
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
    
    slicer = sv.InferenceSlicer(callback=ball_callback, slice_wh=(640, 640))
    
    # Team classifier (train on first pass)
    print("Training team classifier...")
    team_classifier = TeamClassifier(device=device)
    
    # Collect crops for training
    STRIDE = 60
    frame_generator = sv.get_video_frames_generator(source_path=source_path, stride=STRIDE)
    crops = []
    
    for frame in frame_generator:
        result = player_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        players = detections[detections.class_id == PLAYER_CLASS_ID]
        crops.extend([sv.crop_image(frame, xyxy) for xyxy in players.xyxy])
        if len(crops) > 500:  # Enough samples
            break
    
    team_classifier.fit(crops)
    
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
        total_frames=video_info.total_frames
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
        for frame in tqdm(frame_generator, total=video_info.total_frames):
            
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
    
    print(f"✓ Processing complete! Output saved to: {target_path}")

def main():
    parser = argparse.ArgumentParser(description="Split-Screen Soccer Analysis")
    parser.add_argument("--source_video_path", type=str, required=True)
    parser.add_argument("--target_video_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    
    args = parser.parse_args()
    process_video(args.source_video_path, args.target_video_path, args.device)

if __name__ == "__main__":
    main()
```

---

## Key Implementation Notes

### Split-Screen Assembly
```python
# Option 1: np.hstack (horizontal stack)
output_frame = np.hstack([video_frame, tactical_board])

# Option 2: Manual slicing
output_frame = np.zeros((height, video_w + board_w, 3), dtype=np.uint8)
output_frame[:, :video_w] = video_frame
output_frame[:, video_w:] = tactical_board
```

### Height Matching
```python
# Ensure both sides have same height
if video_h != board_h:
    if video_h < board_h:
        video_frame = cv2.resize(video_frame, (video_w, board_h))
    else:
        tactical_board = cv2.resize(tactical_board, (board_w, video_h))
```

### Error Handling
```python
try:
    tactical_board = draw_tactical_board(...)
except Exception as e:
    print(f"Tactical board error: {e}")
    # Fallback: draw empty pitch
    tactical_board = draw_pitch(config=CONFIG)
    tactical_board = cv2.resize(tactical_board, (board_w, board_h))
```

---

## Testing Checklist

- [ ] Video side shows player tracking with IDs
- [ ] Video side shows ball tracking with trail
- [ ] Video side shows team colors (2 distinct colors)
- [ ] Tactical board is large and clear (not pixelated)
- [ ] Tactical board shows small circles (not boxes)
- [ ] Circles contain only numbers (no "ID:" prefix)
- [ ] Two team colors are clearly different
- [ ] Ball appears on tactical board (white dot)
- [ ] Both sides are at same height
- [ ] Both sides are displayed side-by-side (not overlaid)
- [ ] No mini-radar overlay on video
- [ ] System runs without crashes

---

## Common Pitfalls to Avoid

1. **Don't overlay tactical board on video** - use side-by-side layout
2. **Don't use text boxes** - use circles with numbers inside
3. **Don't add "ID:" or "D:" prefixes** - just show the number
4. **Don't make circles too large** - keep radius ~15 pixels
5. **Don't stretch tactical board** - maintain aspect ratio
6. **Don't use low-quality resizing** - use `cv2.INTER_LANCZOS4`
7. **Don't skip coordinate transformation** - use ViewTransformer
8. **Don't forget to match heights** - video and board must align

---

## Expected Output

```
INPUT: video.mp4 (1920x1080)

OUTPUT: split_screen_output.mp4 (3120x1080)
├── Left (1920x1080): Player tracking video
└── Right (1200x1080): Clean tactical board
```

The final video should look professional, with clear player positions on both the video and tactical board, using consistent IDs and team colors across both views.
