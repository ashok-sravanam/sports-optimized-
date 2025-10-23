# CRITICAL FIXES - Position Assignment is BROKEN
## Instructions to Fix Incorrect Position Numbers

---

## PROBLEMS IDENTIFIED:

1. ❌ **Referee getting position numbers** (should be ignored completely)
2. ❌ **Multiple players with same number** (111, 111, 111)
3. ❌ **Wrong position assignments** - not matching spatial location rules
4. ❌ **Position numbers changing every frame** - should be PERSISTENT

---

## ROOT CAUSE:

The position assignment is being called **EVERY FRAME** and reassigning positions randomly. This is WRONG.

**CORRECT BEHAVIOR:**
- Assign position number ONCE when player is first detected
- Keep that same number for the entire video
- Use tracker_id to maintain consistency

---

## FIX 1: Assign Position ONLY ONCE (First Detection)

### WRONG CODE (Current):
```python
# This runs EVERY frame - WRONG!
for i in range(len(all_detections)):
    tracker_id = all_detections.tracker_id[i]
    
    # Assigns position EVERY frame - positions keep changing!
    position_num = position_mgr.assign_position(
        tracker_id, classified_team,
        pitch_x=float(pitch_xy[0]),
        pitch_y=float(pitch_xy[1])
    )
```

### CORRECT CODE:
```python
# In PositionBasedAssignmentManager.assign_position():

def assign_position(self, tracker_id: int, classified_team: int,
                   pitch_x: float, pitch_y: float) -> int:
    """
    Assign position number ONCE and keep it forever.
    """
    
    # CHECK IF ALREADY ASSIGNED - CRITICAL!
    if tracker_id in self.tracker_to_position:
        # RETURN EXISTING POSITION - DO NOT REASSIGN!
        return self.tracker_to_position[tracker_id]
    
    # Only assign if this is the FIRST time seeing this tracker_id
    # ... rest of assignment logic ...
```

**KEY POINT:** The `if tracker_id in self.tracker_to_position:` check MUST happen at the very beginning and MUST return immediately. This ensures position is assigned ONLY ONCE.

---

## FIX 2: NEVER Assign Positions to Referees

### WRONG CODE:
```python
for i in range(len(all_detections)):
    tracker_id = all_detections.tracker_id[i]
    classified_team = classified_teams[i]
    
    if classified_team == 2:  # Referee
        continue  # This skip is TOO LATE - referee already in all_detections!
    
    position_num = position_mgr.assign_position(...)
```

### CORRECT CODE:
```python
# Process ONLY players and goalkeepers - EXCLUDE referees completely

for i in range(len(all_detections)):
    tracker_id = all_detections.tracker_id[i]
    classified_team = classified_teams[i]
    
    # SKIP REFEREES COMPLETELY - DO NOT PROCESS AT ALL
    if classified_team == 2:
        continue
    
    # SKIP if not a valid team
    if classified_team not in [0, 1]:
        continue
    
    # Now safe to assign position
    video_xy = all_detections.get_anchors_coordinates(...)[i]
    
    if transformer:
        pitch_xy = transformer.transform_points(...)[0]
        
        position_num = position_mgr.assign_position(
            tracker_id, classified_team,
            pitch_x=float(pitch_xy[0]),
            pitch_y=float(pitch_xy[1])
        )
```

---

## FIX 3: Correct Position Assignment Logic

### The position determination is WRONG. Here's the CORRECT logic:

```python
def _determine_position_from_location(self, norm_x: float, norm_y: float) -> int:
    """
    STRICT position assignment based on pitch coordinates.
    
    Pitch orientation:
    - norm_x: 0.0 (defending/own goal) → 1.0 (attacking/opponent goal)
    - norm_y: 0.0 (left side) → 1.0 (right side)
    
    IMPORTANT: Use STRICT thresholds to avoid overlaps
    """
    
    # === GOALKEEPER ZONE (Very deep, near own goal) ===
    if norm_x < 0.15:
        return 1  # GK - must be very close to own goal
    
    # === DEFENDER ZONE (Deep third) ===
    elif norm_x < 0.30:
        # Left Back - deep, far left
        if norm_y < 0.25:
            return 3  # LB
        
        # Right Back - deep, far right
        elif norm_y > 0.75:
            return 2  # RB
        
        # Center Backs - deep, middle area
        elif norm_y < 0.45:
            return 5  # CB_L (center-left)
        else:
            return 4  # CB_R (center-right)
    
    # === MIDFIELD ZONE (Middle third) ===
    elif norm_x < 0.65:
        # Wide left midfielder
        if norm_y < 0.25:
            return 10  # CM_L or CAM (left side)
        
        # Wide right midfielder
        elif norm_y > 0.75:
            return 8   # CM_R (right side)
        
        # Central midfielder
        else:
            return 6   # CDM (central defensive mid)
    
    # === ATTACKING ZONE (Forward third) ===
    else:
        # Left winger - high up, left side
        if norm_y < 0.25:
            return 7   # LW
        
        # Right winger - high up, right side
        elif norm_y > 0.75:
            return 11  # RW
        
        # Striker - high up, central
        else:
            return 9   # ST
```

**CRITICAL THRESHOLDS:**
- GK: norm_x < 0.15 (must be very deep)
- Defenders: 0.15 ≤ norm_x < 0.30
- Midfielders: 0.30 ≤ norm_x < 0.65
- Attackers: norm_x ≥ 0.65

**CRITICAL Y-AXIS:**
- Left positions: norm_y < 0.25
- Right positions: norm_y > 0.75
- Central: 0.25 ≤ norm_y ≤ 0.75

---

## FIX 4: Handle Position Conflicts

### If position already taken by another player on same team:

```python
def _find_nearest_available(self, norm_x: float, norm_y: float,
                            team_id: int, preferred: int) -> int:
    """
    Find nearest available position if preferred is taken.
    MUST ensure each team has unique positions 1-11.
    """
    
    # Get available positions for this team
    all_positions = set(range(1, 12))  # 1-11
    available = all_positions - self.team_positions_filled[team_id]
    
    if not available:
        # All 11 positions filled - this shouldn't happen!
        # Return a number > 11 as fallback
        print(f"⚠ WARNING: All positions 1-11 filled for team {team_id}")
        return preferred + 100
    
    # Define similar positions (fallback groups)
    fallback_groups = {
        1: [1],           # GK has no fallback
        2: [2, 4],        # RB → CB_R
        3: [3, 5],        # LB → CB_L
        4: [4, 5, 2],     # CB_R → CB_L → RB
        5: [5, 4, 3],     # CB_L → CB_R → LB
        6: [6, 8, 10],    # CDM → CM_R → CM_L
        7: [7, 10],       # LW → CM_L
        8: [8, 6, 10],    # CM_R → CDM → CM_L
        9: [9, 7, 11],    # ST → LW → RW
        10: [10, 8, 6],   # CM_L → CM_R → CDM
        11: [11, 8]       # RW → CM_R
    }
    
    # Try fallback positions in order
    for fallback in fallback_groups.get(preferred, [preferred]):
        if fallback in available:
            print(f"  Position {preferred} taken, using fallback: {fallback}")
            return fallback
    
    # Last resort - return any available position
    return min(available)
```

---

## FIX 5: Ensure Goalkeeper is Always Position 1

### Special handling for goalkeepers:

```python
# In main processing loop:

# Merge detections
all_detections = sv.Detections.merge([players, goalkeepers, referees])

# Create team classification array
classified_teams = np.concatenate([
    players_team_id,
    goalkeepers_team_id,
    np.array([2] * len(referees))  # Referees = team 2
]) if len(all_detections) > 0 else np.array([])

# CRITICAL: Mark which detections are goalkeepers
is_goalkeeper = np.concatenate([
    np.array([False] * len(players)),
    np.array([True] * len(goalkeepers)),
    np.array([False] * len(referees))
]) if len(all_detections) > 0 else np.array([])

# Process each detection
for i in range(len(all_detections)):
    tracker_id = all_detections.tracker_id[i]
    classified_team = classified_teams[i]
    is_gk = is_goalkeeper[i]
    
    # Skip referees
    if classified_team == 2:
        continue
    
    # Skip invalid teams
    if classified_team not in [0, 1]:
        continue
    
    video_xy = all_detections.get_anchors_coordinates(...)[i]
    
    if transformer:
        pitch_xy = transformer.transform_points(...)[0]
        
        # FORCE goalkeeper to position 1
        if is_gk:
            position_num = position_mgr.assign_goalkeeper(tracker_id, classified_team)
        else:
            position_num = position_mgr.assign_position(
                tracker_id, classified_team,
                pitch_x=float(pitch_xy[0]),
                pitch_y=float(pitch_xy[1])
            )
```

### Add this method to PositionBasedAssignmentManager:

```python
def assign_goalkeeper(self, tracker_id: int, classified_team: int) -> int:
    """
    Force assign goalkeeper to position 1.
    """
    
    # Check if already assigned
    if tracker_id in self.tracker_to_position:
        return self.tracker_to_position[tracker_id]
    
    # Get team_id
    if self.db:
        team_id = self.db.get_team_id_from_classification(classified_team)
    else:
        team_id = classified_team
    
    if team_id is None or team_id not in self.team_positions_filled:
        return tracker_id
    
    # Check if position 1 is available
    if 1 not in self.team_positions_filled[team_id]:
        # Assign position 1
        self.tracker_to_position[tracker_id] = 1
        self.team_positions_filled[team_id].add(1)
        
        print(f"✓ Tracker {tracker_id} → Position #1 (GK)")
        
        if self.db:
            self.db.assign_tracker_to_jersey(tracker_id, 1, classified_team)
        
        return 1
    else:
        # Position 1 already taken - this shouldn't happen!
        print(f"⚠ WARNING: Position 1 (GK) already assigned for team {team_id}")
        return 1  # Force it anyway
```

---

## FIX 6: Display Position Numbers Correctly

### On Video (Left Side):

```python
# When annotating video
if len(all_detections) > 0:
    labels = []
    
    for i, tid in enumerate(all_detections.tracker_id):
        classified_team = classified_teams[i]
        
        # Skip referees completely - don't show label
        if classified_team == 2:
            labels.append("")  # Empty label for referee
            continue
        
        # Get position number
        position = position_mgr.get_position(tid)
        
        # Show only position number
        labels.append(f"#{position}")
    
    annotated_video = ellipse_annotator.annotate(annotated_video, all_detections)
    annotated_video = label_annotator.annotate(annotated_video, all_detections, labels)
```

### On Tactical Board (Right Side):

```python
# When drawing tactical board
for i in range(len(detections)):
    tracker_id = detections.tracker_id[i]
    classified_team = classified_teams[i]
    
    # SKIP REFEREES - DO NOT DRAW ON TACTICAL BOARD
    if classified_team == 2:
        continue
    
    # SKIP INVALID TEAMS
    if classified_team not in [0, 1]:
        continue
    
    x, y = board_positions[i]
    
    # Clip to bounds
    x = np.clip(x, 30, board_w - 30)
    y = np.clip(y, 30, board_h - 30)
    
    # Get color
    if classified_team == 0:
        color = team_a_color
    elif classified_team == 1:
        color = team_b_color
    else:
        continue  # Should never reach here
    
    # Get position number
    position = position_mgr.get_position(tracker_id)
    
    # Draw circle with black border
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
```

---

## FIX 7: Debug Logging

### Add logging to understand what's happening:

```python
def assign_position(self, tracker_id: int, classified_team: int,
                   pitch_x: float, pitch_y: float) -> int:
    """Assign position with detailed logging"""
    
    # Check if already assigned
    if tracker_id in self.tracker_to_position:
        existing = self.tracker_to_position[tracker_id]
        # Removed: Don't print every frame
        # print(f"  Tracker {tracker_id} already has position {existing}")
        return existing
    
    # Get team_id
    if self.db:
        team_id = self.db.get_team_id_from_classification(classified_team)
    else:
        team_id = classified_team
    
    if team_id is None or team_id not in self.team_positions_filled:
        print(f"⚠ Invalid team for tracker {tracker_id}: {team_id}")
        return tracker_id
    
    # Normalize coordinates
    norm_x = pitch_x / self.PITCH_LENGTH
    norm_y = pitch_y / self.PITCH_WIDTH
    
    # Log coordinates for first assignment
    print(f"  Assigning tracker {tracker_id} at pitch ({pitch_x:.1f}, {pitch_y:.1f}) "
          f"→ norm ({norm_x:.2f}, {norm_y:.2f})")
    
    # Determine position
    position_num = self._determine_position_from_location(norm_x, norm_y)
    
    print(f"  Initial position determination: {position_num} "
          f"({self.POSITION_NAMES.get(position_num, 'Unknown')})")
    
    # Check if position taken
    if position_num in self.team_positions_filled[team_id]:
        print(f"  Position {position_num} already taken by team {team_id}")
        position_num = self._find_nearest_available(norm_x, norm_y, team_id, position_num)
        print(f"  Using fallback position: {position_num}")
    
    # Assign position
    self.tracker_to_position[tracker_id] = position_num
    self.team_positions_filled[team_id].add(position_num)
    
    pos_name = self.POSITION_NAMES.get(position_num, "Unknown")
    print(f"✓ ASSIGNED: Tracker {tracker_id} → Position #{position_num} ({pos_name})")
    
    return position_num
```

---

## COMPLETE CHECKLIST FOR AGENT:

### MUST FIX:
- [ ] Position assigned ONLY ONCE per tracker_id (check at start of assign_position)
- [ ] Referees COMPLETELY EXCLUDED from position assignment
- [ ] Referees NOT DRAWN on tactical board
- [ ] Referees get empty label "" on video (no position number shown)
- [ ] Goalkeeper ALWAYS gets position 1
- [ ] Position determination uses STRICT thresholds (see FIX 3)
- [ ] Each team has unique positions 1-11 (no duplicates)
- [ ] Fallback logic when position already taken
- [ ] Skip invalid teams (not 0 or 1)
- [ ] Add is_goalkeeper flag to track goalkeeper detections

### MUST NOT:
- [ ] Do NOT reassign position every frame
- [ ] Do NOT assign positions to referees
- [ ] Do NOT allow duplicate positions per team
- [ ] Do NOT show referee labels on video
- [ ] Do NOT draw referees on tactical board

### EXPECTED OUTPUT:
- Team A: Positions 1-11 (unique, no duplicates)
- Team B: Positions 1-11 (unique, no duplicates)
- Referees: No position numbers, not shown on tactical board
- Position 1: Always goalkeeper (deepest player)
- Positions consistent throughout entire video

---

## VALIDATION TEST:

After implementing fixes, verify:

1. **Check CSV output:**
```csv
# Should see positions 1-11 for each team, ball with tracker_id=-1
frame,timestamp,tracker_id,object_type,position,team_id,...
0,0.033,15,player,1,0,...  # GK team 0
0,0.033,23,player,2,0,...  # RB team 0
0,0.033,8,player,3,0,...   # LB team 0
...
0,0.033,42,player,1,1,...  # GK team 1
0,0.033,51,player,2,1,...  # RB team 1
...
0,0.033,-1,ball,NULL,NULL,... # Ball
```

2. **No duplicate positions per team** (each team should have 1-11 only once)

3. **Position 1 is always goalkeeper** (deepest player on pitch)

4. **No referee positions** (referees should not appear in CSV)

5. **Positions stay same** (tracker 15 should be position 1 in all frames)

This is CRITICAL - the current implementation is completely broken!

---

## FIX 8: Confidence Threshold & Position Interpolation

### Problem:
Low confidence detections (<0.65) are unreliable and cause:
- Incorrect position assignments
- Noisy data in CSV
- Players "jumping" around tactical board

### Solution:
**Do NOT store positions with confidence < 0.65**. Instead, interpolate from previous and next valid positions.

---

### Implementation:

#### Step 1: Filter Low Confidence Detections

```python
# In main processing loop:

# After getting all detections and coordinates
for i in range(len(all_detections)):
    tracker_id = all_detections.tracker_id[i]
    classified_team = classified_teams[i]
    
    # Skip referees
    if classified_team == 2:
        continue
    
    # Skip invalid teams
    if classified_team not in [0, 1]:
        continue
    
    # Get confidence
    confidence = float(all_detections.confidence[i]) \
                if hasattr(all_detections, 'confidence') else 1.0
    
    # SKIP LOW CONFIDENCE DETECTIONS - DO NOT STORE
    if confidence < 0.65:
        # Mark for interpolation later
        continue
    
    # Only process high confidence detections
    video_xy = all_detections.get_anchors_coordinates(...)[i]
    
    if transformer:
        pitch_xy = transformer.transform_points(...)[0]
        
        # ... assign position and export to CSV ...
```

---

#### Step 2: Buffer System for Interpolation

```python
class PositionInterpolator:
    """
    Buffers high-confidence positions and interpolates missing frames.
    """
    
    def __init__(self):
        # Buffer format: {tracker_id: [(frame, timestamp, pitch_x, pitch_y, board_x, board_y), ...]}
        self.position_buffer = {}
        
        # Pending low-confidence frames to interpolate
        # Format: {tracker_id: [(frame, timestamp), ...]}
        self.pending_interpolation = {}
    
    def add_high_confidence_position(self, tracker_id: int, frame_idx: int, 
                                     timestamp: float, pitch_x: float, pitch_y: float,
                                     board_x: float, board_y: float):
        """
        Store high-confidence position for interpolation reference.
        """
        if tracker_id not in self.position_buffer:
            self.position_buffer[tracker_id] = []
        
        self.position_buffer[tracker_id].append({
            'frame': frame_idx,
            'timestamp': timestamp,
            'pitch_x': pitch_x,
            'pitch_y': pitch_y,
            'board_x': board_x,
            'board_y': board_y
        })
        
        # Keep only last 60 frames (2 seconds at 30fps) for memory efficiency
        if len(self.position_buffer[tracker_id]) > 60:
            self.position_buffer[tracker_id].pop(0)
    
    def mark_low_confidence(self, tracker_id: int, frame_idx: int, timestamp: float):
        """
        Mark a frame as needing interpolation due to low confidence.
        """
        if tracker_id not in self.pending_interpolation:
            self.pending_interpolation[tracker_id] = []
        
        self.pending_interpolation[tracker_id].append({
            'frame': frame_idx,
            'timestamp': timestamp
        })
    
    def interpolate_positions(self, tracker_id: int, exporter, position: int, 
                             team_id: int, confidence: float = 0.65):
        """
        Interpolate missing positions for low-confidence frames.
        
        Uses linear interpolation between previous and next valid positions.
        """
        if tracker_id not in self.pending_interpolation:
            return
        
        if tracker_id not in self.position_buffer:
            # No reference positions available - skip interpolation
            self.pending_interpolation[tracker_id] = []
            return
        
        buffer = self.position_buffer[tracker_id]
        pending = self.pending_interpolation[tracker_id]
        
        if len(buffer) < 2:
            # Need at least 2 reference points for interpolation
            return
        
        # Process each pending frame
        interpolated_count = 0
        
        for pending_frame in pending:
            frame_idx = pending_frame['frame']
            timestamp = pending_frame['timestamp']
            
            # Find previous and next valid positions
            prev_pos = None
            next_pos = None
            
            for pos in buffer:
                if pos['frame'] < frame_idx:
                    prev_pos = pos
                elif pos['frame'] > frame_idx and next_pos is None:
                    next_pos = pos
                    break
            
            # Interpolate if we have both prev and next
            if prev_pos and next_pos:
                # Calculate interpolation factor
                frame_diff = next_pos['frame'] - prev_pos['frame']
                current_diff = frame_idx - prev_pos['frame']
                t = current_diff / frame_diff  # 0 to 1
                
                # Linear interpolation
                pitch_x = prev_pos['pitch_x'] + t * (next_pos['pitch_x'] - prev_pos['pitch_x'])
                pitch_y = prev_pos['pitch_y'] + t * (next_pos['pitch_y'] - prev_pos['pitch_y'])
                board_x = prev_pos['board_x'] + t * (next_pos['board_x'] - prev_pos['board_x'])
                board_y = prev_pos['board_y'] + t * (next_pos['board_y'] - prev_pos['board_y'])
                
                # Estimate video position (inverse transform not available, use approximation)
                video_x = 0.0  # Placeholder
                video_y = 0.0  # Placeholder
                
                # Export interpolated position
                exporter.add_player(
                    frame_idx, timestamp, tracker_id,
                    position, team_id,
                    video_x, video_y,
                    pitch_x, pitch_y,
                    board_x, board_y,
                    confidence  # Use threshold confidence
                )
                
                interpolated_count += 1
            
            elif prev_pos and not next_pos:
                # Use last known position (extrapolation)
                exporter.add_player(
                    frame_idx, timestamp, tracker_id,
                    position, team_id,
                    0.0, 0.0,  # video coords placeholder
                    prev_pos['pitch_x'], prev_pos['pitch_y'],
                    prev_pos['board_x'], prev_pos['board_y'],
                    confidence
                )
                interpolated_count += 1
        
        if interpolated_count > 0:
            print(f"  ✓ Interpolated {interpolated_count} low-confidence frames for tracker {tracker_id}")
        
        # Clear pending interpolations
        self.pending_interpolation[tracker_id] = []
    
    def finalize_all_interpolations(self, exporter, position_mgr, team_id_map):
        """
        At end of video, interpolate all remaining pending frames.
        """
        print("\nFinalizing position interpolations...")
        
        for tracker_id in self.pending_interpolation.keys():
            if tracker_id in position_mgr.tracker_to_position:
                position = position_mgr.get_position(tracker_id)
                team_id = team_id_map.get(tracker_id)
                
                if team_id is not None:
                    self.interpolate_positions(tracker_id, exporter, position, team_id)
```

---

#### Step 3: Integration in Main Loop

```python
def process_video_final(...):
    """Process video with confidence threshold and interpolation"""
    
    # ... model loading, setup ...
    
    # Initialize interpolator
    interpolator = PositionInterpolator()
    
    # Track team assignments for interpolation
    tracker_to_team = {}  # tracker_id -> team_id
    
    # ... video processing setup ...
    
    with sv.VideoSink(target_path, output_video_info) as sink:
        frame_idx = 0
        
        for frame in tqdm(frame_generator, total=video_info.total_frames):
            
            # ... detection, tracking, classification ...
            
            # Process players
            for i in range(len(all_detections)):
                tracker_id = all_detections.tracker_id[i]
                classified_team = classified_teams[i]
                
                # Skip referees
                if classified_team == 2:
                    continue
                
                # Skip invalid teams
                if classified_team not in [0, 1]:
                    continue
                
                # Get confidence
                confidence = float(all_detections.confidence[i]) \
                            if hasattr(all_detections, 'confidence') else 1.0
                
                video_xy = all_detections.get_anchors_coordinates(...)[i]
                
                if transformer:
                    try:
                        pitch_xy = transformer.transform_points(...)[0]
                        
                        # Calculate board coordinates
                        board_xy = pitch_xy.copy()
                        board_xy[0] *= scale_x
                        board_xy[1] *= scale_y
                        
                        # Clip to bounds
                        board_xy[0] = np.clip(board_xy[0], 30, board_w - 30)
                        board_xy[1] = np.clip(board_xy[1], 30, board_h - 30)
                        
                        # Assign position (only once)
                        position_num = position_mgr.assign_position(
                            tracker_id, classified_team,
                            pitch_x=float(pitch_xy[0]),
                            pitch_y=float(pitch_xy[1])
                        )
                        
                        # Store team mapping for interpolation
                        if tracker_id not in tracker_to_team:
                            tracker_to_team[tracker_id] = classified_team
                        
                        # CHECK CONFIDENCE THRESHOLD
                        if confidence >= 0.65:
                            # HIGH CONFIDENCE - Store position
                            exporter.add_player(
                                frame_idx, timestamp_sec, tracker_id,
                                position_num, classified_team,
                                float(video_xy[0]), float(video_xy[1]),
                                float(pitch_xy[0]), float(pitch_xy[1]),
                                float(board_xy[0]), float(board_xy[1]),
                                confidence
                            )
                            
                            # Add to interpolation buffer
                            interpolator.add_high_confidence_position(
                                tracker_id, frame_idx, timestamp_sec,
                                float(pitch_xy[0]), float(pitch_xy[1]),
                                float(board_xy[0]), float(board_xy[1])
                            )
                        else:
                            # LOW CONFIDENCE - Mark for interpolation
                            interpolator.mark_low_confidence(
                                tracker_id, frame_idx, timestamp_sec
                            )
                        
                        # Interpolate every 30 frames (1 second)
                        if frame_idx % 30 == 0 and frame_idx > 0:
                            interpolator.interpolate_positions(
                                tracker_id, exporter, position_num, 
                                classified_team, confidence=0.65
                            )
                    
                    except Exception as e:
                        print(f"Position processing error: {e}")
            
            # ... ball tracking ...
            # ... render video and tactical board ...
            
            frame_idx += 1
    
    # Finalize all interpolations at end of video
    interpolator.finalize_all_interpolations(exporter, position_mgr, tracker_to_team)
    
    # Export all data
    exporter.export_all()
```

---

#### Alternative: Simple Approach (No Interpolation)

If interpolation is too complex, just skip low-confidence frames:

```python
# Simple approach - just filter out low confidence
for i in range(len(all_detections)):
    # ... get tracker_id, classified_team, etc ...
    
    confidence = float(all_detections.confidence[i]) \
                if hasattr(all_detections, 'confidence') else 1.0
    
    # SKIP LOW CONFIDENCE - DO NOT STORE AT ALL
    if confidence < 0.65:
        continue
    
    # Only process and store high-confidence detections
    # ... rest of position assignment and export ...
```

---

### Configuration

Add confidence threshold as command-line argument:

```python
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_video_path", type=str, required=True)
    parser.add_argument("--target_video_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    
    # Confidence threshold
    parser.add_argument("--confidence_threshold", type=float, default=0.65,
                       help="Minimum detection confidence (default: 0.65)")
    
    # Interpolation mode
    parser.add_argument("--interpolate", action="store_true",
                       help="Enable position interpolation for low-confidence frames")
    
    args = parser.parse_args()
    
    # Pass to processing function
    process_video_final(
        source_path=args.source_video_path,
        target_path=args.target_video_path,
        device=args.device,
        confidence_threshold=args.confidence_threshold,
        enable_interpolation=args.interpolate
    )
```

---

### Usage:

```bash
# Simple - skip low confidence frames
python soccer_analysis.py \
    --source_video_path input.mp4 \
    --target_video_path output.mp4 \
    --confidence_threshold 0.65

# With interpolation
python soccer_analysis.py \
    --source_video_path input.mp4 \
    --target_video_path output.mp4 \
    --confidence_threshold 0.65 \
    --interpolate
```

---

### Expected Behavior:

**Without Interpolation:**
- Detections with confidence < 0.65 are completely skipped
- CSV may have gaps (missing frames for some players)
- Cleaner data, no noise

**With Interpolation:**
- Low confidence frames are filled in using linear interpolation
- Smooth trajectories in CSV (no gaps)
- More complete data, but requires extra computation

---

### Validation:

Check CSV output - all confidence values should be >= 0.65:

```csv
frame,timestamp,tracker_id,object_type,position,team_id,...,confidence
0,0.033,15,player,1,0,...,0.920  ✓ > 0.65
1,0.067,15,player,1,0,...,0.918  ✓ > 0.65
2,0.100,15,player,1,0,...,0.450  ✗ < 0.65 - SHOULD NOT BE IN CSV
3,0.133,15,player,1,0,...,0.650  ✓ = 0.65 (on threshold)
```

---

## UPDATED CHECKLIST:

### MUST ADD:
- [ ] Confidence threshold check (skip if < 0.65)
- [ ] PositionInterpolator class (if using interpolation)
- [ ] Buffer system for high-confidence positions
- [ ] Linear interpolation between frames
- [ ] Command-line arg for confidence_threshold
- [ ] Command-line arg for --interpolate flag
- [ ] Finalize interpolations at end of video

### VALIDATION:
- [ ] All CSV records have confidence >= 0.65
- [ ] No noisy/jittery positions on tactical board
- [ ] Smooth player trajectories (if using interpolation)
- [ ] Gaps in CSV acceptable (if not using interpolation)
