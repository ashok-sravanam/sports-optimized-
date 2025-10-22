# Soccer Tracking Data - Column Reference Guide

**Generated**: January 21, 2025  
**Version**: 1.0  
**System**: Split-Screen Soccer Analysis with Database Integration

---

## 📊 **Complete Column-by-Column Explanation**

### **🕐 Frame & Time Information**

#### **`frame`**
- **What**: Sequential frame number in the video
- **Range**: 0 to total_frames-1
- **Purpose**: Identifies exactly which video frame the data comes from
- **Example**: `3` = 4th frame of the video (0-indexed)
- **Use Cases**: 
  - Sync data with specific moments in video
  - Calculate frame rate and timing
  - Debug tracking issues at specific frames

#### **`timestamp`**
- **What**: Time elapsed since video start in seconds
- **Range**: 0.0 to video_duration
- **Purpose**: Real-world time reference for analysis
- **Example**: `0.10714285714285714` = ~0.107 seconds into the video
- **Use Cases**:
  - Match events to real time
  - Calculate player speeds and accelerations
  - Create time-based analysis graphs

### **👤 Player Identification**

#### **`tracker_id`**
- **What**: Unique identifier assigned by the tracking algorithm
- **Range**: Integer (varies per detection)
- **Purpose**: Maintains player identity across frames even when they're temporarily lost
- **Example**: `2`, `8`, `9` = Different players detected
- **Use Cases**:
  - Follow individual players throughout the match
  - Handle player re-identification after occlusion
  - Debug tracking consistency

#### **`jersey`**
- **What**: Jersey number assigned to the player (1-11 for each team)
- **Range**: 1-11 (standard soccer jersey numbers)
- **Purpose**: Human-readable player identification
- **Example**: `1` = Goalkeeper, `10` = Star player, etc.
- **Use Cases**:
  - Match players to known team members
  - Generate reports with player names
  - Validate tracking accuracy

#### **`team_id`**
- **What**: Numeric identifier for which team the player belongs to
- **Range**: 1, 2 (in your test: 1=Real Madrid, 2=Barcelona)
- **Purpose**: Distinguish between the two teams
- **Example**: `1` = Home team, `2` = Away team
- **Use Cases**:
  - Separate analysis by team
  - Calculate team formations and tactics
  - Generate team-specific statistics

#### **`player_name`**
- **What**: Full name of the player from the database
- **Range**: String (e.g., "Courtois", "ter Stegen")
- **Purpose**: Human-readable player identification
- **Example**: `"Courtois"` = Real Madrid goalkeeper
- **Use Cases**:
  - Generate readable reports
  - Match to player profiles and statistics
  - Create personalized analysis

### **📍 Position Coordinates (3 Coordinate Systems)**

#### **`video_x`, `video_y`**
- **What**: Raw pixel coordinates in the original video frame
- **Range**: 0 to video_width/height (in pixels)
- **Purpose**: Exact position in the video image
- **Example**: `410.24, 506.64` = pixel position in video
- **Use Cases**:
  - Debug detection accuracy
  - Create video overlays
  - Handle video-specific transformations

#### **`pitch_x`, `pitch_y`**
- **What**: Real-world soccer pitch coordinates in meters/yards
- **Range**: 0 to pitch_length/width (standard pitch dimensions)
- **Purpose**: Standardized coordinates for tactical analysis
- **Example**: `5974.64, 6664.16` = position on real soccer pitch
- **Use Cases**:
  - **Tactical Analysis**: "Player was 25 meters from goal"
  - **Formation Analysis**: Analyze team shapes and positions
  - **Distance Calculations**: Calculate real distances between players
  - **Heat Maps**: Create player movement visualizations
  - **Performance Metrics**: Distance covered, average positions
  - **Cross-Match Comparison**: Compare data across different videos
  - **Professional Analytics**: Feed into coaching software

#### **`board_x`, `board_y`**
- **What**: Coordinates scaled for tactical board visualization
- **Range**: 0 to board_width/height (display dimensions)
- **Purpose**: Optimized coordinates for drawing on tactical board
- **Example**: `597.46, 761.61` = position on tactical board display
- **Use Cases**:
  - Render players on tactical board interface
  - Create real-time tactical visualizations
  - Display formations and player positions

### **🎯 Detection Quality**

#### **`confidence`**
- **What**: Detection confidence score from the YOLO model
- **Range**: 0.0 to 1.0 (higher = more confident)
- **Purpose**: Indicates how reliable the detection is
- **Example**: `0.916814` = 91.68% confidence in the detection
- **Use Cases**:
  - **Quality Control**: Filter out low-confidence detections
  - **Data Validation**: Identify potentially incorrect tracking
  - **Performance Monitoring**: Track detection quality over time
  - **Error Analysis**: Debug when tracking fails

## 🔄 **Coordinate System Relationships**

```
Video Coordinates → Pitch Coordinates → Board Coordinates
     ↓                    ↓                    ↓
Raw pixels in    →  Real-world pitch   →  Display pixels
video frame           measurements          on tactical board
```

## 💡 **Why All Three Coordinate Systems?**

1. **Video Coordinates**: For debugging and video-specific operations
2. **Pitch Coordinates**: For professional analysis and real-world measurements
3. **Board Coordinates**: For user interface and visualization

Each serves a different purpose in the complete soccer analytics pipeline!

## 📁 **File Structure**

```
tracking_data/
├── positions_YYYYMMDD_HHMMSS.csv    # Main tracking data
├── summary_YYYYMMDD_HHMMSS.json     # Statistical summary
├── stats_YYYYMMDD_HHMMSS.txt        # Human-readable report
└── heatmap_positions_YYYYMMDD_HHMMSS.png  # Visualizations
```

## 🎯 **Quick Reference**

| Column | Type | Purpose | Example |
|--------|------|---------|---------|
| frame | int | Video frame number | 3 |
| timestamp | float | Time in seconds | 0.107 |
| tracker_id | int | Unique player ID | 2 |
| jersey | int | Jersey number (1-11) | 1 |
| team_id | int | Team identifier | 1 |
| player_name | string | Player name | "Courtois" |
| video_x | float | Video pixel X | 410.24 |
| video_y | float | Video pixel Y | 506.64 |
| pitch_x | float | Real pitch X (meters) | 5974.64 |
| pitch_y | float | Real pitch Y (meters) | 6664.16 |
| board_x | float | Board display X | 597.46 |
| board_y | float | Board display Y | 761.61 |
| confidence | float | Detection confidence | 0.916 |

---

**System Features Implemented**:
- ✅ Jersey number assignment with database integration
- ✅ Team classification and tracking
- ✅ Real-world coordinate transformation
- ✅ Local data export (CSV/JSON/TXT)
- ✅ Real-time tactical board visualization
- ✅ Boundary clipping for accurate positioning
- ✅ Mock database for testing without PostgreSQL

**Next Steps**: OCR jersey number detection, formation analysis, event logging
