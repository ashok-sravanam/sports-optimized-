# Soccer Tactical Analysis System

A comprehensive system for tracking soccer players with jersey numbers, performing homography transformations, and storing tactical data in a PostgreSQL database.

## Features

- **Manual Jersey Number Assignment**: Interactive interface to assign jersey numbers to tracked players
- **Homography Transformation**: Convert player positions to tactical board coordinates
- **Database Integration**: Store player positions, events, and formations in PostgreSQL
- **Real-time Tracking**: Track players across video frames with consistent IDs
- **Team Classification**: Automatically classify players into teams
- **Radar Visualization**: Overhead view of player positions on tactical board

## System Architecture

```
Video Input → Player Detection → Tracking → Jersey Assignment → Homography → Database Storage
     ↓              ↓              ↓            ↓              ↓           ↓
  YOLO Models → ByteTrack → Manual UI → ViewTransformer → PostgreSQL → Analysis
```

## Installation

### 1. Install Dependencies

```bash
# Install core sports analysis dependencies
pip install -r requirements.txt

# Install tactical analysis dependencies
pip install -r tactical_requirements.txt
```

### 2. Setup Database

```bash
# Run the setup script
python setup_tactical_system.py
```

This will:
- Install PostgreSQL dependencies
- Create the database schema
- Set up sample teams and players
- Configure database connection

### 3. Database Configuration

The system uses PostgreSQL. Make sure you have:
- PostgreSQL server running
- Database user with appropriate permissions
- Database created (the setup script will create it)

## Usage

### Batch Analysis Mode

Process a video file with automatic jersey assignment:

```bash
python tactical_analysis.py \
  --source_video_path "match_video.mp4" \
  --target_video_path "analysis_output.mp4" \
  --match_id 1 \
  --team_a_id 1 \
  --team_b_id 2
```

### Interactive Analysis Mode

Run with real-time jersey assignment interface:

```bash
python tactical_analysis.py \
  --source_video_path "match_video.mp4" \
  --target_video_path "analysis_output.mp4" \
  --mode interactive
```

## Jersey Assignment Interface

### Controls

- **A** - Toggle assignment mode
- **1-4** - Select team (0-3)
- **Click** - Select player for assignment
- **0-9** - Enter jersey number
- **ENTER** - Assign jersey number
- **ESC** - Cancel assignment
- **S** - Save assignments to file
- **L** - Load assignments from file
- **SPACE** - Pause/Resume (interactive mode)
- **Q** - Quit (interactive mode)

### Assignment Process

1. Press **A** to enter assignment mode
2. Select team using **1-4** keys
3. Click on a player to select them
4. Type jersey number using **0-9** keys
5. Press **ENTER** to assign
6. Repeat for all players
7. Press **S** to save assignments

## Database Schema

### Core Tables

- **teams** - Team information
- **matches** - Match details
- **players** - Player roster with jersey numbers
- **formations** - Tactical formations
- **events** - Match events (goals, cards, etc.)
- **tracked_positions** - Player position data

### Key Relationships

- Players belong to teams with unique jersey numbers
- Positions are tracked by jersey number and team
- Events are linked to players and matches
- Formations define starting positions

## Data Flow

1. **Video Processing**: YOLO models detect players, goalkeepers, referees, and ball
2. **Tracking**: ByteTrack maintains consistent player IDs across frames
3. **Team Classification**: SigLIP model classifies players into teams
4. **Jersey Assignment**: Manual assignment of jersey numbers to tracked players
5. **Homography**: Transform pixel coordinates to tactical board coordinates
6. **Database Storage**: Store positions with jersey numbers and timestamps

## Output Files

- **Analysis Video**: Annotated video with radar view and jersey numbers
- **Jersey Assignments**: JSON file with tracker ID to jersey number mappings
- **Database Records**: Player positions stored in PostgreSQL

## Example Workflow

1. **Setup**: Run `setup_tactical_system.py` to configure database
2. **Analysis**: Run tactical analysis on video file
3. **Assignment**: Assign jersey numbers to tracked players
4. **Storage**: Player positions automatically stored in database
5. **Query**: Use database to analyze player movements and tactics

## Database Queries

### Get Player Positions for Analysis

```sql
SELECT tp.*, p.name, t.team_name
FROM tracked_positions tp
JOIN players p ON tp.jersey_number = p.jersey_number AND tp.team_id = p.team_id
JOIN teams t ON tp.team_id = t.team_id
WHERE tp.match_id = 1
ORDER BY tp.timestamp, tp.jersey_number;
```

### Get Team Formation

```sql
SELECT fp.*, p.name, p.position
FROM formation_positions fp
JOIN players p ON fp.player_id = p.player_id
WHERE fp.formation_id = 1
ORDER BY fp.tactical_position;
```

## Troubleshooting

### Common Issues

1. **Database Connection Failed**
   - Check PostgreSQL is running
   - Verify database credentials
   - Ensure database exists

2. **Jersey Assignment Not Working**
   - Make sure assignment mode is ON (press A)
   - Select team before assigning
   - Click on player bounding box

3. **Radar View Not Showing**
   - Check pitch detection is working
   - Verify homography transformation
   - Ensure keypoints are detected

### Performance Tips

- Use GPU if available (`--device cuda`)
- Process shorter video segments for testing
- Save assignments frequently during long sessions

## Future Enhancements

- **OCR Integration**: Automatic jersey number detection
- **Formation UI**: Drag-drop tactical board interface
- **Event Logging**: Real-time match event input
- **LLM Integration**: AI-powered tactical analysis
- **Multi-camera Support**: Multiple camera angle analysis

## File Structure

```
soccer/
├── tactical_analysis.py          # Main analysis script
├── jersey_assignment.py          # Jersey assignment interface
├── database.py                   # Database operations
├── database_schema.sql           # Database schema
├── setup_tactical_system.py      # Setup script
├── tactical_requirements.txt     # Additional dependencies
└── TACTICAL_ANALYSIS_README.md   # This file
```

## Support

For issues or questions:
1. Check the troubleshooting section
2. Verify database setup
3. Test with sample video files
4. Check console output for error messages
