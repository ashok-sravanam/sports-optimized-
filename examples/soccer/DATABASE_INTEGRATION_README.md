# Database Integration System

This system integrates the split-screen soccer analysis with a PostgreSQL database for comprehensive match tracking, player management, and tactical analysis.

## Overview

The database integration system provides:
- **Real-time position tracking** with database storage
- **Player identification** using jersey numbers and team classification
- **Match context** with teams, formations, and events
- **Tactical analysis** with formation tracking
- **Event logging** for goals, cards, substitutions

## Files

### Core Components
- `soccer_database_manager.py` - Main database manager class
- `enhanced_jersey_manager.py` - Jersey assignment with database integration
- `split_screen_database_analysis.py` - Full analysis system with database
- `database_schema.sql` - PostgreSQL database schema

### Setup & Testing
- `setup_database_sample.py` - Creates sample data for testing
- `test_database_integration.py` - Tests database functionality
- `DATABASE_INTEGRATION_README.md` - This documentation

## Quick Start

### 1. Database Setup

```bash
# Create database
createdb soccer_analysis

# Run schema
psql -d soccer_analysis -f database_schema.sql

# Add sample data
python3 setup_database_sample.py --db_password your_password
```

### 2. Test Database

```bash
# Test database integration
python3 test_database_integration.py --db_password your_password
```

### 3. Run Analysis

```bash
# Run full analysis with database
python3 split_screen_database_analysis.py \
    --source_video_path video_outputs/psgVSliv.mov \
    --target_video_path video_outputs/database_analysis_test.mp4 \
    --match_id 1 \
    --db_password your_password \
    --max_frames 50
```

## Database Schema

### Core Tables

#### `teams`
- `team_id` (PK) - Unique team identifier
- `team_name` - Team name (e.g., "Real Madrid")
- `team_color` - Hex color code (e.g., "#FFFFFF")
- `full_name` - Full team name

#### `matches`
- `match_id` (PK) - Unique match identifier
- `team_a_id` (FK) - Home team
- `team_b_id` (FK) - Away team
- `match_date` - Match date/time
- `venue` - Stadium name
- `competition` - League/competition name

#### `players`
- `player_id` (PK) - Unique player identifier
- `team_id` (FK) - Team the player belongs to
- `jersey_number` - Jersey number (1-11)
- `name` - Player name
- `position` - Playing position (GK, CB, CM, etc.)
- `is_active` - Whether player is active

#### `formations`
- `formation_id` (PK) - Unique formation identifier
- `match_id` (FK) - Match this formation is for
- `team_id` (FK) - Team using this formation
- `formation_type` - Formation name (e.g., "4-3-3")
- `is_active` - Whether formation is currently active

#### `formation_positions`
- `fp_id` (PK) - Unique formation position identifier
- `formation_id` (FK) - Formation this position belongs to
- `player_id` (FK) - Player in this position
- `jersey_number` - Jersey number
- `tactical_position` - Tactical position (GK, RCB, LCM, etc.)

#### `tracked_positions`
- `position_id` (PK) - Unique position identifier
- `frame_id` - Video frame number
- `match_id` (FK) - Match this position is from
- `timestamp` - Time in seconds
- `jersey_number` - Player's jersey number
- `team_id` (FK) - Player's team
- `x` - X coordinate on pitch
- `y` - Y coordinate on pitch
- `confidence` - Detection confidence
- `tracker_id` - ByteTrack tracker ID

#### `events`
- `event_id` (PK) - Unique event identifier
- `match_id` (FK) - Match this event occurred in
- `event_type` - Type of event (GOAL, YELLOW_CARD, etc.)
- `team_id` (FK) - Team involved
- `timestamp` - Time of event (e.g., "45:30")
- `player_id` (FK) - Player involved
- `jersey_number` - Player's jersey number
- `details` - Additional event data (JSONB)

#### `substitutions`
- `substitution_id` (PK) - Unique substitution identifier
- `match_id` (FK) - Match this substitution occurred in
- `team_id` (FK) - Team making substitution
- `player_out` (FK) - Player leaving
- `player_in` (FK) - Player entering
- `jersey_out` - Jersey number of player leaving
- `jersey_in` - Jersey number of player entering
- `timestamp` - Time of substitution

## Usage Examples

### Basic Analysis

```python
from soccer_database_manager import SoccerDatabaseManager
from enhanced_jersey_manager import EnhancedJerseyManager

# Setup database
db_config = {
    'host': 'localhost',
    'database': 'soccer_analysis',
    'user': 'postgres',
    'password': 'your_password',
    'port': 5432
}

db = SoccerDatabaseManager(db_config)
db.connect()
db.setup_match(match_id=1)

# Initialize jersey manager
jersey_manager = EnhancedJerseyManager(db)

# Assign jersey to tracker
jersey_num = jersey_manager.assign_jersey(tracker_id=15, classified_team=0)
print(f"Assigned jersey: {jersey_num}")
```

### Position Tracking

```python
# Insert tracked positions
positions = [
    {
        'frame_id': 100,
        'timestamp': 3.33,
        'jersey_number': 7,
        'team_id': 1,
        'x': 52.5,
        'y': 34.0,
        'confidence': 0.95,
        'tracker_id': 15
    }
]

db.insert_tracked_positions_batch(positions)
```

### Event Logging

```python
# Log a goal
db.insert_event(
    event_type='GOAL',
    team_id=1,
    timestamp='45:30',
    jersey_number=7,
    details={'assist_jersey': 10, 'goal_type': 'open_play'}
)

# Log a substitution
db.insert_substitution(
    team_id=1,
    jersey_out=7,
    jersey_in=9,
    timestamp='60:00'
)
```

## Query Examples

### Player Trajectory

```sql
-- Get player's movement over time
SELECT timestamp, x, y, confidence
FROM tracked_positions
WHERE match_id = 1 
  AND jersey_number = 7 
  AND team_id = 1
ORDER BY timestamp;
```

### Team Heatmap

```sql
-- Get team's average positions
SELECT jersey_number, AVG(x) as avg_x, AVG(y) as avg_y, COUNT(*) as touches
FROM tracked_positions
WHERE match_id = 1 AND team_id = 1
GROUP BY jersey_number;
```

### Match Events

```sql
-- Get all events in a match
SELECT event_type, timestamp, jersey_number, details
FROM events
WHERE match_id = 1
ORDER BY timestamp;
```

### Formation Analysis

```sql
-- Get formation details
SELECT f.formation_type, fp.jersey_number, p.name, fp.tactical_position
FROM formations f
JOIN formation_positions fp ON f.formation_id = fp.formation_id
JOIN players p ON fp.player_id = p.player_id
WHERE f.match_id = 1 AND f.team_id = 1
ORDER BY fp.jersey_number;
```

## Configuration

### Database Connection

```python
db_config = {
    'host': 'localhost',        # Database host
    'database': 'soccer_analysis',  # Database name
    'user': 'postgres',         # Username
    'password': 'your_password', # Password
    'port': 5432               # Port
}
```

### Analysis Parameters

```python
# Video processing
source_path = "input_video.mp4"
target_path = "video_outputs/output.mp4"
match_id = 1
device = "cpu"  # or "cuda", "mps"
max_frames = 100  # or None for all frames
```

## Error Handling

The system includes comprehensive error handling:

- **Database connection errors** - Graceful fallback
- **Missing players** - Warning messages
- **Invalid formations** - Default formations used
- **Position tracking errors** - Batch processing continues
- **Event logging errors** - Non-critical failures

## Performance

### Batch Processing
- Positions are inserted in batches of 330 records
- Reduces database load and improves performance
- Automatic batch flushing at end of processing

### Caching
- Player information is cached in memory
- Team mappings are cached
- Reduces database queries during processing

### Memory Management
- Large video files processed frame by frame
- Position data flushed regularly
- Minimal memory footprint

## Troubleshooting

### Common Issues

1. **Database Connection Failed**
   - Check PostgreSQL is running
   - Verify connection parameters
   - Check firewall settings

2. **Match Not Found**
   - Ensure match exists in database
   - Check match_id is correct
   - Run setup_database_sample.py

3. **Player Not Found**
   - Check player exists in database
   - Verify jersey number assignment
   - Check team_id mapping

4. **Formation Errors**
   - Ensure formations exist for match
   - Check formation_positions data
   - Verify team_id in formations

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Future Enhancements

- **Real-time streaming** - Live match analysis
- **Advanced analytics** - Heatmaps, pass networks
- **Machine learning** - Player performance prediction
- **Web interface** - Browser-based analysis
- **API endpoints** - REST API for data access

## Support

For issues or questions:
1. Check the troubleshooting section
2. Run test_database_integration.py
3. Check database logs
4. Verify sample data setup

## License

This system is part of the sports analysis project and follows the same license terms.
