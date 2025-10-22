# Soccer Analysis System - Version 3.0

**Professional soccer tracking and tactical analysis with database integration**

## 🚀 **Quick Start**

```bash
# Clone and setup
git clone https://github.com/ashok-sravanam/sports-optimized-.git
cd sports-optimized/examples/soccer

# Install dependencies
pip3 install -r requirements.txt
pip3 install -r tactical_requirements.txt

# Download models and sample video
chmod +x setup.sh
./setup.sh

# Run analysis
python3 test_bug_fixes.py \
    --source_video_path "video_outputs/psgVSliv.mov" \
    --target_video_path "video_outputs/analysis_output.mp4" \
    --max_frames 500 \
    --device cpu
```

## ✨ **Features**

- **Split-Screen Analysis**: Live video feed + tactical board
- **Player Tracking**: 20+ players with jersey numbers
- **Team Classification**: Automatic team identification
- **Data Export**: CSV/JSON/TXT with complete tracking data
- **Mock Database**: Works without PostgreSQL
- **Real-Time Progress**: Live processing updates
- **Professional Visualization**: Tactical board with formations

## 📁 **File Structure**

```
soccer/
├── 📄 Core Files
│   ├── split_screen_database_analysis.py    # Main analysis system
│   ├── test_bug_fixes.py                    # Easy testing script
│   ├── local_data_exporter.py              # Data export system
│   ├── analyze_tracking_data.py             # Data analysis tool
│   └── soccer_database_manager.py           # Database integration
│
├── 📄 Configuration
│   ├── enhanced_jersey_manager.py          # Jersey assignment
│   ├── setup_database_sample.py            # Database setup
│   ├── requirements.txt                    # Python dependencies
│   ├── tactical_requirements.txt           # Additional dependencies
│   └── setup.sh                           # Model download script
│
├── 📄 Documentation
│   ├── README.md                           # This file
│   ├── DATABASE_INTEGRATION_README.md      # Database guide
│   ├── TACTICAL_ANALYSIS_README.md        # Tactical analysis guide
│   └── database_schema.sql                # Database schema
│
├── 📁 data/                                # Models and sample videos
│   ├── football-player-detection.pt
│   ├── football-pitch-detection.pt
│   ├── football-ball-detection.pt
│   └── psgVSliv.mov
│
├── 📁 tracking_data/                       # Exported data
│   └── TRACKING_DATA_COLUMNS_README.md    # Data documentation
│
├── 📁 video_outputs/                       # Output videos
├── 📁 misc/                                # Legacy files
└── 📁 notebooks/                           # Training notebooks
```

## 🎯 **Output Files**

After running analysis, you'll get:

- **📹 Video**: `video_outputs/analysis_output.mp4` (split-screen)
- **📊 Data**: `tracking_data/positions_YYYYMMDD_HHMMSS.csv` (10,000+ records)
- **📈 Stats**: `tracking_data/summary_YYYYMMDD_HHMMSS.json` (statistics)
- **📋 Report**: `tracking_data/stats_YYYYMMDD_HHMMSS.txt` (human-readable)
- **📖 Docs**: `tracking_data/TRACKING_DATA_COLUMNS_README.md` (data guide)

## 🔧 **Requirements**

- **Python 3.8+**
- **4GB+ RAM**
- **~2GB disk space**
- **macOS/Linux/Windows**

## 📊 **Data Columns**

| Column | Description | Example |
|--------|-------------|---------|
| frame | Video frame number | 123 |
| timestamp | Time in seconds | 4.39 |
| tracker_id | Unique player ID | 2 |
| jersey | Jersey number (1-11) | 1 |
| team_id | Team identifier | 1 |
| player_name | Player name | "Courtois" |
| video_x, video_y | Video pixel coordinates | 410, 506 |
| pitch_x, pitch_y | Real-world coordinates | 5974, 6664 |
| board_x, board_y | Tactical board coordinates | 597, 761 |
| confidence | Detection confidence | 0.916 |

## 🎮 **Usage Examples**

```bash
# Quick test (50 frames)
python3 test_bug_fixes.py --max_frames 50

# Full analysis (500 frames)
python3 test_bug_fixes.py --max_frames 500

# Custom video
python3 test_bug_fixes.py \
    --source_video_path "your_video.mp4" \
    --target_video_path "output.mp4" \
    --max_frames 1000
```

## 🔍 **Troubleshooting**

- **Missing models**: Run `./setup.sh`
- **Permission error**: `chmod +x setup.sh`
- **Python error**: Check Python 3.8+
- **Memory issue**: Reduce `--max_frames`

## 📈 **Performance**

- **50 frames**: ~4 minutes
- **500 frames**: ~40 minutes
- **1000 frames**: ~80 minutes

## 🎯 **Version 3.0 Improvements**

- ✅ Fixed jersey number visibility
- ✅ Added boundary clipping
- ✅ Implemented local data export
- ✅ Added real-time data overlay
- ✅ Created comprehensive documentation
- ✅ Added mock database support
- ✅ Enhanced split-screen analysis

---

**Built with**: Python, OpenCV, YOLO, Supervision, PostgreSQL
**License**: MIT
**Repository**: https://github.com/ashok-sravanam/sports-optimized-