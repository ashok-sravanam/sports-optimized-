# Soccer Analysis System - Version 3.0

**Professional soccer tracking and tactical analysis with unified ball tracking**

*Developed by Ashok Sravanam*

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

# Run analysis - Working Version V3 with Ball Tracking
python3 implementations/working_version/soccer_analysis_v3_with_ball.py \
    --source_video_path "video_outputs/psgVSliv.mov" \
    --target_video_path "video_outputs/analysis_output.mp4" \
    --max_frames 500 \
    --device cpu
```

## ✨ **Features**

- **Unified Tracking**: Players AND ball in same CSV table
- **Split-Screen Analysis**: Live video feed + tactical board
- **Player Tracking**: 20+ players with jersey numbers (1-15 per team)
- **Ball Tracking**: Complete ball position data with confidence scores
- **Team Classification**: Automatic team identification
- **Data Export**: CSV/JSON/TXT with complete tracking data
- **Mock Database**: Works without PostgreSQL
- **Real-Time Progress**: Live processing updates
- **Professional Visualization**: Tactical board with formations

## 📁 **File Structure**

```
soccer/
├── 📁 implementations/                    # All implementations organized
│   ├── working_version/                  # ✅ Production-ready system
│   │   ├── soccer_analysis_v3_with_ball.py  # Main analysis script
│   │   ├── split_screen_database_analysis.py # Core processing
│   │   ├── local_data_exporter.py        # Data export with ball tracking
│   │   ├── soccer_database_manager.py    # Database integration
│   │   └── enhanced_jersey_manager.py   # Jersey assignment
│   ├── experimental/                      # 🔬 Advanced features
│   │   ├── soccer_analysis_final.py     # Position-based IDs (1-11)
│   │   ├── position_assignment.py       # Spatial position logic
│   │   ├── unified_exporter.py          # Advanced export
│   │   └── formation_manager.py         # Dynamic formations
│   ├── legacy/                           # 📚 Historical versions
│   │   ├── comprehensive_analysis.py    # Original 6-mode system
│   │   ├── clean_analysis.py           # Simplified version
│   │   ├── tactical_analysis.py        # Early tactical system
│   │   └── split_screen_soccer_analysis.py # First split-screen
│   └── optimization_attempts/             # ⚡ Performance experiments
│       └── *_openvino_model/            # OpenVINO optimization
│
├── 📄 Current Files
│   ├── test_bug_fixes.py                # Easy testing script (legacy)
│   ├── analyze_tracking_data.py         # Data analysis tool
│   ├── database_schema.sql              # PostgreSQL schema
│   └── setup_database_sample.py        # Database setup
│
├── 📁 tracking_data/                     # Exported data
│   ├── positions_*.csv                  # Unified tracking data
│   ├── summary_*.json                   # Statistical summaries
│   ├── stats_*.txt                      # Human-readable reports
│   └── TRACKING_DATA_COLUMNS_README.md  # Data documentation
│
├── 📁 video_outputs/                     # Analysis videos
├── 📁 misc/                              # Additional files
└── 📄 README.md                          # This file
```

## 🎯 **Output Files**

After running analysis, you'll get:

- **📹 Video**: `video_outputs/analysis_output.mp4` (split-screen with tactical board)
- **📊 Data**: `tracking_data/positions_YYYYMMDD_HHMMSS.csv` (players + ball in unified table)
- **📈 Stats**: `tracking_data/summary_YYYYMMDD_HHMMSS.json` (statistics including ball metrics)
- **📋 Report**: `tracking_data/stats_YYYYMMDD_HHMMSS.txt` (human-readable analysis)
- **📖 Docs**: `tracking_data/TRACKING_DATA_COLUMNS_README.md` (complete data guide)

## 🔧 **Requirements**

- **Python 3.8+**
- **4GB+ RAM**
- **~2GB disk space**
- **macOS/Linux/Windows**

## 📊 **Data Columns**

| Column | Description | Example | Notes |
|--------|-------------|---------|-------|
| frame | Video frame number | 123 | |
| timestamp | Time in seconds | 4.39 | |
| tracker_id | Unique player ID | 2 | -1 for ball |
| jersey | Jersey number (1-15) | 1 | "BALL" for ball |
| team_id | Team identifier | 1 | Empty for ball |
| player_name | Player name | "Courtois" | "Ball" for ball |
| video_x, video_y | Video pixel coordinates | 410, 506 | |
| pitch_x, pitch_y | Real-world coordinates | 5974, 6664 | |
| board_x, board_y | Tactical board coordinates | 597, 761 | |
| confidence | Detection confidence | 0.916 | |

## 🎮 **Usage Examples**

```bash
# Quick test (50 frames) - Working Version V3 with Ball Tracking
python3 implementations/working_version/soccer_analysis_v3_with_ball.py \
    --source_video_path "video_outputs/psgVSliv.mov" \
    --target_video_path "video_outputs/test_50frames.mp4" \
    --max_frames 50 \
    --device cpu

# Full analysis (500 frames)
python3 implementations/working_version/soccer_analysis_v3_with_ball.py \
    --source_video_path "video_outputs/psgVSliv.mov" \
    --target_video_path "video_outputs/full_analysis.mp4" \
    --max_frames 500 \
    --device cpu

# Custom video
python3 implementations/working_version/soccer_analysis_v3_with_ball.py \
    --source_video_path "your_video.mp4" \
    --target_video_path "output.mp4" \
    --max_frames 1000 \
    --device cpu
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

- ✅ Fixed jersey number visibility with contrasting text colors
- ✅ Added boundary clipping to keep players within pitch bounds
- ✅ Implemented comprehensive local data export (CSV/JSON/TXT)
- ✅ Added real-time data overlay showing coordinates and stats
- ✅ Created detailed documentation for all tracking data columns
- ✅ Added mock database support for testing without PostgreSQL
- ✅ Enhanced split-screen analysis with professional tactical board
- ✅ Implemented coordinate transformation across multiple systems
- ✅ Added automated jersey number assignment system
- ✅ Created modular architecture with clean code organization

## 🏆 **Project Highlights**

This project demonstrates advanced computer vision and machine learning techniques applied to sports analytics. The system processes soccer videos to extract detailed player tracking data, perform tactical analysis, and export comprehensive datasets for further analysis.

### **Key Technical Achievements**
- **Real-time Processing**: Handles 500+ frames with full analysis pipeline
- **Multi-coordinate Systems**: Video → Pitch → Tactical board transformations
- **Database Integration**: PostgreSQL with mock database fallback
- **Professional UI**: Split-screen interface with tactical board visualization
- **Data Export**: 10,000+ position records with complete metadata

---

**Developer**: Ashok Sravanam  
**Technologies**: Python, OpenCV, YOLO, Supervision, PostgreSQL, scikit-learn  
**Repository**: https://github.com/ashok-sravanam/sports-optimized-