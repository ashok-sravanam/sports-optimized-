<div align="center">

  <h1>Soccer Analysis System</h1>
  <h3>Professional Computer Vision & Machine Learning Project</h3>
  
  <p><strong>Developed by Ashok Sravanam</strong></p>

</div>

## 👋 About This Project

This repository contains a comprehensive soccer analysis system that demonstrates advanced computer vision and machine learning techniques. The system performs real-time player tracking, team classification, tactical analysis, and database integration with professional-grade data export capabilities.

## 🎯 Project Overview

The Soccer Analysis System processes soccer videos to extract detailed player tracking data, perform tactical analysis, and export comprehensive datasets. It features a split-screen interface with real-time video analysis and tactical board visualization.

## 🎯 Technical Challenges Solved

This project addresses several complex computer vision challenges in sports analytics:

- **Real-time Player Tracking:** Implemented robust player tracking using YOLO and ByteTrack to maintain consistent player identification across frames
- **Team Classification:** Developed automated team classification using SigLIP feature extraction and machine learning clustering
- **Multi-coordinate Transformation:** Created coordinate transformation pipeline from video pixels to real-world pitch coordinates
- **Database Integration:** Built comprehensive PostgreSQL integration with mock database fallback for testing
- **Data Export Pipeline:** Implemented complete ETL pipeline with CSV/JSON/TXT export formats
- **Professional UI:** Developed split-screen interface with tactical board visualization and real-time progress tracking

## 💻 Installation

Install the system in a [**Python>=3.8**](https://www.python.org/) environment.

```bash
# Clone the repository
git clone https://github.com/ashok-sravanam/sports-optimized-.git
cd sports-optimized/examples/soccer

# Install dependencies
pip3 install -r requirements.txt
pip3 install -r tactical_requirements.txt

# Download models and sample video
chmod +x setup.sh
./setup.sh
```

## 🚀 Quick Start

```bash
# Run the analysis system
python3 test_bug_fixes.py \
    --source_video_path "video_outputs/psgVSliv.mov" \
    --target_video_path "video_outputs/analysis_output.mp4" \
    --max_frames 500 \
    --device cpu
```

## 🎯 Features

- **Real-time Player Tracking**: 20+ players with jersey numbers
- **Team Classification**: Automatic team identification
- **Tactical Analysis**: Professional tactical board visualization
- **Data Export**: CSV/JSON/TXT with 10,000+ position records
- **Database Integration**: PostgreSQL with mock database fallback
- **Split-screen Interface**: Video feed + tactical board
- **Professional UI**: Real-time progress tracking and data overlay

## 📊 Output

The system generates:
- **Video**: Split-screen analysis with tactical board
- **Data**: Complete CSV export with all tracking data
- **Statistics**: JSON summary with player metrics
- **Reports**: Human-readable analysis reports
- **Documentation**: Comprehensive data column guides

## 🏆 Technical Achievements

- **Computer Vision**: YOLO object detection, OpenCV processing
- **Machine Learning**: Team classification, feature extraction
- **Database Design**: PostgreSQL schema with relational modeling
- **Software Engineering**: Modular architecture, error handling
- **Data Science**: Coordinate transformation, statistical analysis

---

**Developer**: Ashok Sravanam  
**Technologies**: Python, OpenCV, YOLO, PostgreSQL, scikit-learn  
**Repository**: https://github.com/ashok-sravanam/sports-optimized-
