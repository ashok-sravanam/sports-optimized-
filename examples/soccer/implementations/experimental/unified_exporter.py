import csv
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

class UnifiedTrackingExporter:
    """
    Export player AND ball tracking data to a single unified CSV table.
    NO player names, only position numbers (1-11).
    """
    
    def __init__(self, output_dir: str = "tracking_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Generate timestamp for this session
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Output files
        self.tracking_csv = self.output_dir / f"tracking_{self.timestamp}.csv"
        self.summary_json = self.output_dir / f"summary_{self.timestamp}.json"
        self.stats_txt = self.output_dir / f"stats_{self.timestamp}.txt"
        
        # Data buffer (holds all tracking records)
        self.tracking_buffer = []
        
        print(f"✓ Unified tracking export initialized: {self.output_dir}")
    
    def add_player(self, frame_idx: int, timestamp: float, tracker_id: int,
                   position: int, team_id: int,
                   video_x: float, video_y: float,
                   pitch_x: float, pitch_y: float,
                   board_x: float, board_y: float,
                   confidence: float):
        """
        Add player tracking record.
        
        Args:
            frame_idx: Frame number
            timestamp: Time in seconds
            tracker_id: ByteTrack ID (for internal tracking)
            position: Position number (1-11)
            team_id: Team database ID
            video_x, video_y: Coordinates in video frame
            pitch_x, pitch_y: Homography-transformed pitch coordinates (meters)
            board_x, board_y: Tactical board display coordinates (pixels)
            confidence: Detection confidence (0-1)
        """
        
        self.tracking_buffer.append({
            'frame': frame_idx,
            'timestamp': round(timestamp, 3),
            'tracker_id': tracker_id,
            'object_type': 'player',
            'position': position,
            'team_id': team_id,
            'video_x': round(video_x, 2),
            'video_y': round(video_y, 2),
            'pitch_x': round(pitch_x, 2),
            'pitch_y': round(pitch_y, 2),
            'board_x': round(board_x, 2),
            'board_y': round(board_y, 2),
            'confidence': round(confidence, 3)
        })
    
    def add_ball(self, frame_idx: int, timestamp: float,
                 video_x: float, video_y: float,
                 pitch_x: float, pitch_y: float,
                 board_x: float, board_y: float,
                 confidence: float):
        """
        Add ball tracking record.
        
        Args:
            frame_idx: Frame number
            timestamp: Time in seconds
            video_x, video_y: Coordinates in video frame
            pitch_x, pitch_y: Homography-transformed pitch coordinates (meters)
            board_x, board_y: Tactical board display coordinates (pixels)
            confidence: Detection confidence (0-1)
        """
        
        self.tracking_buffer.append({
            'frame': frame_idx,
            'timestamp': round(timestamp, 3),
            'tracker_id': -1,  # Special ID for ball
            'object_type': 'ball',
            'position': None,  # Ball has no position number
            'team_id': None,   # Ball has no team
            'video_x': round(video_x, 2),
            'video_y': round(video_y, 2),
            'pitch_x': round(pitch_x, 2),
            'pitch_y': round(pitch_y, 2),
            'board_x': round(board_x, 2),
            'board_y': round(board_y, 2),
            'confidence': round(confidence, 3)
        })
    
    def write_csv(self):
        """Write unified tracking data to CSV file."""
        if not self.tracking_buffer:
            print("⚠ No tracking data to export")
            return
        
        with open(self.tracking_csv, 'w', newline='') as f:
            fieldnames = [
                'frame', 'timestamp', 'tracker_id', 'object_type',
                'position', 'team_id', 'video_x', 'video_y',
                'pitch_x', 'pitch_y', 'board_x', 'board_y', 'confidence'
            ]
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.tracking_buffer)
        
        print(f"✓ Exported {len(self.tracking_buffer)} records to: {self.tracking_csv}")
    
    def write_summary(self):
        """Write summary statistics to JSON."""
        if not self.tracking_buffer:
            return
        
        df = pd.DataFrame(self.tracking_buffer)
        
        summary = {
            'session_timestamp': self.timestamp,
            'total_records': len(df),
            'total_frames': int(df['frame'].max()) if len(df) > 0 else 0,
            'duration_seconds': float(df['timestamp'].max()) if len(df) > 0 else 0.0,
            'players': {},
            'ball': {}
        }
        
        # Player statistics
        player_df = df[df['object_type'] == 'player']
        
        if len(player_df) > 0:
            for tracker_id, group in player_df.groupby('tracker_id'):
                position = int(group.iloc[0]['position'])
                team_id = int(group.iloc[0]['team_id'])
                
                summary['players'][str(tracker_id)] = {
                    'position': position,
                    'team_id': team_id,
                    'frames_tracked': len(group),
                    'avg_pitch_position': {
                        'x': round(float(group['pitch_x'].mean()), 2),
                        'y': round(float(group['pitch_y'].mean()), 2)
                    },
                    'position_range': {
                        'x_min': round(float(group['pitch_x'].min()), 2),
                        'x_max': round(float(group['pitch_x'].max()), 2),
                        'y_min': round(float(group['pitch_y'].min()), 2),
                        'y_max': round(float(group['pitch_y'].max()), 2)
                    },
                    'avg_confidence': round(float(group['confidence'].mean()), 3)
                }
        
        # Ball statistics
        ball_df = df[df['object_type'] == 'ball']
        
        if len(ball_df) > 0:
            summary['ball'] = {
                'frames_tracked': len(ball_df),
                'avg_pitch_position': {
                    'x': round(float(ball_df['pitch_x'].mean()), 2),
                    'y': round(float(ball_df['pitch_y'].mean()), 2)
                },
                'position_range': {
                    'x_min': round(float(ball_df['pitch_x'].min()), 2),
                    'x_max': round(float(ball_df['pitch_x'].max()), 2),
                    'y_min': round(float(ball_df['pitch_y'].min()), 2),
                    'y_max': round(float(ball_df['pitch_y'].max()), 2)
                },
                'avg_confidence': round(float(ball_df['confidence'].mean()), 3)
            }
        
        with open(self.summary_json, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✓ Exported summary to: {self.summary_json}")
    
    def write_stats_report(self):
        """Write human-readable statistics report."""
        if not self.tracking_buffer:
            return
        
        df = pd.DataFrame(self.tracking_buffer)
        
        with open(self.stats_txt, 'w') as f:
            f.write("="*70 + "\n")
            f.write("UNIFIED TRACKING DATA ANALYSIS REPORT\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Session: {self.timestamp}\n")
            f.write(f"Total Records: {len(df)}\n")
            f.write(f"Total Frames: {df['frame'].max()}\n")
            f.write(f"Duration: {df['timestamp'].max():.2f} seconds\n\n")
            
            # Player stats
            player_df = df[df['object_type'] == 'player']
            
            if len(player_df) > 0:
                f.write("-"*70 + "\n")
                f.write("PLAYER STATISTICS\n")
                f.write("-"*70 + "\n\n")
                
                for tracker_id, group in player_df.groupby('tracker_id'):
                    position = int(group.iloc[0]['position'])
                    team_id = int(group.iloc[0]['team_id'])
                    
                    f.write(f"Position #{position} (Team {team_id})\n")
                    f.write(f"  Tracker ID: {tracker_id}\n")
                    f.write(f"  Frames Tracked: {len(group)}\n")
                    f.write(f"  Avg Pitch Position: ({group['pitch_x'].mean():.2f}, "
                           f"{group['pitch_y'].mean():.2f})\n")
                    f.write(f"  Pitch Range: X=[{group['pitch_x'].min():.2f}, "
                           f"{group['pitch_x'].max():.2f}], "
                           f"Y=[{group['pitch_y'].min():.2f}, {group['pitch_y'].max():.2f}]\n")
                    f.write(f"  Avg Confidence: {group['confidence'].mean():.3f}\n\n")
            
            # Ball stats
            ball_df = df[df['object_type'] == 'ball']
            
            if len(ball_df) > 0:
                f.write("-"*70 + "\n")
                f.write("BALL STATISTICS\n")
                f.write("-"*70 + "\n\n")
                
                f.write(f"Frames Tracked: {len(ball_df)}\n")
                f.write(f"Avg Pitch Position: ({ball_df['pitch_x'].mean():.2f}, "
                       f"{ball_df['pitch_y'].mean():.2f})\n")
                f.write(f"Pitch Range: X=[{ball_df['pitch_x'].min():.2f}, "
                       f"{ball_df['pitch_x'].max():.2f}], "
                       f"Y=[{ball_df['pitch_y'].min():.2f}, {ball_df['pitch_y'].max():.2f}]\n")
                f.write(f"Avg Confidence: {ball_df['confidence'].mean():.3f}\n")
        
        print(f"✓ Exported stats report to: {self.stats_txt}")
    
    def export_all(self):
        """Export all data files."""
        self.write_csv()
        self.write_summary()
        self.write_stats_report()
        
        print(f"\n{'='*70}")
        print("EXPORT COMPLETE")
        print(f"{'='*70}")
        print(f"CSV:     {self.tracking_csv}")
        print(f"JSON:    {self.summary_json}")
        print(f"Report:  {self.stats_txt}")
        print(f"{'='*70}\n")
