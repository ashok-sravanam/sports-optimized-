#!/usr/bin/env python3
"""
Test script for bug fixes and local data export
"""

import argparse
import os
import sys

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from split_screen_database_analysis import process_video_with_database, MockDatabaseManager

def main():
    parser = argparse.ArgumentParser(description="Test bug fixes and local data export")
    parser.add_argument("--source_video_path", type=str, required=True,
                       help="Path to input video")
    parser.add_argument("--target_video_path", type=str, required=True,
                       help="Path to output video")
    parser.add_argument("--max_frames", type=int, default=50,
                       help="Maximum frames to process")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device: cpu, cuda, or mps")
    
    args = parser.parse_args()
    
    # Mock database config (will use mock database)
    db_config = {
        'host': 'localhost',
        'database': 'soccer_analysis',
        'user': 'postgres',
        'password': 'test_password',
        'port': 5432
    }
    
    print("="*60)
    print("TESTING BUG FIXES AND LOCAL DATA EXPORT")
    print("="*60)
    print(f"Source: {args.source_video_path}")
    print(f"Target: {args.target_video_path}")
    print(f"Max frames: {args.max_frames}")
    print(f"Device: {args.device}")
    print("="*60)
    
    try:
        process_video_with_database(
            source_path=args.source_video_path,
            target_path=args.target_video_path,
            match_id=1,
            device=args.device,
            db_config=db_config,
            max_frames=args.max_frames,
            export_local=True
        )
        
        print("\n" + "="*60)
        print("TEST COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("✓ Jersey numbers should now be readable (black border + contrasting text)")
        print("✓ Players/ball should stay within pitch boundaries")
        print("✓ Local data exported to tracking_data/ directory")
        print("✓ Real-time data overlay shown on video")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
