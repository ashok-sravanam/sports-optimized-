#!/usr/bin/env python3
"""
Test Database Integration
Tests the database integration system with sample data
"""

import argparse
import os
from soccer_database_manager import SoccerDatabaseManager

def test_database_integration(db_config: dict):
    """Test database integration functionality"""
    
    print("Testing database integration...")
    
    # Connect to database
    db = SoccerDatabaseManager(db_config)
    db.connect()
    
    try:
        # Test 1: Setup match
        print("\n1. Testing match setup...")
        db.setup_match(match_id=1)
        
        # Test 2: Get team info
        print("\n2. Testing team info retrieval...")
        team_a_name = db.get_team_name(db.team_a_id)
        team_b_name = db.get_team_name(db.team_b_id)
        team_a_color = db.get_team_color(db.team_a_id)
        team_b_color = db.get_team_color(db.team_b_id)
        
        print(f"Team A: {team_a_name} ({team_a_color})")
        print(f"Team B: {team_b_name} ({team_b_color})")
        
        # Test 3: Get formations
        print("\n3. Testing formation retrieval...")
        formation_a = db.get_formation(db.team_a_id)
        formation_b = db.get_formation(db.team_b_id)
        
        print(f"Team A Formation: {formation_a}")
        print(f"Team B Formation: {formation_b}")
        
        # Test 4: Test team classification mapping
        print("\n4. Testing team classification mapping...")
        team_id_0 = db.get_team_id_from_classification(0)
        team_id_1 = db.get_team_id_from_classification(1)
        
        print(f"Classification 0 → Team ID: {team_id_0}")
        print(f"Classification 1 → Team ID: {team_id_1}")
        
        # Test 5: Test jersey assignment
        print("\n5. Testing jersey assignment...")
        test_tracker_id = 15
        test_jersey = 7
        test_classification = 0
        
        player_info = db.assign_tracker_to_jersey(
            tracker_id=test_tracker_id,
            jersey_number=test_jersey,
            classified_team=test_classification
        )
        
        if player_info:
            print(f"✓ Tracker {test_tracker_id} → Jersey #{test_jersey}")
            print(f"  Player: {player_info['name']} ({player_info['position']})")
        else:
            print("✗ Jersey assignment failed")
        
        # Test 6: Test position insertion
        print("\n6. Testing position insertion...")
        test_positions = [
            {
                'frame_id': 100,
                'timestamp': 3.33,
                'jersey_number': 7,
                'team_id': team_id_0,
                'x': 52.5,
                'y': 34.0,
                'confidence': 0.95,
                'tracker_id': test_tracker_id
            },
            {
                'frame_id': 101,
                'timestamp': 3.36,
                'jersey_number': 7,
                'team_id': team_id_0,
                'x': 53.0,
                'y': 35.0,
                'confidence': 0.92,
                'tracker_id': test_tracker_id
            }
        ]
        
        db.insert_tracked_positions_batch(test_positions)
        print("✓ Position batch inserted successfully")
        
        # Test 7: Test event insertion
        print("\n7. Testing event insertion...")
        db.insert_event(
            event_type='GOAL',
            team_id=team_id_0,
            timestamp='45:30',
            jersey_number=7,
            details={'assist_jersey': 10, 'goal_type': 'open_play'}
        )
        print("✓ Event inserted successfully")
        
        # Test 8: Test substitution
        print("\n8. Testing substitution...")
        db.insert_substitution(
            team_id=team_id_0,
            jersey_out=7,
            jersey_in=9,
            timestamp='60:00'
        )
        print("✓ Substitution recorded successfully")
        
        print("\n✓ All database integration tests passed!")
        
    except Exception as e:
        print(f"✗ Database integration test failed: {e}")
        raise
    finally:
        db.disconnect()

def main():
    parser = argparse.ArgumentParser(description="Test database integration")
    parser.add_argument("--db_host", type=str, default="localhost")
    parser.add_argument("--db_name", type=str, default="soccer_analysis")
    parser.add_argument("--db_user", type=str, default="postgres")
    parser.add_argument("--db_password", type=str, required=True,
                       help="Database password")
    parser.add_argument("--db_port", type=int, default=5432)
    
    args = parser.parse_args()
    
    db_config = {
        'host': args.db_host,
        'database': args.db_name,
        'user': args.db_user,
        'password': args.db_password,
        'port': args.db_port
    }
    
    test_database_integration(db_config)

if __name__ == "__main__":
    main()
