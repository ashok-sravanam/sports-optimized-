#!/usr/bin/env python3
"""
Enhanced Soccer Database Manager
Integrates with existing PostgreSQL schema for comprehensive match tracking
"""

import psycopg2
from psycopg2.extras import execute_batch, RealDictCursor
from typing import List, Dict, Optional, Tuple
import numpy as np

class SoccerDatabaseManager:
    """
    Database manager for soccer tactical analysis
    Uses existing schema with teams, matches, players, formations, tracked_positions
    """
    
    def __init__(self, db_config: dict):
        """
        db_config = {
            'host': 'localhost',
            'database': 'soccer_analysis',
            'user': 'postgres',
            'password': 'your_password',
            'port': 5432
        }
        """
        self.config = db_config
        self.conn = None
        self.cursor = None
        
        # Caches
        self.tracker_to_player = {}  # tracker_id -> {'player_id', 'jersey_number', 'team_id'}
        self.jersey_to_player = {}   # (team_id, jersey_number) -> player_id
        
        # Current match context
        self.current_match_id = None
        self.team_a_id = None
        self.team_b_id = None
    
    def connect(self):
        """Establish database connection"""
        self.conn = psycopg2.connect(**self.config)
        self.cursor = self.conn.cursor(cursor_factory=RealDictCursor)
        print("✓ Database connected")
    
    def disconnect(self):
        """Close database connection"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.commit()
            self.conn.close()
        print("✓ Database disconnected")
    
    def setup_match(self, match_id: int):
        """
        Setup match context - load teams and players
        MUST be called before processing video
        """
        self.current_match_id = match_id
        
        # Get match details
        self.cursor.execute(
            """
            SELECT team_a_id, team_b_id, match_date, venue, competition
            FROM matches
            WHERE match_id = %s
            """,
            (match_id,)
        )
        match = self.cursor.fetchone()
        
        if not match:
            raise ValueError(f"Match {match_id} not found in database")
        
        self.team_a_id = match['team_a_id']
        self.team_b_id = match['team_b_id']
        
        print(f"✓ Match setup: Team {self.team_a_id} vs Team {self.team_b_id}")
        print(f"  Venue: {match['venue']}, Competition: {match['competition']}")
        
        # Load all players for both teams
        self._load_players()
        
        # Load formations if available
        self._load_formations()
    
    def _load_players(self):
        """Load all players for both teams in current match"""
        self.cursor.execute(
            """
            SELECT player_id, team_id, jersey_number, name, position
            FROM players
            WHERE team_id IN (%s, %s) AND is_active = TRUE
            ORDER BY team_id, jersey_number
            """,
            (self.team_a_id, self.team_b_id)
        )
        
        players = self.cursor.fetchall()
        
        for player in players:
            key = (player['team_id'], player['jersey_number'])
            self.jersey_to_player[key] = player
            
        print(f"✓ Loaded {len(players)} players from database")
        return players
    
    def _load_formations(self):
        """Load active formations for current match"""
        self.cursor.execute(
            """
            SELECT f.formation_id, f.team_id, f.formation_type,
                   fp.player_id, fp.jersey_number, fp.tactical_position
            FROM formations f
            LEFT JOIN formation_positions fp ON f.formation_id = fp.formation_id
            WHERE f.match_id = %s AND f.is_active = TRUE
            ORDER BY f.team_id, fp.jersey_number
            """,
            (self.current_match_id,)
        )
        
        formations = self.cursor.fetchall()
        
        if formations:
            team_formations = {}
            for row in formations:
                team_id = row['team_id']
                if team_id not in team_formations:
                    team_formations[team_id] = {
                        'formation_type': row['formation_type'],
                        'players': []
                    }
                if row['player_id']:
                    team_formations[team_id]['players'].append({
                        'jersey': row['jersey_number'],
                        'position': row['tactical_position']
                    })
            
            print(f"✓ Loaded formations:")
            for team_id, data in team_formations.items():
                print(f"  Team {team_id}: {data['formation_type']} ({len(data['players'])} players)")
            
            return team_formations
        else:
            print("⚠ No formations found for this match")
            return {}
    
    def get_team_id_from_classification(self, classified_team: int) -> int:
        """
        Convert team classifier output (0 or 1) to actual database team_id
        
        Args:
            classified_team: 0 or 1 from team classifier
        
        Returns:
            Actual team_id from database (e.g., 1, 2, 3, 4)
        """
        if classified_team == 0:
            return self.team_a_id
        elif classified_team == 1:
            return self.team_b_id
        else:
            return None  # Referee or unknown
    
    def assign_tracker_to_jersey(self, tracker_id: int, jersey_number: int, 
                                 classified_team: int) -> Dict:
        """
        Map tracker_id to jersey_number and get player info
        
        Args:
            tracker_id: ByteTrack tracker ID
            jersey_number: Assigned jersey (1-11)
            classified_team: Team classification (0 or 1)
        
        Returns:
            Dict with player_id, jersey_number, team_id, name, position
        """
        # Check if already mapped
        if tracker_id in self.tracker_to_player:
            return self.tracker_to_player[tracker_id]
        
        # Convert classified team to actual team_id
        team_id = self.get_team_id_from_classification(classified_team)
        
        if team_id is None:
            return None
        
        # Get player from database
        key = (team_id, jersey_number)
        player = self.jersey_to_player.get(key)
        
        if not player:
            print(f"⚠ Warning: Player not found for Team {team_id}, Jersey #{jersey_number}")
            return None
        
        # Cache the mapping
        mapping = {
            'player_id': player['player_id'],
            'jersey_number': player['jersey_number'],
            'team_id': player['team_id'],
            'name': player['name'],
            'position': player['position']
        }
        self.tracker_to_player[tracker_id] = mapping
        
        print(f"✓ Mapped tracker {tracker_id} → Jersey #{jersey_number} ({player['name']}, {player['position']})")
        
        return mapping
    
    def get_player_info(self, tracker_id: int) -> Optional[Dict]:
        """Get player info for tracker_id"""
        return self.tracker_to_player.get(tracker_id)
    
    def insert_tracked_positions_batch(self, positions: List[Dict]):
        """
        Batch insert tracked positions
        
        Args:
            positions: List of position dicts with:
                - frame_id: int
                - timestamp: float (seconds)
                - jersey_number: int
                - team_id: int
                - x: float (homography transformed)
                - y: float (homography transformed)
                - confidence: float
                - tracker_id: int
        """
        if not positions:
            return
        
        if self.current_match_id is None:
            raise ValueError("No match setup. Call setup_match() first.")
        
        # Add match_id to all positions
        for pos in positions:
            pos['match_id'] = self.current_match_id
        
        query = """
            INSERT INTO tracked_positions 
            (frame_id, match_id, timestamp, jersey_number, team_id, 
             x, y, confidence, tracker_id)
            VALUES (%(frame_id)s, %(match_id)s, %(timestamp)s, %(jersey_number)s, 
                    %(team_id)s, %(x)s, %(y)s, %(confidence)s, %(tracker_id)s)
        """
        
        try:
            execute_batch(self.cursor, query, positions, page_size=1000)
            self.conn.commit()
            # print(f"✓ Inserted {len(positions)} position records")
        except Exception as e:
            print(f"✗ Error inserting positions: {e}")
            self.conn.rollback()
    
    def insert_event(self, event_type: str, team_id: int, timestamp: str,
                    jersey_number: int, details: dict = None):
        """
        Insert match event (goal, card, assist, etc.)
        
        Args:
            event_type: 'GOAL', 'YELLOW_CARD', 'RED_CARD', 'ASSIST', 'SUBSTITUTION'
            team_id: Database team_id
            timestamp: Time string (e.g., '45:30')
            jersey_number: Player's jersey number
            details: Additional event data (JSONB)
        """
        # Get player_id from jersey_number and team_id
        key = (team_id, jersey_number)
        player = self.jersey_to_player.get(key)
        player_id = player['player_id'] if player else None
        
        self.cursor.execute(
            """
            INSERT INTO events 
            (match_id, event_type, team_id, timestamp, player_id, jersey_number, details)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (self.current_match_id, event_type, team_id, timestamp, 
             player_id, jersey_number, details)
        )
        self.conn.commit()
        
        print(f"✓ Event recorded: {event_type} - Jersey #{jersey_number} at {timestamp}")
    
    def insert_substitution(self, team_id: int, jersey_out: int, jersey_in: int, 
                          timestamp: str):
        """
        Record player substitution
        
        Args:
            team_id: Database team_id
            jersey_out: Player leaving the field
            jersey_in: Player entering the field
            timestamp: Time string (e.g., '60:00')
        """
        # Get player IDs
        key_out = (team_id, jersey_out)
        key_in = (team_id, jersey_in)
        
        player_out = self.jersey_to_player.get(key_out)
        player_in = self.jersey_to_player.get(key_in)
        
        player_out_id = player_out['player_id'] if player_out else None
        player_in_id = player_in['player_id'] if player_in else None
        
        self.cursor.execute(
            """
            INSERT INTO substitutions 
            (match_id, team_id, player_out, player_in, jersey_out, jersey_in, timestamp)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (self.current_match_id, team_id, player_out_id, player_in_id,
             jersey_out, jersey_in, timestamp)
        )
        self.conn.commit()
        
        out_name = player_out['name'] if player_out else f"#{jersey_out}"
        in_name = player_in['name'] if player_in else f"#{jersey_in}"
        print(f"✓ Substitution: {out_name} ➔ {in_name} at {timestamp}")
    
    def get_formation(self, team_id: int) -> Optional[str]:
        """Get formation type for team"""
        self.cursor.execute(
            """
            SELECT formation_type
            FROM formations
            WHERE match_id = %s AND team_id = %s AND is_active = TRUE
            LIMIT 1
            """,
            (self.current_match_id, team_id)
        )
        result = self.cursor.fetchone()
        return result['formation_type'] if result else None
    
    def get_team_color(self, team_id: int) -> Optional[str]:
        """Get team color hex code"""
        self.cursor.execute(
            "SELECT team_color FROM teams WHERE team_id = %s",
            (team_id,)
        )
        result = self.cursor.fetchone()
        return result['team_color'] if result else None
    
    def get_team_name(self, team_id: int) -> Optional[str]:
        """Get team name"""
        self.cursor.execute(
            "SELECT team_name FROM teams WHERE team_id = %s",
            (team_id,)
        )
        result = self.cursor.fetchone()
        return result['team_name'] if result else None
