"""
Database connection and operations for soccer tactical analysis system.
Handles PostgreSQL connections and basic CRUD operations.
"""

import psycopg2
import psycopg2.extras
import json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, time
import os

@dataclass
class DatabaseConfig:
    """Database configuration"""
    host: str = "localhost"
    port: int = 5432
    database: str = "soccer_analysis"
    user: str = "postgres"
    password: str = "password"

class SoccerDatabase:
    """Main database class for soccer analysis operations"""
    
    def __init__(self, config: DatabaseConfig = None):
        self.config = config or DatabaseConfig()
        self.connection = None
        
    def connect(self):
        """Establish database connection"""
        try:
            self.connection = psycopg2.connect(
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
                user=self.config.user,
                password=self.config.password
            )
            self.connection.autocommit = True
            print(f"✓ Connected to database: {self.config.database}")
            return True
        except psycopg2.Error as e:
            print(f"✗ Database connection failed: {e}")
            return False
    
    def disconnect(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            print("✓ Database connection closed")
    
    def execute_query(self, query: str, params: tuple = None) -> List[Dict]:
        """Execute SELECT query and return results"""
        try:
            with self.connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute(query, params)
                return cursor.fetchall()
        except psycopg2.Error as e:
            print(f"✗ Query execution failed: {e}")
            return []
    
    def execute_insert(self, query: str, params: tuple = None) -> int:
        """Execute INSERT query and return inserted ID"""
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query, params)
                return cursor.fetchone()[0] if cursor.description else 0
        except psycopg2.Error as e:
            print(f"✗ Insert execution failed: {e}")
            return 0
    
    def execute_update(self, query: str, params: tuple = None) -> int:
        """Execute UPDATE/DELETE query and return affected rows"""
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query, params)
                return cursor.rowcount
        except psycopg2.Error as e:
            print(f"✗ Update execution failed: {e}")
            return 0
    
    # Team operations
    def get_teams(self) -> List[Dict]:
        """Get all teams"""
        return self.execute_query("SELECT * FROM teams ORDER BY team_name")
    
    def get_team_by_id(self, team_id: int) -> Optional[Dict]:
        """Get team by ID"""
        results = self.execute_query("SELECT * FROM teams WHERE team_id = %s", (team_id,))
        return results[0] if results else None
    
    def create_team(self, team_name: str, team_color: str = None) -> int:
        """Create new team"""
        query = "INSERT INTO teams (team_name, team_color) VALUES (%s, %s) RETURNING team_id"
        return self.execute_insert(query, (team_name, team_color))
    
    # Match operations
    def get_matches(self) -> List[Dict]:
        """Get all matches"""
        query = """
        SELECT m.*, 
               ta.team_name as team_a_name, tb.team_name as team_b_name
        FROM matches m
        LEFT JOIN teams ta ON m.team_a_id = ta.team_id
        LEFT JOIN teams tb ON m.team_b_id = tb.team_id
        ORDER BY m.match_date DESC
        """
        return self.execute_query(query)
    
    def create_match(self, team_a_id: int, team_b_id: int, match_date: datetime = None, 
                    venue: str = None, competition: str = None) -> int:
        """Create new match"""
        query = """
        INSERT INTO matches (team_a_id, team_b_id, match_date, venue, competition) 
        VALUES (%s, %s, %s, %s, %s) RETURNING match_id
        """
        return self.execute_insert(query, (team_a_id, team_b_id, match_date, venue, competition))
    
    # Player operations
    def get_players_by_team(self, team_id: int) -> List[Dict]:
        """Get all players for a team"""
        return self.execute_query(
            "SELECT * FROM players WHERE team_id = %s ORDER BY jersey_number", 
            (team_id,)
        )
    
    def get_player_by_jersey(self, team_id: int, jersey_number: int) -> Optional[Dict]:
        """Get player by team and jersey number"""
        results = self.execute_query(
            "SELECT * FROM players WHERE team_id = %s AND jersey_number = %s", 
            (team_id, jersey_number)
        )
        return results[0] if results else None
    
    def create_player(self, team_id: int, jersey_number: int, name: str = None, 
                     position: str = None) -> int:
        """Create new player"""
        query = """
        INSERT INTO players (team_id, jersey_number, name, position) 
        VALUES (%s, %s, %s, %s) RETURNING player_id
        """
        return self.execute_insert(query, (team_id, jersey_number, name, position))
    
    # Position tracking operations
    def insert_tracked_position(self, frame_id: int, match_id: int, timestamp: float,
                               jersey_number: int, team_id: int, x: float, y: float,
                               confidence: float = 1.0, tracker_id: int = None) -> int:
        """Insert tracked player position"""
        query = """
        INSERT INTO tracked_positions 
        (frame_id, match_id, timestamp, jersey_number, team_id, x, y, confidence, tracker_id)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING position_id
        """
        return self.execute_insert(query, (frame_id, match_id, timestamp, jersey_number, 
                                         team_id, x, y, confidence, tracker_id))
    
    def get_positions_by_match(self, match_id: int, start_time: float = None, 
                              end_time: float = None) -> List[Dict]:
        """Get tracked positions for a match within time range"""
        query = """
        SELECT tp.*, p.name as player_name, t.team_name
        FROM tracked_positions tp
        LEFT JOIN players p ON tp.jersey_number = p.jersey_number AND tp.team_id = p.team_id
        LEFT JOIN teams t ON tp.team_id = t.team_id
        WHERE tp.match_id = %s
        """
        params = [match_id]
        
        if start_time is not None:
            query += " AND tp.timestamp >= %s"
            params.append(start_time)
        
        if end_time is not None:
            query += " AND tp.timestamp <= %s"
            params.append(end_time)
        
        query += " ORDER BY tp.timestamp, tp.jersey_number"
        return self.execute_query(query, tuple(params))
    
    # Event operations
    def create_event(self, match_id: int, event_type: str, team_id: int, 
                    timestamp: time, player_id: int = None, jersey_number: int = None,
                    details: Dict = None) -> int:
        """Create match event"""
        query = """
        INSERT INTO events (match_id, event_type, team_id, timestamp, player_id, jersey_number, details)
        VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING event_id
        """
        details_json = json.dumps(details) if details else None
        return self.execute_insert(query, (match_id, event_type, team_id, timestamp, 
                                         player_id, jersey_number, details_json))
    
    def get_events_by_match(self, match_id: int) -> List[Dict]:
        """Get all events for a match"""
        query = """
        SELECT e.*, p.name as player_name, t.team_name
        FROM events e
        LEFT JOIN players p ON e.player_id = p.player_id
        LEFT JOIN teams t ON e.team_id = t.team_id
        WHERE e.match_id = %s
        ORDER BY e.timestamp
        """
        return self.execute_query(query, (match_id,))
    
    # Formation operations
    def create_formation(self, match_id: int, team_id: int, formation_type: str) -> int:
        """Create formation"""
        query = """
        INSERT INTO formations (match_id, team_id, formation_type)
        VALUES (%s, %s, %s) RETURNING formation_id
        """
        return self.execute_insert(query, (match_id, team_id, formation_type))
    
    def add_formation_position(self, formation_id: int, player_id: int, jersey_number: int,
                              tactical_position: str, x_start: float, y_start: float):
        """Add player position to formation"""
        query = """
        INSERT INTO formation_positions 
        (formation_id, player_id, jersey_number, tactical_position, x_start, y_start)
        VALUES (%s, %s, %s, %s, %s, %s)
        """
        self.execute_update(query, (formation_id, player_id, jersey_number, 
                                   tactical_position, x_start, y_start))

# Utility functions
def create_database_if_not_exists(config: DatabaseConfig):
    """Create database if it doesn't exist"""
    try:
        # Connect to default postgres database
        conn = psycopg2.connect(
            host=config.host,
            port=config.port,
            database="postgres",
            user=config.user,
            password=config.password
        )
        conn.autocommit = True
        
        with conn.cursor() as cursor:
            cursor.execute(f"SELECT 1 FROM pg_database WHERE datname = '{config.database}'")
            if not cursor.fetchone():
                cursor.execute(f"CREATE DATABASE {config.database}")
                print(f"✓ Created database: {config.database}")
            else:
                print(f"✓ Database already exists: {config.database}")
        
        conn.close()
        return True
    except psycopg2.Error as e:
        print(f"✗ Database creation failed: {e}")
        return False

def setup_database_schema(config: DatabaseConfig):
    """Setup database schema from SQL file"""
    try:
        db = SoccerDatabase(config)
        if not db.connect():
            return False
        
        # Read and execute schema file
        schema_file = os.path.join(os.path.dirname(__file__), 'database_schema.sql')
        with open(schema_file, 'r') as f:
            schema_sql = f.read()
        
        # Split by semicolon and execute each statement
        statements = [stmt.strip() for stmt in schema_sql.split(';') if stmt.strip()]
        
        with db.connection.cursor() as cursor:
            for statement in statements:
                if statement:
                    cursor.execute(statement)
        
        print("✓ Database schema setup completed")
        db.disconnect()
        return True
    except Exception as e:
        print(f"✗ Schema setup failed: {e}")
        return False

if __name__ == "__main__":
    # Test database connection
    config = DatabaseConfig()
    
    print("Setting up database...")
    if create_database_if_not_exists(config):
        if setup_database_schema(config):
            print("✓ Database setup completed successfully!")
        else:
            print("✗ Schema setup failed")
    else:
        print("✗ Database creation failed")
