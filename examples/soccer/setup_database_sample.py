#!/usr/bin/env python3
"""
Database Setup Script with Sample Data
Creates sample teams, players, and match data for testing
"""

import psycopg2
from psycopg2.extras import RealDictCursor
import argparse
from datetime import datetime

def setup_database_sample(db_config: dict):
    """Setup database with sample data for testing"""
    
    conn = psycopg2.connect(**db_config)
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    try:
        print("Setting up sample database data...")
        
        # 1. Insert sample teams
        print("1. Creating teams...")
        teams_data = [
            (1, 'Real Madrid', '#FFFFFF', 'Real Madrid CF'),
            (2, 'Barcelona', '#A50044', 'FC Barcelona'),
            (3, 'PSG', '#004170', 'Paris Saint-Germain'),
            (4, 'Liverpool', '#C8102E', 'Liverpool FC')
        ]
        
        for team_id, team_name, team_color, full_name in teams_data:
            cursor.execute(
                """
                INSERT INTO teams (team_id, team_name, team_color, full_name)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (team_id) DO UPDATE SET
                    team_name = EXCLUDED.team_name,
                    team_color = EXCLUDED.team_color,
                    full_name = EXCLUDED.full_name
                """,
                (team_id, team_name, team_color, full_name)
            )
        
        # 2. Insert sample players
        print("2. Creating players...")
        players_data = [
            # Real Madrid (Team 1)
            (1, 1, 1, 'Thibaut Courtois', 'GK'),
            (2, 1, 2, 'Dani Carvajal', 'RB'),
            (3, 1, 3, 'Eder Militao', 'CB'),
            (4, 1, 4, 'David Alaba', 'CB'),
            (5, 1, 5, 'Ferland Mendy', 'LB'),
            (6, 1, 6, 'Casemiro', 'CDM'),
            (7, 1, 7, 'Luka Modric', 'CM'),
            (8, 1, 8, 'Toni Kroos', 'CM'),
            (9, 1, 9, 'Karim Benzema', 'ST'),
            (10, 1, 10, 'Luka Jovic', 'ST'),
            (11, 1, 11, 'Vinicius Jr', 'LW'),
            
            # Barcelona (Team 2)
            (12, 2, 1, 'Marc-Andre ter Stegen', 'GK'),
            (13, 2, 2, 'Sergino Dest', 'RB'),
            (14, 2, 3, 'Gerard Pique', 'CB'),
            (15, 2, 4, 'Clement Lenglet', 'CB'),
            (16, 2, 5, 'Jordi Alba', 'LB'),
            (17, 2, 6, 'Sergio Busquets', 'CDM'),
            (18, 2, 7, 'Pedri', 'CM'),
            (19, 2, 8, 'Frenkie de Jong', 'CM'),
            (20, 2, 9, 'Robert Lewandowski', 'ST'),
            (21, 2, 10, 'Ansu Fati', 'ST'),
            (22, 2, 11, 'Ousmane Dembele', 'RW'),
            
            # PSG (Team 3)
            (23, 3, 1, 'Gianluigi Donnarumma', 'GK'),
            (24, 3, 2, 'Achraf Hakimi', 'RB'),
            (25, 3, 3, 'Marquinhos', 'CB'),
            (26, 3, 4, 'Presnel Kimpembe', 'CB'),
            (27, 3, 5, 'Nuno Mendes', 'LB'),
            (28, 3, 6, 'Marco Verratti', 'CDM'),
            (29, 3, 7, 'Neymar Jr', 'LW'),
            (30, 3, 8, 'Kylian Mbappe', 'ST'),
            (31, 3, 9, 'Lionel Messi', 'RW'),
            (32, 3, 10, 'Angel Di Maria', 'CAM'),
            (33, 3, 11, 'Sergio Ramos', 'CB'),
            
            # Liverpool (Team 4)
            (34, 4, 1, 'Alisson Becker', 'GK'),
            (35, 4, 2, 'Trent Alexander-Arnold', 'RB'),
            (36, 4, 3, 'Virgil van Dijk', 'CB'),
            (37, 4, 4, 'Joel Matip', 'CB'),
            (38, 4, 5, 'Andrew Robertson', 'LB'),
            (39, 4, 6, 'Fabinho', 'CDM'),
            (40, 4, 7, 'Jordan Henderson', 'CM'),
            (41, 4, 8, 'Thiago Alcantara', 'CM'),
            (42, 4, 9, 'Roberto Firmino', 'ST'),
            (43, 4, 10, 'Sadio Mane', 'LW'),
            (44, 4, 11, 'Mohamed Salah', 'RW')
        ]
        
        for player_id, team_id, jersey_number, name, position in players_data:
            cursor.execute(
                """
                INSERT INTO players (player_id, team_id, jersey_number, name, position, is_active)
                VALUES (%s, %s, %s, %s, %s, TRUE)
                ON CONFLICT (player_id) DO UPDATE SET
                    team_id = EXCLUDED.team_id,
                    jersey_number = EXCLUDED.jersey_number,
                    name = EXCLUDED.name,
                    position = EXCLUDED.position,
                    is_active = EXCLUDED.is_active
                """,
                (player_id, team_id, jersey_number, name, position)
            )
        
        # 3. Insert sample match
        print("3. Creating match...")
        cursor.execute(
            """
            INSERT INTO matches (match_id, team_a_id, team_b_id, match_date, venue, competition, status)
            VALUES (1, 1, 2, %s, 'Santiago Bernabeu', 'La Liga', 'completed')
            ON CONFLICT (match_id) DO UPDATE SET
                team_a_id = EXCLUDED.team_a_id,
                team_b_id = EXCLUDED.team_b_id,
                match_date = EXCLUDED.match_date,
                venue = EXCLUDED.venue,
                competition = EXCLUDED.competition,
                status = EXCLUDED.status
            """,
            (datetime.now(),)
        )
        
        # 4. Insert formations
        print("4. Creating formations...")
        cursor.execute(
            """
            INSERT INTO formations (formation_id, match_id, team_id, formation_type, is_active)
            VALUES (1, 1, 1, '4-3-3', TRUE), (2, 1, 2, '4-4-2', TRUE)
            ON CONFLICT (formation_id) DO UPDATE SET
                match_id = EXCLUDED.match_id,
                team_id = EXCLUDED.team_id,
                formation_type = EXCLUDED.formation_type,
                is_active = EXCLUDED.is_active
            """
        )
        
        # 5. Insert formation positions
        print("5. Creating formation positions...")
        formation_positions = [
            # Real Madrid 4-3-3
            (1, 1, 1, 1, 'GK'),
            (2, 1, 2, 2, 'RB'),
            (3, 1, 3, 3, 'RCB'),
            (4, 1, 4, 4, 'LCB'),
            (5, 1, 5, 5, 'LB'),
            (6, 1, 6, 6, 'CDM'),
            (7, 1, 7, 7, 'RCM'),
            (8, 1, 8, 8, 'LCM'),
            (9, 1, 9, 9, 'ST'),
            (10, 1, 10, 10, 'ST'),
            (11, 1, 11, 11, 'LW'),
            
            # Barcelona 4-4-2
            (12, 2, 12, 1, 'GK'),
            (13, 2, 13, 2, 'RB'),
            (14, 2, 14, 3, 'RCB'),
            (15, 2, 15, 4, 'LCB'),
            (16, 2, 16, 5, 'LB'),
            (17, 2, 17, 6, 'RDM'),
            (18, 2, 18, 7, 'LDM'),
            (19, 2, 19, 8, 'RM'),
            (20, 2, 20, 9, 'ST'),
            (21, 2, 21, 10, 'ST'),
            (22, 2, 22, 11, 'LM')
        ]
        
        for fp_id, formation_id, player_id, jersey_number, tactical_position in formation_positions:
            cursor.execute(
                """
                INSERT INTO formation_positions (fp_id, formation_id, player_id, jersey_number, tactical_position)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (fp_id) DO UPDATE SET
                    formation_id = EXCLUDED.formation_id,
                    player_id = EXCLUDED.player_id,
                    jersey_number = EXCLUDED.jersey_number,
                    tactical_position = EXCLUDED.tactical_position
                """,
                (fp_id, formation_id, player_id, jersey_number, tactical_position)
            )
        
        conn.commit()
        print("✓ Sample database setup complete!")
        print("\nSample data created:")
        print("- 4 teams (Real Madrid, Barcelona, PSG, Liverpool)")
        print("- 44 players (11 per team)")
        print("- 1 match (Real Madrid vs Barcelona)")
        print("- 2 formations (4-3-3 vs 4-4-2)")
        print("- Formation positions for all players")
        
        # Show sample queries
        print("\nSample queries to test:")
        print("1. SELECT * FROM teams;")
        print("2. SELECT * FROM players WHERE team_id = 1;")
        print("3. SELECT * FROM matches;")
        print("4. SELECT * FROM formations WHERE match_id = 1;")
        
    except Exception as e:
        print(f"Error setting up database: {e}")
        conn.rollback()
        raise
    finally:
        cursor.close()
        conn.close()

def main():
    parser = argparse.ArgumentParser(description="Setup database with sample data")
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
    
    setup_database_sample(db_config)

if __name__ == "__main__":
    main()
