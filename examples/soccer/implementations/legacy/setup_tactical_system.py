"""
Setup script for the tactical analysis system.
Installs dependencies, sets up database, and creates initial data.
"""

import os
import sys
import subprocess
from database import create_database_if_not_exists, setup_database_schema, DatabaseConfig

def install_requirements():
    """Install additional requirements for tactical analysis"""
    print("Installing tactical analysis requirements...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "tactical_requirements.txt"
        ])
        print("✓ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install requirements: {e}")
        return False

def setup_database():
    """Setup PostgreSQL database"""
    print("Setting up database...")
    
    # Get database configuration from user
    print("\nDatabase Configuration:")
    host = input("Database host [localhost]: ").strip() or "localhost"
    port = input("Database port [5432]: ").strip() or "5432"
    database = input("Database name [soccer_analysis]: ").strip() or "soccer_analysis"
    user = input("Database user [postgres]: ").strip() or "postgres"
    password = input("Database password: ").strip()
    
    if not password:
        print("✗ Database password is required")
        return False
    
    config = DatabaseConfig(
        host=host,
        port=int(port),
        database=database,
        user=user,
        password=password
    )
    
    # Create database if it doesn't exist
    if not create_database_if_not_exists(config):
        print("✗ Failed to create database")
        return False
    
    # Setup schema
    if not setup_database_schema(config):
        print("✗ Failed to setup database schema")
        return False
    
    print("✓ Database setup completed successfully")
    return True

def create_sample_data():
    """Create sample data for testing"""
    print("Creating sample data...")
    
    from database import SoccerDatabase, DatabaseConfig
    
    config = DatabaseConfig()
    db = SoccerDatabase(config)
    
    if not db.connect():
        print("✗ Failed to connect to database")
        return False
    
    try:
        # Check if data already exists
        teams = db.get_teams()
        if len(teams) > 0:
            print("✓ Sample data already exists")
            return True
        
        # Create sample teams
        real_madrid_id = db.create_team("Real Madrid", "#FFFFFF")
        getafe_id = db.create_team("Getafe", "#0066CC")
        psg_id = db.create_team("PSG", "#004170")
        liverpool_id = db.create_team("Liverpool", "#C8102E")
        
        # Create sample match
        match_id = db.create_match(
            team_a_id=real_madrid_id,
            team_b_id=getafe_id,
            venue="Santiago Bernabeu",
            competition="La Liga"
        )
        
        # Create sample players for Real Madrid
        real_madrid_players = [
            (1, "Courtois", "GK"),
            (2, "Carvajal", "RB"),
            (4, "Ramos", "CB"),
            (5, "Varane", "CB"),
            (12, "Marcelo", "LB"),
            (8, "Kroos", "CM"),
            (10, "Modric", "CM"),
            (14, "Casemiro", "CDM"),
            (7, "Hazard", "LW"),
            (9, "Benzema", "ST"),
            (11, "Vinicius", "RW")
        ]
        
        for jersey_number, name, position in real_madrid_players:
            db.create_player(real_madrid_id, jersey_number, name, position)
        
        # Create sample players for Getafe
        getafe_players = [
            (1, "Soria", "GK"),
            (2, "Dakonam", "RB"),
            (4, "Cabrera", "CB"),
            (6, "Djene", "CB"),
            (17, "Olivera", "LB"),
            (8, "Arambarri", "CM"),
            (20, "Maksimovic", "CM"),
            (15, "Cucurella", "LW"),
            (7, "Mata", "ST"),
            (11, "Unal", "ST"),
            (19, "Nyom", "RW")
        ]
        
        for jersey_number, name, position in getafe_players:
            db.create_player(getafe_id, jersey_number, name, position)
        
        print("✓ Sample data created successfully")
        return True
        
    except Exception as e:
        print(f"✗ Failed to create sample data: {e}")
        return False
    finally:
        db.disconnect()

def main():
    """Main setup function"""
    print("=" * 60)
    print("SOCCER TACTICAL ANALYSIS SYSTEM SETUP")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not os.path.exists("tactical_analysis.py"):
        print("✗ Please run this script from the soccer examples directory")
        return False
    
    # Install requirements
    if not install_requirements():
        return False
    
    # Setup database
    if not setup_database():
        return False
    
    # Create sample data
    if not create_sample_data():
        return False
    
    print("\n" + "=" * 60)
    print("SETUP COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Run tactical analysis:")
    print("   python tactical_analysis.py --source_video_path your_video.mp4 --target_video_path output.mp4")
    print("\n2. Run interactive analysis:")
    print("   python tactical_analysis.py --source_video_path your_video.mp4 --target_video_path output.mp4 --mode interactive")
    print("\n3. Assign jersey numbers during analysis using the interface")
    print("4. View tracked positions in the database")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
