class DynamicFormationManager:
    """
    Manage formations with user input or command-line arguments.
    NO hardcoded formations.
    """
    
    def __init__(self):
        self.formations = {
            0: None,  # Team A formation
            1: None   # Team B formation
        }
        self.team_names = {
            0: "Team A",
            1: "Team B"
        }
    
    def set_formation(self, team_idx: int, formation: str, team_name: str = None):
        """
        Set formation for a team.
        
        Args:
            team_idx: 0 or 1
            formation: Formation string (e.g., "4-3-3", "4-4-2", "3-5-2")
            team_name: Optional team name
        """
        self.formations[team_idx] = formation
        
        if team_name:
            self.team_names[team_idx] = team_name
        
        print(f"✓ {self.team_names[team_idx]} formation set: {formation}")
    
    def get_formation(self, team_idx: int) -> str:
        """Get formation for team"""
        return self.formations.get(team_idx, "Unknown")
    
    def get_team_name(self, team_idx: int) -> str:
        """Get team name"""
        return self.team_names.get(team_idx, f"Team {team_idx}")
    
    def prompt_interactive(self):
        """
        Interactive prompt for formations.
        Call this if no formations provided via command-line.
        """
        print("\n" + "="*70)
        print("FORMATION INPUT (or press Enter for defaults)")
        print("="*70)
        
        # Team A
        team_a_name = input("Team A name [Team A]: ").strip()
        if not team_a_name:
            team_a_name = "Team A"
        
        team_a_formation = input(f"{team_a_name} formation [4-3-3]: ").strip()
        if not team_a_formation:
            team_a_formation = "4-3-3"
        
        # Team B
        team_b_name = input("Team B name [Team B]: ").strip()
        if not team_b_name:
            team_b_name = "Team B"
        
        team_b_formation = input(f"{team_b_name} formation [4-4-2]: ").strip()
        if not team_b_formation:
            team_b_formation = "4-4-2"
        
        # Set formations
        self.set_formation(0, team_a_formation, team_a_name)
        self.set_formation(1, team_b_formation, team_b_name)
        
        print("="*70 + "\n")
