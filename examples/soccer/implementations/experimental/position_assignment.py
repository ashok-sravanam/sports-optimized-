import numpy as np
from typing import Dict, Set, Optional

class PositionBasedAssignmentManager:
    """
    Assign position numbers 1-11 based on player's spatial position on pitch.
    Uses pitch coordinates to determine which position each player occupies.
    """
    
    def __init__(self, db_manager):
        """
        Args:
            db_manager: Database manager instance (optional, can be None)
        """
        self.db = db_manager
        
        # tracker_id -> position_number (1-11)
        self.tracker_to_position = {}
        
        # Track which positions are filled per team
        if db_manager:
            self.team_positions_filled = {
                db_manager.team_a_id: set(),
                db_manager.team_b_id: set()
            }
        else:
            # Fallback for mock/testing
            self.team_positions_filled = {
                0: set(),
                1: set()
            }
        
        # Position names for logging
        self.POSITION_NAMES = {
            1: "GK",
            2: "RB",
            3: "LB",
            4: "CB_R",
            5: "CB_L",
            6: "CDM",
            7: "LW",
            8: "CM_R",
            9: "ST",
            10: "CM_L/CAM",
            11: "RW"
        }
        
        # Standard pitch dimensions (FIFA)
        self.PITCH_LENGTH = 105.0  # meters
        self.PITCH_WIDTH = 68.0    # meters
    
    def assign_position(self, tracker_id: int, classified_team: int,
                       pitch_x: float, pitch_y: float, confidence: float = 1.0) -> int:
        """
        Assign position number (1-11) based on player's location on pitch.
        FLEXIBLE: Can reassign if confidence is high enough and we have all 22 players.
        
        Args:
            tracker_id: ByteTrack tracker ID (can be any integer)
            classified_team: Team classification (0 or 1)
            pitch_x: X coordinate on pitch (0-105m, 0=own goal, 105=opp goal)
            pitch_y: Y coordinate on pitch (0-68m, 0=left, 68=right)
            confidence: Detection confidence (0-1)
        
        Returns:
            Position number (1-11)
        """
        
        # If already assigned and confidence is not high enough, keep existing
        if tracker_id in self.tracker_to_position and confidence < 0.8:
            existing = self.tracker_to_position[tracker_id]
            return existing
        
        # Get actual team_id
        if self.db:
            team_id = self.db.get_team_id_from_classification(classified_team)
        else:
            team_id = classified_team
        
        if team_id is None or team_id not in self.team_positions_filled:
            # Referee or unknown team - should not happen with proper filtering
            print(f"⚠ Invalid team for tracker {tracker_id}: {team_id}")
            return tracker_id
        
        # Normalize coordinates to 0-1 range
        norm_x = pitch_x / self.PITCH_LENGTH  # 0 = defensive, 1 = attacking
        norm_y = pitch_y / self.PITCH_WIDTH   # 0 = left, 1 = right
        
        # Log coordinates for first assignment
        print(f"  Assigning tracker {tracker_id} at pitch ({pitch_x:.1f}, {pitch_y:.1f}) "
              f"→ norm ({norm_x:.2f}, {norm_y:.2f})")
        
        # Determine position based on spatial location
        position_num = self._determine_position_from_location(norm_x, norm_y)
        
        print(f"  Initial position determination: {position_num} "
              f"({self.POSITION_NAMES.get(position_num, 'Unknown')})")
        
        # Check if position already filled
        if position_num in self.team_positions_filled[team_id]:
            print(f"  Position {position_num} already taken by team {team_id}")
            position_num = self._find_nearest_available(
                norm_x, norm_y, team_id, position_num
            )
            print(f"  Using fallback position: {position_num}")
        
        # Assign position
        self.tracker_to_position[tracker_id] = position_num
        self.team_positions_filled[team_id].add(position_num)
        
        # Log assignment
        pos_name = self.POSITION_NAMES.get(position_num, "Unknown")
        print(f"✓ ASSIGNED: Tracker {tracker_id} → Position #{position_num} ({pos_name})")
        
        # Register with database if available
        if self.db:
            self.db.assign_tracker_to_jersey(tracker_id, position_num, classified_team)
        
        return position_num
    
    def _determine_position_from_location(self, norm_x: float, norm_y: float) -> int:
        """
        STRICT position assignment based on pitch coordinates.
        
        Pitch orientation:
        - norm_x: 0.0 (defending/own goal) → 1.0 (attacking/opponent goal)
        - norm_y: 0.0 (left side) → 1.0 (right side)
        
        IMPORTANT: Use STRICT thresholds to avoid overlaps
        """
        
        # === GOALKEEPER ZONE (Very deep, near own goal) ===
        if norm_x < 0.15:
            return 1  # GK - must be very close to own goal
        
        # === DEFENDER ZONE (Deep third) ===
        elif norm_x < 0.30:
            # Left Back - deep, far left
            if norm_y < 0.25:
                return 3  # LB
            
            # Right Back - deep, far right
            elif norm_y > 0.75:
                return 2  # RB
            
            # Center Backs - deep, middle area
            elif norm_y < 0.45:
                return 5  # CB_L (center-left)
            else:
                return 4  # CB_R (center-right)
        
        # === MIDFIELD ZONE (Middle third) ===
        elif norm_x < 0.65:
            # Wide left midfielder
            if norm_y < 0.25:
                return 10  # CM_L or CAM (left side)
            
            # Wide right midfielder
            elif norm_y > 0.75:
                return 8   # CM_R (right side)
            
            # Central midfielder
            else:
                return 6   # CDM (central defensive mid)
        
        # === ATTACKING ZONE (Forward third) ===
        else:
            # Left winger - high up, left side
            if norm_y < 0.25:
                return 7   # LW
            
            # Right winger - high up, right side
            elif norm_y > 0.75:
                return 11  # RW
            
            # Striker - high up, central
            else:
                return 9   # ST
    
    def _find_nearest_available(self, norm_x: float, norm_y: float,
                                team_id: int, preferred: int) -> int:
        """
        Find nearest available position if preferred is already taken.
        
        Args:
            norm_x, norm_y: Normalized pitch coordinates
            team_id: Team ID
            preferred: Preferred position number that's taken
        
        Returns:
            Available position number
        """
        
        # Get available positions
        all_positions = set(range(1, 12))
        available = all_positions - self.team_positions_filled[team_id]
        
        if not available:
            # All positions filled (shouldn't happen with 11 players max)
            # Use tracker_id or increment
            return preferred + 100
        
        # Define position groups for fallback
        position_groups = {
            1: [1],                    # GK (unique)
            2: [2, 4],                 # RB can fall to CB_R
            3: [3, 5],                 # LB can fall to CB_L
            4: [4, 5, 2],              # CB_R can fall to CB_L or RB
            5: [5, 4, 3],              # CB_L can fall to CB_R or LB
            6: [6, 8, 10],             # CDM can fall to CM
            7: [7, 10, 3],             # LW can fall to CM_L or LB
            8: [8, 6, 10],             # CM_R can fall to CDM or CM_L
            9: [9, 7, 11],             # ST can fall to wingers
            10: [10, 8, 6],            # CM_L can fall to CM_R or CDM
            11: [11, 8, 2]             # RW can fall to CM_R or RB
        }
        
        # Try positions in order of preference
        for fallback in position_groups.get(preferred, [preferred]):
            if fallback in available:
                return fallback
        
        # Last resort: return any available
        return min(available)
    
    def get_position(self, tracker_id: int) -> int:
        """Get position number for tracker_id"""
        return self.tracker_to_position.get(tracker_id, tracker_id)
    
    def get_position_name(self, tracker_id: int) -> str:
        """Get position name (e.g., 'GK', 'ST', 'CM_R')"""
        position = self.get_position(tracker_id)
        return self.POSITION_NAMES.get(position, f"P{position}")
    
    def assign_goalkeeper(self, tracker_id: int, classified_team: int) -> int:
        """
        Force assign goalkeeper to position 1.
        """
        
        # Check if already assigned
        if tracker_id in self.tracker_to_position:
            return self.tracker_to_position[tracker_id]
        
        # Get team_id
        if self.db:
            team_id = self.db.get_team_id_from_classification(classified_team)
        else:
            team_id = classified_team
        
        if team_id is None or team_id not in self.team_positions_filled:
            return tracker_id
        
        # Check if position 1 is available
        if 1 not in self.team_positions_filled[team_id]:
            # Assign position 1
            self.tracker_to_position[tracker_id] = 1
            self.team_positions_filled[team_id].add(1)
            
            print(f"✓ Tracker {tracker_id} → Position #1 (GK)")
            
            if self.db:
                self.db.assign_tracker_to_jersey(tracker_id, 1, classified_team)
            
            return 1
        else:
            # Position 1 already taken - this shouldn't happen!
            print(f"⚠ WARNING: Position 1 (GK) already assigned for team {team_id}")
            return 1  # Force it anyway
    
    def reset_team_positions(self, team_id: int):
        """Reset position assignments for a team (e.g., after substitution)"""
        if team_id in self.team_positions_filled:
            self.team_positions_filled[team_id].clear()
            print(f"✓ Reset positions for team {team_id}")
    
    def reset_all_positions(self):
        """Reset all position assignments - useful when we have high confidence detections"""
        self.tracker_to_position.clear()
        for team_id in self.team_positions_filled:
            self.team_positions_filled[team_id].clear()
        print("✓ Reset all position assignments")
    
    def should_reassign_positions(self, high_confidence_count: int) -> bool:
        """
        Determine if we should reassign positions based on high confidence detections.
        
        Args:
            high_confidence_count: Number of high confidence detections (>0.8)
        
        Returns:
            True if we should reassign positions
        """
        # If we have 20+ high confidence detections, consider reassigning
        return high_confidence_count >= 20
