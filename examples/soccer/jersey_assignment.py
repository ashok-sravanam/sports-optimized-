"""
Jersey Number Assignment System
Allows manual assignment of jersey numbers to tracked players for tactical analysis.
"""

import cv2
import numpy as np
import supervision as sv
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import json

@dataclass
class JerseyAssignment:
    """Represents a jersey number assignment for a player"""
    tracker_id: int
    jersey_number: int
    team_id: int
    confidence: float = 1.0
    assigned_at_frame: int = 0

class JerseyAssignmentManager:
    """Manages jersey number assignments for tracked players"""
    
    def __init__(self):
        self.assignments: Dict[int, JerseyAssignment] = {}  # tracker_id -> assignment
        self.team_colors = {
            0: (255, 20, 147),  # Deep Pink - Team 1
            1: (0, 191, 255),   # Deep Sky Blue - Team 2
            2: (255, 99, 71),   # Tomato - Team 3
            3: (255, 215, 0),   # Gold - Team 4
        }
        self.assignment_mode = False
        self.selected_tracker_id = None
        self.current_team_id = 0
        self.jersey_input = ""
        
    def assign_jersey(self, tracker_id: int, jersey_number: int, team_id: int, 
                     frame_idx: int, confidence: float = 1.0):
        """Assign jersey number to a tracked player"""
        assignment = JerseyAssignment(
            tracker_id=tracker_id,
            jersey_number=jersey_number,
            team_id=team_id,
            confidence=confidence,
            assigned_at_frame=frame_idx
        )
        self.assignments[tracker_id] = assignment
        print(f"✓ Assigned jersey #{jersey_number} (Team {team_id}) to tracker {tracker_id}")
    
    def get_jersey_number(self, tracker_id: int) -> Optional[int]:
        """Get jersey number for a tracker ID"""
        if tracker_id in self.assignments:
            return self.assignments[tracker_id].jersey_number
        return None
    
    def get_team_id(self, tracker_id: int) -> Optional[int]:
        """Get team ID for a tracker ID"""
        if tracker_id in self.assignments:
            return self.assignments[tracker_id].team_id
        return None
    
    def get_assignment(self, tracker_id: int) -> Optional[JerseyAssignment]:
        """Get full assignment for a tracker ID"""
        return self.assignments.get(tracker_id)
    
    def remove_assignment(self, tracker_id: int):
        """Remove jersey assignment for a tracker"""
        if tracker_id in self.assignments:
            del self.assignments[tracker_id]
            print(f"✓ Removed assignment for tracker {tracker_id}")
    
    def get_assignments_by_team(self, team_id: int) -> List[JerseyAssignment]:
        """Get all assignments for a specific team"""
        return [assignment for assignment in self.assignments.values() 
                if assignment.team_id == team_id]
    
    def get_assigned_jersey_numbers(self, team_id: int) -> List[int]:
        """Get all assigned jersey numbers for a team"""
        return [assignment.jersey_number for assignment in self.get_assignments_by_team(team_id)]
    
    def is_jersey_assigned(self, team_id: int, jersey_number: int) -> bool:
        """Check if a jersey number is already assigned for a team"""
        return jersey_number in self.get_assigned_jersey_numbers(team_id)
    
    def save_assignments(self, filepath: str):
        """Save assignments to JSON file"""
        data = {
            'assignments': {
                str(tracker_id): {
                    'jersey_number': assignment.jersey_number,
                    'team_id': assignment.team_id,
                    'confidence': assignment.confidence,
                    'assigned_at_frame': assignment.assigned_at_frame
                }
                for tracker_id, assignment in self.assignments.items()
            }
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"✓ Saved assignments to {filepath}")
    
    def load_assignments(self, filepath: str):
        """Load assignments from JSON file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self.assignments.clear()
            for tracker_id_str, assignment_data in data['assignments'].items():
                tracker_id = int(tracker_id_str)
                assignment = JerseyAssignment(
                    tracker_id=tracker_id,
                    jersey_number=assignment_data['jersey_number'],
                    team_id=assignment_data['team_id'],
                    confidence=assignment_data.get('confidence', 1.0),
                    assigned_at_frame=assignment_data.get('assigned_at_frame', 0)
                )
                self.assignments[tracker_id] = assignment
            
            print(f"✓ Loaded {len(self.assignments)} assignments from {filepath}")
        except Exception as e:
            print(f"✗ Failed to load assignments: {e}")
    
    def draw_assignment_interface(self, frame: np.ndarray, detections: sv.Detections, 
                                 frame_idx: int) -> np.ndarray:
        """Draw jersey assignment interface on frame"""
        annotated_frame = frame.copy()
        h, w = frame.shape[:2]
        
        # Draw assignment mode indicator
        if self.assignment_mode:
            cv2.rectangle(annotated_frame, (10, 10), (400, 100), (0, 255, 0), 2)
            cv2.putText(annotated_frame, "JERSEY ASSIGNMENT MODE", (20, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Team: {self.current_team_id}", (20, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(annotated_frame, f"Jersey: {self.jersey_input}", (20, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw player bounding boxes with jersey numbers
        if len(detections) > 0:
            for i, tracker_id in enumerate(detections.tracker_id):
                if tracker_id is None:
                    continue
                
                # Get bounding box
                x1, y1, x2, y2 = detections.xyxy[i].astype(int)
                
                # Get team and jersey info
                team_id = self.get_team_id(tracker_id)
                jersey_number = self.get_jersey_number(tracker_id)
                
                # Choose color
                if team_id is not None:
                    color = self.team_colors.get(team_id, (255, 255, 255))
                else:
                    color = (128, 128, 128)  # Gray for unassigned
                
                # Draw bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw jersey number or tracker ID
                if jersey_number is not None:
                    label = f"#{jersey_number}"
                    if team_id is not None:
                        label += f" (T{team_id})"
                else:
                    label = f"ID:{tracker_id}"
                
                # Draw label background
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(annotated_frame, (x1, y1 - 25), (x1 + label_size[0] + 10, y1), color, -1)
                cv2.putText(annotated_frame, label, (x1 + 5, y1 - 8), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Highlight selected player
                if self.selected_tracker_id == tracker_id:
                    cv2.rectangle(annotated_frame, (x1-5, y1-5), (x2+5, y2+5), (0, 255, 255), 3)
        
        # Draw instructions
        instructions = [
            "CONTROLS:",
            "A - Toggle assignment mode",
            "1-4 - Select team",
            "Click player to select",
            "0-9 - Enter jersey number",
            "ENTER - Assign jersey",
            "ESC - Cancel assignment",
            "S - Save assignments",
            "L - Load assignments"
        ]
        
        y_offset = h - 200
        for i, instruction in enumerate(instructions):
            cv2.putText(annotated_frame, instruction, (w - 300, y_offset + i * 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return annotated_frame
    
    def handle_key_press(self, key: int, detections: sv.Detections) -> bool:
        """Handle keyboard input for jersey assignment"""
        if key == ord('a') or key == ord('A'):
            self.assignment_mode = not self.assignment_mode
            if not self.assignment_mode:
                self.selected_tracker_id = None
                self.jersey_input = ""
            print(f"Assignment mode: {'ON' if self.assignment_mode else 'OFF'}")
            return True
        
        elif key >= ord('1') and key <= ord('4'):
            if self.assignment_mode:
                self.current_team_id = key - ord('1')
                print(f"Selected team: {self.current_team_id}")
                return True
        
        elif key >= ord('0') and key <= ord('9'):
            if self.assignment_mode and self.selected_tracker_id is not None:
                self.jersey_input += chr(key)
                print(f"Jersey input: {self.jersey_input}")
                return True
        
        elif key == 13:  # Enter
            if self.assignment_mode and self.selected_tracker_id is not None and self.jersey_input:
                try:
                    jersey_number = int(self.jersey_input)
                    if not self.is_jersey_assigned(self.current_team_id, jersey_number):
                        self.assign_jersey(self.selected_tracker_id, jersey_number, 
                                         self.current_team_id, 0)  # frame_idx will be updated
                        self.jersey_input = ""
                        self.selected_tracker_id = None
                    else:
                        print(f"Jersey #{jersey_number} already assigned to team {self.current_team_id}")
                except ValueError:
                    print("Invalid jersey number")
                return True
        
        elif key == 27:  # ESC
            if self.assignment_mode:
                self.selected_tracker_id = None
                self.jersey_input = ""
                print("Assignment cancelled")
                return True
        
        elif key == ord('s') or key == ord('S'):
            self.save_assignments("jersey_assignments.json")
            return True
        
        elif key == ord('l') or key == ord('L'):
            self.load_assignments("jersey_assignments.json")
            return True
        
        return False
    
    def handle_mouse_click(self, x: int, y: int, detections: sv.Detections) -> bool:
        """Handle mouse click to select player for jersey assignment"""
        if not self.assignment_mode or len(detections) == 0:
            return False
        
        # Find clicked detection
        for i, tracker_id in enumerate(detections.tracker_id):
            if tracker_id is None:
                continue
            
            x1, y1, x2, y2 = detections.xyxy[i].astype(int)
            if x1 <= x <= x2 and y1 <= y <= y2:
                self.selected_tracker_id = tracker_id
                self.jersey_input = ""
                print(f"Selected player with tracker ID: {tracker_id}")
                return True
        
        return False

def create_jersey_assignment_callback(manager: JerseyAssignmentManager):
    """Create callback function for jersey assignment during video processing"""
    def callback(frame: np.ndarray, detections: sv.Detections, frame_idx: int) -> np.ndarray:
        return manager.draw_assignment_interface(frame, detections, frame_idx)
    
    return callback

# Utility functions for integration with existing analysis
def get_jersey_annotated_detections(detections: sv.Detections, 
                                  manager: JerseyAssignmentManager) -> sv.Detections:
    """Create detections with jersey number labels"""
    if len(detections) == 0:
        return detections
    
    # Create labels with jersey numbers
    labels = []
    for tracker_id in detections.tracker_id:
        if tracker_id is None:
            labels.append("Unknown")
        else:
            jersey_number = manager.get_jersey_number(tracker_id)
            team_id = manager.get_team_id(tracker_id)
            if jersey_number is not None:
                labels.append(f"#{jersey_number} (T{team_id})")
            else:
                labels.append(f"ID:{tracker_id}")
    
    # Create new detections with labels
    annotated_detections = detections.copy()
    annotated_detections.labels = labels
    return annotated_detections

def filter_detections_by_team(detections: sv.Detections, manager: JerseyAssignmentManager, 
                            team_id: int) -> sv.Detections:
    """Filter detections to only include players from specified team"""
    if len(detections) == 0:
        return detections
    
    team_mask = []
    for tracker_id in detections.tracker_id:
        if tracker_id is None:
            team_mask.append(False)
        else:
            player_team_id = manager.get_team_id(tracker_id)
            team_mask.append(player_team_id == team_id)
    
    team_mask = np.array(team_mask)
    return detections[team_mask] if team_mask.any() else sv.Detections.empty()
