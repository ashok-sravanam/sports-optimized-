#!/usr/bin/env python3
"""
Soccer Analysis - Position-Based Tracking (1-11) with Unified Ball Data
"""

import argparse
import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO

from sports.annotators.soccer import draw_pitch
from sports.common.ball import BallTracker, BallAnnotator
from sports.common.team import TeamClassifier
from sports.common.view import ViewTransformer
from sports.configs.soccer import SoccerPitchConfiguration

# Import custom classes
from position_assignment import PositionBasedAssignmentManager
from unified_exporter import UnifiedTrackingExporter
from formation_manager import DynamicFormationManager

CONFIG = SoccerPitchConfiguration()

def resolve_goalkeepers_team_id(
    players: sv.Detections,
    players_team_id: np.array,
    goalkeepers: sv.Detections
) -> np.ndarray:
    """
    Resolve the team IDs for detected goalkeepers based on the proximity to team
    centroids.
    """
    if len(goalkeepers) == 0:
        return np.array([])
    
    goalkeepers_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    
    if len(players_xy) == 0:
        # No players to determine centroids, assign goalkeepers to team 0
        return np.array([0] * len(goalkeepers))
    
    team_0_centroid = players_xy[players_team_id == 0].mean(axis=0)
    team_1_centroid = players_xy[players_team_id == 1].mean(axis=0)
    
    goalkeepers_team_id = []
    for goalkeeper_xy in goalkeepers_xy:
        dist_0 = np.linalg.norm(goalkeeper_xy - team_0_centroid)
        dist_1 = np.linalg.norm(goalkeeper_xy - team_1_centroid)
        goalkeepers_team_id.append(0 if dist_0 < dist_1 else 1)
    
    return np.array(goalkeepers_team_id)


def assign_jersey_number(tracker_id: int, classified_team: int, 
                        tracker_to_jersey: dict, team_jersey_counts: dict) -> int:
    """
    Assign jersey number (1-15) to player based on tracker_id and team.
    Each player gets a CONSISTENT jersey number throughout the video.
    """
    # If already assigned, return existing jersey number
    if tracker_id in tracker_to_jersey:
        return tracker_to_jersey[tracker_id]
    
    # Assign new jersey number (1-15 per team)
    team_jersey_counts[classified_team] += 1
    jersey_num = team_jersey_counts[classified_team]
    
    # Store assignment
    tracker_to_jersey[tracker_id] = jersey_num
    
    print(f"✓ Jersey #{jersey_num} assigned to tracker {tracker_id} (team {classified_team})")
    
    return jersey_num


def filter_referees_from_players(
    players: sv.Detections,
    players_team_id: np.array,
    frame_shape: tuple
) -> tuple:
    """
    Filter out referees from player detections based on position and characteristics.
    Referees are typically:
    - Near the sidelines (not in the center of the field)
    - Not clustered with team players
    - Often in different colored uniforms
    """
    if len(players) == 0:
        return players, players_team_id
    
    frame_h, frame_w = frame_shape[:2]
    players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    
    # Calculate field boundaries (approximate)
    field_center_x = frame_w // 2
    field_center_y = frame_h // 2
    
    # Filter out detections that are too far from field center (likely referees)
    # Referees are typically near sidelines or outside main field area
    valid_indices = []
    
    for i, (x, y) in enumerate(players_xy):
        # Check if detection is within reasonable field bounds
        # Allow some margin for players near sidelines
        margin_x = frame_w * 0.15  # 15% margin from sides
        margin_y = frame_h * 0.1   # 10% margin from top/bottom
        
        if (margin_x <= x <= frame_w - margin_x and 
            margin_y <= y <= frame_h - margin_y):
            valid_indices.append(i)
    
    if len(valid_indices) == 0:
        return players, players_team_id
    
    # Filter detections and team IDs
    filtered_players = players[valid_indices]
    filtered_team_ids = players_team_id[valid_indices]
    
    return filtered_players, filtered_team_ids


def process_video_final(
    source_path: str,
    target_path: str,
    device: str = 'cpu',
    team_a_formation: str = None,
    team_b_formation: str = None,
    team_a_name: str = None,
    team_b_name: str = None,
    max_frames: int = None,
    confidence_threshold: float = 0.65
):
    """
    Final video processing with:
    - Position-based player IDs (1-11)
    - Unified player+ball tracking
    - Dynamic formations
    - No player names
    """
    
    # Setup formations
    formation_mgr = DynamicFormationManager()
    
    if team_a_formation and team_b_formation:
        formation_mgr.set_formation(0, team_a_formation, team_a_name)
        formation_mgr.set_formation(1, team_b_formation, team_b_name)
    else:
        formation_mgr.prompt_interactive()
    
    # Get team names and formations
    team_a_name = formation_mgr.get_team_name(0)
    team_b_name = formation_mgr.get_team_name(1)
    formation_a = formation_mgr.get_formation(0)
    formation_b = formation_mgr.get_formation(1)
    
    print(f"\n{'='*70}")
    print(f"MATCH: {team_a_name} ({formation_a}) vs {team_b_name} ({formation_b})")
    print(f"{'='*70}\n")
    
    # Load models
    print("Loading models...")
    PLAYER_MODEL_PATH = 'data/football-player-detection.pt'
    PITCH_MODEL_PATH = 'data/football-pitch-detection.pt'
    BALL_MODEL_PATH = 'data/football-ball-detection.pt'
    
    player_model = YOLO(PLAYER_MODEL_PATH).to(device=device)
    pitch_model = YOLO(PITCH_MODEL_PATH).to(device=device)
    ball_model = YOLO(BALL_MODEL_PATH).to(device=device)
    
    # Initialize trackers
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    ball_tracker = BallTracker(buffer_size=20)
    ball_annotator = BallAnnotator(radius=6, buffer_size=10)
    
    # Ball detection slicer
    def ball_callback(image_slice: np.ndarray) -> sv.Detections:
        result = ball_model(image_slice, imgsz=640, verbose=False)[0]
        return sv.Detections.from_ultralytics(result)
    
    slicer = sv.InferenceSlicer(callback=ball_callback, slice_wh=(640, 640))
    
    # Train team classifier
    print("Training team classifier...")
    team_classifier = TeamClassifier(device=device)
    
    STRIDE = 60
    PLAYER_CLASS_ID = 2
    
    frame_generator = sv.get_video_frames_generator(source_path=source_path, stride=STRIDE)
    crops = []
    
    for frame in frame_generator:
        result = player_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        players = detections[detections.class_id == PLAYER_CLASS_ID]
        crops.extend([sv.crop_image(frame, xyxy) for xyxy in players.xyxy])
        if len(crops) > 500:
            break
    
    team_classifier.fit(crops)
    
    # Initialize managers - use simple jersey assignment like working version
    exporter = UnifiedTrackingExporter(output_dir="tracking_data")
    
    # Simple jersey assignment: tracker_id -> jersey_number (1-15 per team)
    tracker_to_jersey = {}
    team_jersey_counts = {0: 0, 1: 0}  # Track jersey numbers per team
    
    # Setup video
    video_info = sv.VideoInfo.from_video_path(source_path)
    
    video_w, video_h = video_info.width, video_info.height
    board_w, board_h = 1200, 800
    
    if video_h != board_h:
        aspect = video_w / video_h
        video_h = board_h
        video_w = int(board_h * aspect)
    
    output_w = video_w + board_w
    output_h = max(video_h, board_h)
    
    # Set correct total frames for output video
    output_total_frames = min(max_frames, video_info.total_frames) if max_frames else video_info.total_frames
    
    output_video_info = sv.VideoInfo(
        width=output_w,
        height=output_h,
        fps=video_info.fps,
        total_frames=output_total_frames
    )
    
    # Colors
    TEAM_A_COLOR = (255, 100, 100)  # Light blue
    TEAM_B_COLOR = (100, 100, 255)  # Light red
    
    # Annotators
    COLORS = ['#6495ED', '#FF6B6B', '#FFD700']
    ellipse_annotator = sv.EllipseAnnotator(
        color=sv.ColorPalette.from_hex(COLORS), thickness=2
    )
    label_annotator = sv.LabelAnnotator(
        color=sv.ColorPalette.from_hex(COLORS),
        text_color=sv.Color.BLACK,
        text_position=sv.Position.BOTTOM_CENTER
    )
    
    print("Processing video...")
    
    # Set total frames for progress bar
    total_frames_to_process = min(max_frames, video_info.total_frames) if max_frames else video_info.total_frames
    
    frame_generator = sv.get_video_frames_generator(source_path=source_path)
    
    GOALKEEPER_CLASS_ID = 1
    REFEREE_CLASS_ID = 100  # Use 100 to avoid confusion with player classes
    
    with sv.VideoSink(target_path, output_video_info) as sink:
        frame_idx = 0
        
        for frame in tqdm(frame_generator, total=total_frames_to_process):
            
            # Check max_frames limit
            if max_frames and frame_idx >= max_frames:
                break
            
            # Resize frame
            if frame.shape[0] != video_h or frame.shape[1] != video_w:
                frame = cv2.resize(frame, (video_w, video_h))
            
            timestamp_sec = frame_idx / video_info.fps
            
            # Detection & tracking
            player_result = player_model(frame, imgsz=1280, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(player_result)
            detections = tracker.update_with_detections(detections)
            
            ball_detections = slicer(frame).with_nms(threshold=0.1)
            ball_detections = ball_tracker.update(ball_detections)
            
            pitch_result = pitch_model(frame, verbose=False)[0]
            keypoints = sv.KeyPoints.from_ultralytics(pitch_result)
            
            # Team classification
            players = detections[detections.class_id == PLAYER_CLASS_ID]
            goalkeepers = detections[detections.class_id == GOALKEEPER_CLASS_ID]
            referees = detections[detections.class_id == REFEREE_CLASS_ID]
            
            # Debug: Check what class IDs are actually detected
            if frame_idx == 0:  # Only print on first frame
                print(f"Debug - Detection counts:")
                print(f"  Players (class {PLAYER_CLASS_ID}): {len(players)}")
                print(f"  Goalkeepers (class {GOALKEEPER_CLASS_ID}): {len(goalkeepers)}")
                print(f"  Referees (class {REFEREE_CLASS_ID}): {len(referees)}")
                print(f"  Total detections: {len(detections)}")
                if len(detections) > 0:
                    unique_classes = np.unique(detections.class_id)
                    print(f"  Unique class IDs detected: {unique_classes}")
                    
                    # Check if we have any detections that might be referees
                    all_players = detections[detections.class_id == PLAYER_CLASS_ID]
                    if len(all_players) > 0:
                        print(f"  All 'players' detected: {len(all_players)}")
                        # These might include referees misclassified as players
            
            if len(players) > 0:
                player_crops = [sv.crop_image(frame, xyxy) for xyxy in players.xyxy]
                players_team_id = team_classifier.predict(player_crops)
                
                # Filter out referees from player detections
                players, players_team_id = filter_referees_from_players(
                    players, players_team_id, frame.shape
                )
            else:
                players_team_id = np.array([])
            
            if len(goalkeepers) > 0:
                goalkeepers_team_id = resolve_goalkeepers_team_id(
                    players, players_team_id, goalkeepers)
            else:
                goalkeepers_team_id = np.array([])
            
            # Merge detections
            all_detections = sv.Detections.merge([players, goalkeepers, referees])
            classified_teams = np.concatenate([
                players_team_id,
                goalkeepers_team_id,
                np.array([2] * len(referees))  # Referees = team 2 (like working version)
            ]) if len(all_detections) > 0 else np.array([])
            
            # CRITICAL: Mark which detections are goalkeepers
            is_goalkeeper = np.concatenate([
                np.array([False] * len(players)),
                np.array([True] * len(goalkeepers)),
                np.array([False] * len(referees))
            ]) if len(all_detections) > 0 else np.array([])
            
            # Coordinate transformation
            mask = (keypoints.xy[0][:, 0] > 1) & (keypoints.xy[0][:, 1] > 1)
            transformer = None
            
            if mask.any() and mask.sum() >= 4:
                try:
                    transformer = ViewTransformer(
                        source=keypoints.xy[0][mask].astype(np.float32),
                        target=np.array(CONFIG.vertices)[mask].astype(np.float32)
                    )
                except:
                    pass
            
            scale_x = board_w / CONFIG.length
            scale_y = board_h / CONFIG.width
            
            # Process players - JERSEY ASSIGNMENT (like working version)
            for i in range(len(all_detections)):
                tracker_id = all_detections.tracker_id[i]
                classified_team = classified_teams[i]
                
                # SKIP REFEREES COMPLETELY - DO NOT PROCESS AT ALL
                if classified_team == 2:  # Referees = team 2
                    continue
                
                # SKIP if not a valid team
                if classified_team not in [0, 1]:
                    continue
                
                # Get confidence
                confidence = float(all_detections.confidence[i]) \
                            if hasattr(all_detections, 'confidence') else 1.0
                
                # SKIP LOW CONFIDENCE DETECTIONS - DO NOT STORE
                if confidence < confidence_threshold:
                    continue
                
                video_xy = all_detections.get_anchors_coordinates(
                    anchor=sv.Position.BOTTOM_CENTER
                )[i]
                
                if transformer:
                    try:
                        pitch_xy = transformer.transform_points(
                            points=video_xy.reshape(1, -1)
                        )[0]
                        
                        # ASSIGN JERSEY NUMBER (1-15 per team) - CONSISTENT throughout video
                        jersey_num = assign_jersey_number(
                            tracker_id, classified_team, 
                            tracker_to_jersey, team_jersey_counts
                        )
                        
                        # Calculate board coordinates
                        board_xy = pitch_xy.copy()
                        board_xy[0] *= scale_x
                        board_xy[1] *= scale_y
                        
                        # Clip to bounds
                        board_xy[0] = np.clip(board_xy[0], 30, board_w - 30)
                        board_xy[1] = np.clip(board_xy[1], 30, board_h - 30)
                        
                        # Export player data (only high confidence)
                        exporter.add_player(
                            frame_idx, timestamp_sec, tracker_id,
                            jersey_num, classified_team,
                            float(video_xy[0]), float(video_xy[1]),
                            float(pitch_xy[0]), float(pitch_xy[1]),
                            float(board_xy[0]), float(board_xy[1]),
                            confidence
                        )
                    
                    except Exception as e:
                        print(f"Jersey assignment error: {e}")
            
            # Process ball
            if ball_detections is not None and len(ball_detections) > 0 and transformer:
                try:
                    ball_xy = ball_detections.get_anchors_coordinates(
                        anchor=sv.Position.BOTTOM_CENTER
                    )[0]
                    
                    ball_pitch = transformer.transform_points(
                        points=ball_xy.reshape(1, -1)
                    )[0]
                    
                    ball_board = ball_pitch.copy()
                    ball_board[0] *= scale_x
                    ball_board[1] *= scale_y
                    
                    # Clip ball to bounds
                    ball_board[0] = np.clip(ball_board[0], 30, board_w - 30)
                    ball_board[1] = np.clip(ball_board[1], 30, board_h - 30)
                    
                    ball_confidence = float(ball_detections.confidence[0]) \
                                     if hasattr(ball_detections, 'confidence') else 1.0
                    
                    # Export ball data
                    exporter.add_ball(
                        frame_idx, timestamp_sec,
                        float(ball_xy[0]), float(ball_xy[1]),
                        float(ball_pitch[0]), float(ball_pitch[1]),
                        float(ball_board[0]), float(ball_board[1]),
                        ball_confidence
                    )
                
                except Exception as e:
                    print(f"Ball tracking error: {e}")
            
            # Annotate video (LEFT SIDE) - NO NUMBERS, just bounding boxes
            annotated_video = frame.copy()
            
            if len(all_detections) > 0:
                # Show only bounding boxes, NO labels on live camera
                annotated_video = ellipse_annotator.annotate(annotated_video, all_detections)
            
            annotated_video = ball_annotator.annotate(annotated_video, ball_detections)
            
            # Add title
            cv2.putText(annotated_video, f"{team_a_name} vs {team_b_name}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Draw tactical board (RIGHT SIDE)
            tactical_board = draw_tactical_board_final(
                all_detections, classified_teams, ball_detections,
                keypoints, tracker_to_jersey, transformer,
                TEAM_A_COLOR, TEAM_B_COLOR,
                formation_a, formation_b,
                team_a_name, team_b_name,
                board_w, board_h, scale_x, scale_y
            )
            
            # Combine side-by-side
            output_frame = np.hstack([annotated_video, tactical_board])
            
            sink.write_frame(output_frame)
            frame_idx += 1
    
    # Export all data
    exporter.export_all()
    
    print(f"\n✓ Video processing complete: {target_path}")


def draw_tactical_board_final(
    detections, classified_teams, ball_detections,
    keypoints, tracker_to_jersey, transformer,
    team_a_color, team_b_color,
    formation_a, formation_b,
    team_a_name, team_b_name,
    board_w, board_h, scale_x, scale_y
):
    """Draw tactical board with jersey numbers"""
    
    # Referees = team 2 (like working version)
    
    pitch = draw_pitch(config=CONFIG)
    tactical_board = cv2.resize(pitch, (board_w, board_h), 
                                interpolation=cv2.INTER_LANCZOS4)
    
    if transformer and len(detections) > 0:
        try:
            player_xy = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
            transformed_xy = transformer.transform_points(points=player_xy)
            
            board_positions = transformed_xy.copy()
            board_positions[:, 0] *= scale_x
            board_positions[:, 1] *= scale_y
            
            # Draw players
            for i in range(len(detections)):
                tracker_id = detections.tracker_id[i]
                classified_team = classified_teams[i]
                
                # SKIP REFEREES - DO NOT DRAW ON TACTICAL BOARD
                if classified_team == 2:  # Referees = team 2
                    continue
                
                # SKIP INVALID TEAMS
                if classified_team not in [0, 1]:
                    continue
                
                x, y = board_positions[i]
                
                # Clip to bounds
                x = np.clip(x, 30, board_w - 30)
                y = np.clip(y, 30, board_h - 30)
                
                # Get color
                if classified_team == 0:
                    color = team_a_color
                elif classified_team == 1:
                    color = team_b_color
                else:
                    continue  # Should never reach here
                
                # Get jersey number
                jersey_num = tracker_to_jersey.get(tracker_id, tracker_id)
                
                # Draw circle with BLACK border
                radius = 15
                cv2.circle(tactical_board, (int(x), int(y)), radius, color, -1)
                cv2.circle(tactical_board, (int(x), int(y)), radius, (0, 0, 0), 2)
                
                # Draw jersey number with BLACK text
                text = str(jersey_num)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 2
                
                text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                text_x = int(x - text_size[0] / 2)
                text_y = int(y + text_size[1] / 2)
                
                cv2.putText(tactical_board, text, (text_x, text_y),
                           font, font_scale, (0, 0, 0), thickness)
            
            # Draw ball
            if ball_detections is not None and len(ball_detections) > 0:
                ball_xy = ball_detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
                transformed_ball = transformer.transform_points(points=ball_xy)
                
                ball_board = transformed_ball.copy()
                ball_board[:, 0] *= scale_x
                ball_board[:, 1] *= scale_y
                
                for bx, by in ball_board:
                    bx = np.clip(bx, 30, board_w - 30)
                    by = np.clip(by, 30, board_h - 30)
                    
                    cv2.circle(tactical_board, (int(bx), int(by)), 10, (255, 255, 255), -1)
                    cv2.circle(tactical_board, (int(bx), int(by)), 10, (0, 0, 0), 2)
        
        except Exception as e:
            print(f"Tactical board error: {e}")
    
    # Add formations at bottom
    y_offset = board_h - 40
    
    cv2.putText(tactical_board, f"{team_a_name}: {formation_a}", 
                (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, team_a_color, 2)
    
    text_size = cv2.getTextSize(f"{team_b_name}: {formation_b}", 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    cv2.putText(tactical_board, f"{team_b_name}: {formation_b}", 
                (board_w - text_size[0] - 20, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, team_b_color, 2)
    
    # Add title
    cv2.putText(tactical_board, "TACTICAL BOARD", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    return tactical_board


def main():
    parser = argparse.ArgumentParser(
        description="Soccer Analysis - Position-Based Tracking"
    )
    parser.add_argument("--source_video_path", type=str, required=True)
    parser.add_argument("--target_video_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max_frames", type=int, default=None,
                       help="Maximum number of frames to process")
    
    # Formation arguments (optional)
    parser.add_argument("--team_a_formation", type=str, default=None,
                       help="Team A formation (e.g., 4-3-3)")
    parser.add_argument("--team_b_formation", type=str, default=None,
                       help="Team B formation (e.g., 4-4-2)")
    parser.add_argument("--team_a_name", type=str, default=None,
                       help="Team A name")
    parser.add_argument("--team_b_name", type=str, default=None,
                       help="Team B name")
    
    # Confidence threshold
    parser.add_argument("--confidence_threshold", type=float, default=0.65,
                       help="Minimum detection confidence (default: 0.65)")
    
    args = parser.parse_args()
    
    process_video_final(
        source_path=args.source_video_path,
        target_path=args.target_video_path,
        device=args.device,
        team_a_formation=args.team_a_formation,
        team_b_formation=args.team_b_formation,
        team_a_name=args.team_a_name,
        team_b_name=args.team_b_name,
        max_frames=args.max_frames,
        confidence_threshold=args.confidence_threshold
    )


if __name__ == "__main__":
    main()
