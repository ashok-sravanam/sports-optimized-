#!/usr/bin/env python3
"""
analyze_tracking_data.py
Quick script to analyze exported tracking data
"""

import pandas as pd
import json
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

def analyze_latest_tracking_data(data_dir: str = "tracking_data"):
    """Analyze the most recent tracking data export"""
    
    data_path = Path(data_dir)
    
    # Find latest CSV
    csv_files = list(data_path.glob("positions_*.csv"))
    if not csv_files:
        print("No tracking data found!")
        return
    
    latest_csv = max(csv_files, key=lambda p: p.stat().st_mtime)
    print(f"Analyzing: {latest_csv}")
    
    # Load data
    df = pd.read_csv(latest_csv)
    
    print(f"\n{'='*60}")
    print("TRACKING DATA ANALYSIS")
    print(f"{'='*60}\n")
    
    # Basic stats
    print(f"Total records: {len(df)}")
    print(f"Total frames: {df['frame'].max()}")
    print(f"Duration: {df['timestamp'].max():.2f} seconds")
    print(f"Unique players: {df['tracker_id'].nunique()}")
    print(f"Teams: {df['team_id'].unique()}")
    
    # Per-player stats
    print(f"\n{'-'*60}")
    print("PER-PLAYER STATISTICS")
    print(f"{'-'*60}\n")
    
    for _, player_df in df.groupby('tracker_id'):
        player_name = player_df.iloc[0]['player_name']
        jersey = player_df.iloc[0]['jersey']
        team = player_df.iloc[0]['team_id']
        
        print(f"Player: {player_name} (Jersey #{jersey}, Team {team})")
        print(f"  Frames tracked: {len(player_df)}")
        print(f"  Avg position: ({player_df['pitch_x'].mean():.2f}, {player_df['pitch_y'].mean():.2f})")
        print(f"  Avg confidence: {player_df['confidence'].mean():.3f}")
        print()
    
    # Plot heatmaps
    print("Generating visualizations...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Team 1
    team1_df = df[df['team_id'] == df['team_id'].unique()[0]]
    axes[0].scatter(team1_df['pitch_x'], team1_df['pitch_y'], 
                   c='blue', alpha=0.5, s=10)
    axes[0].set_title(f"Team {team1_df['team_id'].iloc[0]} Heatmap")
    axes[0].set_xlabel("Pitch X")
    axes[0].set_ylabel("Pitch Y")
    axes[0].grid(True)
    
    # Team 2
    if len(df['team_id'].unique()) > 1:
        team2_df = df[df['team_id'] == df['team_id'].unique()[1]]
        axes[1].scatter(team2_df['pitch_x'], team2_df['pitch_y'], 
                       c='red', alpha=0.5, s=10)
        axes[1].set_title(f"Team {team2_df['team_id'].iloc[0]} Heatmap")
        axes[1].set_xlabel("Pitch X")
        axes[1].set_ylabel("Pitch Y")
        axes[1].grid(True)
    
    plt.tight_layout()
    
    output_plot = data_path / f"heatmap_{latest_csv.stem}.png"
    plt.savefig(output_plot, dpi=150)
    print(f"✓ Heatmap saved: {output_plot}")
    
    plt.show()

if __name__ == "__main__":
    analyze_latest_tracking_data()
