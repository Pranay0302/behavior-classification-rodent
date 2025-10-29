#!/usr/bin/env python3
"""
Script to download/save all plots from the SIMBA Alternative behavioral analysis notebook.
This script runs the notebook code and saves all generated plots to files.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, silhouette_score
from tqdm import tqdm
import time
import warnings
warnings.filterwarnings('ignore')

# Create output directory for plots
output_dir = "saved_plots"
os.makedirs(output_dir, exist_ok=True)
print(f"Created output directory: {output_dir}")

# Set matplotlib to use a non-interactive backend for saving
plt.switch_backend('Agg')

def save_plot(fig, filename, dpi=300):
    """Save a matplotlib figure to file"""
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"Saved plot: {filepath}")
    return filepath

# Load data
print("=== LOADING DATA ===")
csv_path = "data/labels.v003.000_2025-07-15 Boxes B3&B4_mjpg.analysis.csv"
df = pd.read_csv(csv_path)
print(f"Original data shape: {df.shape}")

# Sample data for efficiency
df_sample = df.iloc[::10].copy()
print(f"Sampled data shape: {df_sample.shape}")

# Sort by track and frame
df_sample = df_sample.sort_values(['track', 'frame_idx']).reset_index(drop=True)

# Calculate velocities
print("=== CALCULATING VELOCITIES ===")

def calculate_velocity_robust(group, x_col, y_col):
    """Calculate velocity for a specific body part with robust error handling"""
    if len(group) < 2:
        return np.full(len(group), np.nan)
    
    try:
        x_coords = group[x_col].values
        y_coords = group[y_col].values
        
        valid_mask = ~(np.isnan(x_coords) | np.isnan(y_coords))
        if not np.any(valid_mask):
            return np.full(len(group), np.nan)
        
        dx = np.diff(x_coords)
        dy = np.diff(y_coords)
        vel = np.sqrt(dx**2 + dy**2)
        vel = np.concatenate([[np.nan], vel])
        return vel
    except Exception as e:
        print(f"Error calculating velocity: {e}")
        return np.full(len(group), np.nan)

# Calculate nose velocity
nose_velocities = []
for track in df_sample['track'].unique():
    track_data = df_sample[df_sample['track'] == track].sort_values('frame_idx')
    vel = calculate_velocity_robust(track_data, 'nose.x', 'nose.y')
    nose_velocities.extend(vel)

df_sample['nose_velocity'] = nose_velocities

# Calculate tail base velocity
tail_velocities = []
for track in df_sample['track'].unique():
    track_data = df_sample[df_sample['track'] == track].sort_values('frame_idx')
    vel = calculate_velocity_robust(track_data, 't_base.x', 't_base.y')
    tail_velocities.extend(vel)

df_sample['tail_base_velocity'] = tail_velocities

# Calculate body length
df_sample['body_length'] = np.sqrt(
    (df_sample['nose.x'] - df_sample['t_base.x'])**2 + 
    (df_sample['nose.y'] - df_sample['t_base.y'])**2
)

# Convert velocities to numeric
df_sample['nose_velocity'] = pd.to_numeric(df_sample['nose_velocity'], errors='coerce')
df_sample['tail_base_velocity'] = pd.to_numeric(df_sample['tail_base_velocity'], errors='coerce')

# Calculate additional velocity metrics
df_sample['velocity_magnitude'] = np.sqrt(
    df_sample['nose_velocity']**2 + df_sample['tail_base_velocity']**2
)

df_sample['velocity_ratio'] = df_sample['nose_velocity'] / (df_sample['tail_base_velocity'] + 1e-6)
df_sample['velocity_difference'] = df_sample['nose_velocity'] - df_sample['tail_base_velocity']

# Calculate accelerations
def calculate_acceleration(group, vel_col):
    """Calculate acceleration for a velocity column"""
    if len(group) < 2:
        return np.full(len(group), np.nan)
    
    velocities = group[vel_col].values
    accel = np.diff(velocities)
    accel = np.concatenate([[np.nan], accel])
    return accel

nose_accelerations = []
tail_accelerations = []

for track in df_sample['track'].unique():
    track_data = df_sample[df_sample['track'] == track].sort_values('frame_idx')
    nose_accel = calculate_acceleration(track_data, 'nose_velocity')
    tail_accel = calculate_acceleration(track_data, 'tail_base_velocity')
    nose_accelerations.extend(nose_accel)
    tail_accelerations.extend(tail_accel)

df_sample['nose_acceleration'] = nose_accelerations
df_sample['tail_base_acceleration'] = tail_accelerations
df_sample['nose_acceleration'] = pd.to_numeric(df_sample['nose_acceleration'], errors='coerce')
df_sample['tail_base_acceleration'] = pd.to_numeric(df_sample['tail_base_acceleration'], errors='coerce')

print("Velocity calculations complete!")

# PLOT 1: Simple velocity plots for nose and tail base
print("=== CREATING PLOT 1: Simple Velocity Plots ===")
tracks = df_sample['track'].unique()
colors = ['blue', 'red', 'green', 'orange', 'purple']

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Nose velocity over time
ax1 = axes[0, 0]
for i, track in enumerate(tracks):
    try:
        track_data = df_sample[df_sample['track'] == track].sort_values('frame_idx')
        if len(track_data) > 0 and track_data['nose_velocity'].notna().any():
            color = colors[i % len(colors)]
            ax1.plot(track_data['frame_idx'], track_data['nose_velocity'], 
                     label=f'Track {track}', alpha=0.8, linewidth=2, color=color)
    except Exception as e:
        print(f"Error plotting track {track}: {e}")

ax1.set_xlabel('Frame Index')
ax1.set_ylabel('Nose Velocity (px/frame)')
ax1.set_title('Nose Velocity Over Time', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Tail base velocity over time
ax2 = axes[0, 1]
for i, track in enumerate(tracks):
    try:
        track_data = df_sample[df_sample['track'] == track].sort_values('frame_idx')
        if len(track_data) > 0 and track_data['tail_base_velocity'].notna().any():
            color = colors[i % len(colors)]
            ax2.plot(track_data['frame_idx'], track_data['tail_base_velocity'], 
                     label=f'Track {track}', alpha=0.8, linewidth=2, color=color)
    except Exception as e:
        print(f"Error plotting track {track}: {e}")

ax2.set_xlabel('Frame Index')
ax2.set_ylabel('Tail Base Velocity (px/frame)')
ax2.set_title('Tail Base Velocity Over Time', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Velocity correlation scatter plot
ax3 = axes[1, 0]
valid_data = df_sample.dropna(subset=['nose_velocity', 'tail_base_velocity'])

if len(valid_data) > 0:
    track_mapping = {track: i for i, track in enumerate(tracks)}
    track_colors = valid_data['track'].map(track_mapping)
    track_colors = track_colors.fillna(0)
    
    scatter = ax3.scatter(valid_data['nose_velocity'], valid_data['tail_base_velocity'], 
                         c=track_colors, alpha=0.6, s=20, cmap='viridis')
    ax3.set_xlabel('Nose Velocity (px/frame)')
    ax3.set_ylabel('Tail Base Velocity (px/frame)')
    ax3.set_title('Nose vs Tail Base Velocity Correlation', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('Track')
    cbar.set_ticks(range(len(tracks)))
    cbar.set_ticklabels([f'Track {track}' for track in tracks])
else:
    ax3.text(0.5, 0.5, 'No valid velocity data for scatter plot', 
             ha='center', va='center', transform=ax3.transAxes)
    ax3.set_title('Nose vs Tail Base Velocity Correlation', fontsize=12, fontweight='bold')

# Plot 4: Velocity distribution histogram
ax4 = axes[1, 1]
nose_vel_data = df_sample['nose_velocity'].dropna()
tail_vel_data = df_sample['tail_base_velocity'].dropna()

if len(nose_vel_data) > 0 and len(tail_vel_data) > 0:
    ax4.hist(nose_vel_data, bins=30, alpha=0.7, 
             label='Nose Velocity', density=True, color='blue', edgecolor='black')
    ax4.hist(tail_vel_data, bins=30, alpha=0.7, 
             label='Tail Base Velocity', density=True, color='red', edgecolor='black')
    ax4.set_xlabel('Velocity (px/frame)')
    ax4.set_ylabel('Density')
    ax4.set_title('Velocity Distribution', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
else:
    ax4.text(0.5, 0.5, 'No valid velocity data for histogram', 
             ha='center', va='center', transform=ax4.transAxes)
    ax4.set_title('Velocity Distribution', fontsize=12, fontweight='bold')

plt.tight_layout()
save_plot(fig, "01_simple_velocity_plots.png")
plt.close()

# Behavioral classification
print("=== CREATING BEHAVIORAL CLASSIFICATION ===")
def classify_behavior(row):
    """Classify behavior based on nose velocity and body length"""
    nose_vel = row['nose_velocity']
    body_len = row['body_length']
    
    if pd.isna(nose_vel) or pd.isna(body_len):
        return 'Unknown'
    
    if nose_vel < 1.0:
        if body_len < 60:
            return 'Sleeping'
        else:
            return 'Resting'
    elif nose_vel < 5.0:
        return 'Slow Movement'
    elif nose_vel < 15.0:
        return 'Moderate Movement'
    else:
        return 'Fast Movement'

df_sample['behavior_state'] = df_sample.apply(classify_behavior, axis=1)

# PLOT 2: Behavioral timeline visualization
print("=== CREATING PLOT 2: Behavioral Timeline ===")
behavior_colors = {
    'Sleeping': 'darkblue',
    'Resting': 'lightblue',
    'Slow Movement': 'green',
    'Moderate Movement': 'orange',
    'Fast Movement': 'red',
    'Unknown': 'gray'
}

fig, axes = plt.subplots(len(tracks), 1, figsize=(20, 6*len(tracks)))
if len(tracks) == 1:
    axes = [axes]

for i, track in enumerate(tracks):
    track_data = df_sample[df_sample['track'] == track].sort_values('frame_idx')
    
    if len(track_data) > 0:
        ax = axes[i]
        
        for _, row in track_data.iterrows():
            behavior = row['behavior_state']
            color = behavior_colors.get(behavior, 'gray')
            ax.barh(i, 20, left=row['frame_idx'], height=0.8, color=color, alpha=0.8)
        
        ax.set_xlabel('Frame Index (sampled every 10th frame)')
        ax.set_ylabel(f'Track {track}')
        ax.set_title(f'Behavioral Timeline - Track {track} (Based on Nose & Tail Base)')
        ax.set_xlim(0, track_data['frame_idx'].max())
        
        if i == 0:
            legend_elements = [plt.Rectangle((0,0),1,1, color=color, label=behavior) 
                             for behavior, color in behavior_colors.items()]
            ax.legend(handles=legend_elements, loc='upper right')

plt.tight_layout()
save_plot(fig, "02_behavioral_timeline.png")
plt.close()

# PLOT 3: Comprehensive velocity plots
print("=== CREATING PLOT 3: Comprehensive Velocity Plots ===")
fig, axes = plt.subplots(2, 2, figsize=(20, 12))

# Plot 1: Nose velocity over time for each track
ax1 = axes[0, 0]
for track in tracks:
    track_data = df_sample[df_sample['track'] == track].sort_values('frame_idx')
    ax1.plot(track_data['frame_idx'], track_data['nose_velocity'], 
             label=f'Track {track}', alpha=0.7, linewidth=1)
ax1.set_xlabel('Frame Index')
ax1.set_ylabel('Nose Velocity (px/frame)')
ax1.set_title('Nose Velocity Over Time')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Tail base velocity over time for each track
ax2 = axes[0, 1]
for track in tracks:
    track_data = df_sample[df_sample['track'] == track].sort_values('frame_idx')
    ax2.plot(track_data['frame_idx'], track_data['tail_base_velocity'], 
             label=f'Track {track}', alpha=0.7, linewidth=1)
ax2.set_xlabel('Frame Index')
ax2.set_ylabel('Tail Base Velocity (px/frame)')
ax2.set_title('Tail Base Velocity Over Time')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Velocity distribution histograms
ax3 = axes[1, 0]
ax3.hist(df_sample['nose_velocity'].dropna(), bins=50, alpha=0.7, 
         label='Nose Velocity', density=True, color='blue')
ax3.hist(df_sample['tail_base_velocity'].dropna(), bins=50, alpha=0.7, 
         label='Tail Base Velocity', density=True, color='red')
ax3.set_xlabel('Velocity (px/frame)')
ax3.set_ylabel('Density')
ax3.set_title('Velocity Distribution')
ax3.legend()
ax3.set_yscale('log')

# Plot 4: Scatter plot of nose vs tail base velocity
ax4 = axes[1, 1]
track_mapping = {track: i for i, track in enumerate(tracks)}
track_colors = df_sample['track'].map(track_mapping)
valid_data = df_sample.dropna(subset=['nose_velocity', 'tail_base_velocity'])

if len(valid_data) > 0:
    valid_track_colors = valid_data['track'].map(track_mapping)
    
    scatter = ax4.scatter(valid_data['nose_velocity'], valid_data['tail_base_velocity'], 
                         c=valid_track_colors, alpha=0.6, s=10, cmap='viridis')
    ax4.set_xlabel('Nose Velocity (px/frame)')
    ax4.set_ylabel('Tail Base Velocity (px/frame)')
    ax4.set_title('Nose vs Tail Base Velocity')
    ax4.grid(True, alpha=0.3)
    
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('Track')
    cbar.set_ticks(range(len(tracks)))
    cbar.set_ticklabels([f'Track {track}' for track in tracks])
else:
    ax4.text(0.5, 0.5, 'No valid velocity data for scatter plot', 
             ha='center', va='center', transform=ax4.transAxes)
    ax4.set_title('Nose vs Tail Base Velocity')

plt.tight_layout()
save_plot(fig, "03_comprehensive_velocity_plots.png")
plt.close()

# Comprehensive analysis with all body parts
print("=== CREATING COMPREHENSIVE ANALYSIS ===")
df_comprehensive = df_sample.copy()

# Define all body parts
body_parts = {
    'nose': ['nose.x', 'nose.y'],
    'neck': ['neck.x', 'neck.y'],
    'left_ear': ['l_ear.x', 'l_ear.y'],
    'right_ear': ['r_ear.x', 'r_ear.y'],
    'left_front_paw': ['l_frpaw.x', 'l_frpaw.y'],
    'right_front_paw': ['r_frpaw.x', 'r_frpaw.y'],
    'left_back_paw': ['l_bcpaw.x', 'l_bcpaw.y'],
    'right_back_paw': ['r_bcpaw.x', 'r_bcpaw.y'],
    'tail_base': ['t_base.x', 't_base.y'],
    'tail_mid': ['t_mid.x', 't_mid.y'],
    'tail_end': ['t_end.x', 't_end.y']
}

# Calculate velocities for all body parts
for part_name, (x_col, y_col) in body_parts.items():
    velocities = []
    for track in df_comprehensive['track'].unique():
        track_data = df_comprehensive[df_comprehensive['track'] == track].sort_values('frame_idx')
        vel = calculate_velocity_robust(track_data, x_col, y_col)
        velocities.extend(vel)
    
    df_comprehensive[f'{part_name}_velocity'] = velocities
    df_comprehensive[f'{part_name}_velocity'] = pd.to_numeric(df_comprehensive[f'{part_name}_velocity'], errors='coerce')

# Calculate additional features
df_comprehensive['head_length'] = np.sqrt(
    (df_comprehensive['nose.x'] - df_comprehensive['neck.x'])**2 + 
    (df_comprehensive['nose.y'] - df_comprehensive['neck.y'])**2
)

df_comprehensive['ear_span'] = np.sqrt(
    (df_comprehensive['l_ear.x'] - df_comprehensive['r_ear.x'])**2 + 
    (df_comprehensive['l_ear.y'] - df_comprehensive['r_ear.y'])**2
)

df_comprehensive['tail_length'] = np.sqrt(
    (df_comprehensive['t_base.x'] - df_comprehensive['t_end.x'])**2 + 
    (df_comprehensive['t_base.y'] - df_comprehensive['t_end.y'])**2
)

# Calculate center of mass
def calculate_center_of_mass(row):
    x_coords = []
    y_coords = []
    
    for part_name, (x_col, y_col) in body_parts.items():
        if not (pd.isna(row[x_col]) or pd.isna(row[y_col])):
            x_coords.append(row[x_col])
            y_coords.append(row[y_col])
    
    if len(x_coords) > 0:
        return np.mean(x_coords), np.mean(y_coords)
    else:
        return np.nan, np.nan

com_x, com_y = zip(*df_comprehensive.apply(calculate_center_of_mass, axis=1))
df_comprehensive['com_x'] = com_x
df_comprehensive['com_y'] = com_y

# Calculate COM velocity
com_velocities = []
for track in df_comprehensive['track'].unique():
    track_data = df_comprehensive[df_comprehensive['track'] == track].sort_values('frame_idx')
    vel = calculate_velocity_robust(track_data, 'com_x', 'com_y')
    com_velocities.extend(vel)

df_comprehensive['com_velocity'] = com_velocities
df_comprehensive['com_velocity'] = pd.to_numeric(df_comprehensive['com_velocity'], errors='coerce')

# Comprehensive behavioral classification
def classify_comprehensive_behavior(row):
    nose_vel = row['nose_velocity']
    com_vel = row['com_velocity']
    body_len = row['body_length']
    head_len = row['head_length']
    ear_span = row['ear_span']
    tail_len = row['tail_length']
    
    l_frpaw_vel = row['left_front_paw_velocity']
    r_frpaw_vel = row['right_front_paw_velocity']
    l_ear_vel = row['left_ear_velocity']
    r_ear_vel = row['right_ear_velocity']
    tail_base_vel = row['tail_base_velocity']
    tail_end_vel = row['tail_end_velocity']
    
    if pd.isna(nose_vel) or pd.isna(com_vel) or pd.isna(body_len):
        return 'Unknown'
    
    avg_paw_vel = np.nanmean([l_frpaw_vel, r_frpaw_vel])
    avg_ear_vel = np.nanmean([l_ear_vel, r_ear_vel])
    avg_tail_vel = np.nanmean([tail_base_vel, tail_end_vel])
    
    # Sleeping
    if com_vel < 0.5 and nose_vel < 0.5 and body_len < 60:
        if avg_ear_vel < 0.2:
            return 'Sleeping_NREM'
        else:
            return 'Sleeping_REM'
    
    # Freezing
    elif com_vel < 0.3 and nose_vel < 0.3 and body_len > 60 and avg_ear_vel > 0.1:
        return 'Freezing'
    
    # Resting
    elif com_vel < 1.0 and nose_vel < 1.0 and body_len > 60:
        return 'Resting'
    
    # Grooming
    elif (avg_paw_vel > 2.0 and com_vel < 2.0 and 
          nose_vel < 3.0 and body_len > 60):
        return 'Grooming'
    
    # Sniffing
    elif (nose_vel > 2.0 and com_vel < 2.0 and 
          head_len > 20 and avg_ear_vel > 0.5):
        return 'Sniffing'
    
    # Rearing
    elif (body_len > 100 and head_len > 30 and 
          com_vel < 3.0 and nose_vel > 1.0):
        return 'Rearing'
    
    # Locomotion
    elif com_vel > 2.0:
        if com_vel > 8.0:
            return 'Locomotion_Fast'
        elif com_vel > 4.0:
            return 'Locomotion_Moderate'
        else:
            return 'Locomotion_Slow'
    
    # Waking
    elif nose_vel > 1.0 or com_vel > 1.0:
        if com_vel > 2.0 or nose_vel > 3.0:
            return 'Waking_Active'
        else:
            return 'Waking_Quiet'
    
    else:
        return 'Unknown'

df_comprehensive['behavior_state'] = df_comprehensive.apply(classify_comprehensive_behavior, axis=1)

# PLOT 4: Focused behavioral visualization
print("=== CREATING PLOT 4: Focused Behavioral Visualization ===")
behavior_colors = {
    'Sleeping_NREM': 'darkblue',
    'Sleeping_REM': 'blue',
    'Freezing': 'purple',
    'Resting': 'lightblue',
    'Grooming': 'orange',
    'Sniffing': 'yellow',
    'Rearing': 'brown',
    'Locomotion_Slow': 'lightgreen',
    'Locomotion_Moderate': 'green',
    'Locomotion_Fast': 'darkgreen',
    'Waking_Quiet': 'pink',
    'Waking_Active': 'red',
    'Unknown': 'gray'
}

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Key body part velocities over time
ax1 = axes[0, 0]
key_parts = ['neck', 'tail_base', 'tail_end']
colors = ['red', 'green', 'orange']
for i, part_name in enumerate(key_parts):
    vel_col = f'{part_name}_velocity'
    if vel_col in df_comprehensive.columns:
        for track in tracks:
            track_data = df_comprehensive[df_comprehensive['track'] == track].sort_values('frame_idx')
            if len(track_data) > 0 and track_data[vel_col].notna().any():
                ax1.plot(track_data['frame_idx'], track_data[vel_col], 
                        label=f'{part_name} (Track {track})', alpha=0.7, linewidth=1, color=colors[i])

ax1.set_xlabel('Frame Index')
ax1.set_ylabel('Velocity (px/frame)')
ax1.set_title('Key Body Part Velocities Over Time (No Nose)', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: COM velocity over time
ax2 = axes[0, 1]
for track in tracks:
    track_data = df_comprehensive[df_comprehensive['track'] == track].sort_values('frame_idx')
    if len(track_data) > 0 and track_data['com_velocity'].notna().any():
        ax2.plot(track_data['frame_idx'], track_data['com_velocity'], 
                label=f'Track {track}', alpha=0.7, linewidth=1)

ax2.set_xlabel('Frame Index')
ax2.set_ylabel('COM Velocity (px/frame)')
ax2.set_title('Center of Mass Velocity Over Time', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Behavioral timeline for track 1
ax3 = axes[1, 0]
if len(tracks) > 0:
    track_data = df_comprehensive[df_comprehensive['track'] == tracks[0]].sort_values('frame_idx')
    if len(track_data) > 0:
        for i, (_, row) in enumerate(track_data.iterrows()):
            behavior = row['behavior_state']
            color = behavior_colors.get(behavior, 'gray')
            height = 0.8 if behavior.startswith('Sleeping') else 0.6
            ax3.barh(0, 20, left=row['frame_idx'], height=height, color=color, alpha=0.8)
        
        ax3.set_xlabel('Frame Index')
        ax3.set_ylabel(f'Track {tracks[0]}')
        ax3.set_title(f'Behavioral Timeline - Track {tracks[0]}', fontsize=12, fontweight='bold')
        ax3.set_xlim(0, track_data['frame_idx'].max())
        
        legend_elements = [plt.Rectangle((0,0),1,1, color=color, label=behavior) 
                         for behavior, color in behavior_colors.items() 
                         if behavior in track_data['behavior_state'].values]
        ax3.legend(handles=legend_elements, loc='upper right', fontsize=8)

# Plot 4: Behavioral timeline for track 2
ax4 = axes[1, 1]
if len(tracks) > 1:
    track_data = df_comprehensive[df_comprehensive['track'] == tracks[1]].sort_values('frame_idx')
    if len(track_data) > 0:
        for i, (_, row) in enumerate(track_data.iterrows()):
            behavior = row['behavior_state']
            color = behavior_colors.get(behavior, 'gray')
            height = 0.8 if behavior.startswith('Sleeping') else 0.6
            ax4.barh(0, 20, left=row['frame_idx'], height=height, color=color, alpha=0.8)
        
        ax4.set_xlabel('Frame Index')
        ax4.set_ylabel(f'Track {tracks[1]}')
        ax4.set_title(f'Behavioral Timeline - Track {tracks[1]}', fontsize=12, fontweight='bold')
        ax4.set_xlim(0, track_data['frame_idx'].max())
        
        legend_elements = [plt.Rectangle((0,0),1,1, color=color, label=behavior) 
                         for behavior, color in behavior_colors.items() 
                         if behavior in track_data['behavior_state'].values]
        ax4.legend(handles=legend_elements, loc='upper right', fontsize=8)

plt.tight_layout()
save_plot(fig, "04_focused_behavioral_visualization.png")
plt.close()

# PLOT 5: Behavioral distribution pie charts
print("=== CREATING PLOT 5: Behavioral Distribution Charts ===")
fig_pie, axes_pie = plt.subplots(1, 2, figsize=(16, 8))

# Overall behavioral distribution pie chart
ax_pie1 = axes_pie[0]
behavior_counts = df_comprehensive['behavior_state'].value_counts()
colors_pie = [behavior_colors.get(behavior, 'gray') for behavior in behavior_counts.index]

wedges, texts, autotexts = ax_pie1.pie(behavior_counts.values, 
                                      labels=behavior_counts.index, 
                                      autopct='%1.1f%%', 
                                      colors=colors_pie,
                                      startangle=90,
                                      explode=[0.05 if behavior.startswith('Sleeping') else 0 for behavior in behavior_counts.index])

for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(10)

ax_pie1.set_title('Overall Behavioral Distribution', fontsize=14, fontweight='bold', pad=20)

# Track-specific behavioral distribution
ax_pie2 = axes_pie[1]
if len(tracks) > 1:
    track1_data = df_comprehensive[df_comprehensive['track'] == tracks[0]]
    track2_data = df_comprehensive[df_comprehensive['track'] == tracks[1]]
    
    track1_counts = track1_data['behavior_state'].value_counts()
    track2_counts = track2_data['behavior_state'].value_counts()
    
    all_behaviors = list(set(track1_counts.index) | set(track2_counts.index))
    
    x = np.arange(len(all_behaviors))
    width = 0.35
    
    track1_values = [track1_counts.get(behavior, 0) for behavior in all_behaviors]
    track2_values = [track2_counts.get(behavior, 0) for behavior in all_behaviors]
    
    bars1 = ax_pie2.bar(x - width/2, track1_values, width, label=f'Track {tracks[0]}', alpha=0.8)
    bars2 = ax_pie2.bar(x + width/2, track2_values, width, label=f'Track {tracks[1]}', alpha=0.8)
    
    for i, behavior in enumerate(all_behaviors):
        color = behavior_colors.get(behavior, 'gray')
        bars1[i].set_color(color)
        bars2[i].set_color(color)
    
    ax_pie2.set_xlabel('Behavioral States')
    ax_pie2.set_ylabel('Number of Frames')
    ax_pie2.set_title('Behavioral Distribution by Track', fontsize=14, fontweight='bold')
    ax_pie2.set_xticks(x)
    ax_pie2.set_xticklabels(all_behaviors, rotation=45, ha='right')
    ax_pie2.legend()
    ax_pie2.grid(True, alpha=0.3)
    
    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            ax_pie2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom', fontsize=8)
    
    for bar in bars2:
        height = bar.get_height()
        if height > 0:
            ax_pie2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom', fontsize=8)

else:
    track_data = df_comprehensive[df_comprehensive['track'] == tracks[0]]
    track_counts = track_data['behavior_state'].value_counts()
    colors_track = [behavior_colors.get(behavior, 'gray') for behavior in track_counts.index]
    
    wedges, texts, autotexts = ax_pie2.pie(track_counts.values, 
                                          labels=track_counts.index, 
                                          autopct='%1.1f%%', 
                                          colors=colors_track,
                                          startangle=90,
                                          explode=[0.05 if behavior.startswith('Sleeping') else 0 for behavior in track_counts.index])
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(10)
    
    ax_pie2.set_title(f'Behavioral Distribution - Track {tracks[0]}', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
save_plot(fig_pie, "05_behavioral_distribution_charts.png")
plt.close()

print(f"\n=== ALL PLOTS SAVED SUCCESSFULLY ===")
print(f"Plots saved to directory: {output_dir}")
print(f"Total plots created: 5")
print("\nPlot files:")
for i, filename in enumerate([
    "01_simple_velocity_plots.png",
    "02_behavioral_timeline.png", 
    "03_comprehensive_velocity_plots.png",
    "04_focused_behavioral_visualization.png",
    "05_behavioral_distribution_charts.png"
], 1):
    print(f"  {i}. {filename}")

print(f"\nAll plots are high-resolution (300 DPI) and ready for presentation/publication!")
