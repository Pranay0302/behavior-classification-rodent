# SIMBA Alternative - Behavioral Analysis

A comprehensive Python-based alternative to SIMBA (Simple Behavioral Analysis) for analyzing animal behavior from pose estimation data. This project provides optimized, vectorized analysis of behavioral patterns with advanced motif detection and comprehensive statistical analysis.

## Overview

This project analyzes behavioral data from pose estimation tracking (DeepLabCut/SLEAP) to identify and classify various animal behaviors including:

- **Basic Behaviors**: Sleeping, Resting, Slow/Moderate/Fast Movement
- **Advanced Motifs**: Sniffing, Freezing, Grooming, Rearing, Exploration, Thigmotaxis, Circling, Jumping
- **Statistical Analysis**: Behavioral transitions, stability analysis, entropy analysis, and feature correlations

## Key Features

### Core Analysis Capabilities
- **Vectorized Processing**: Optimized for large datasets with minimal memory usage
- **Multi-Track Support**: Analyze multiple animals simultaneously
- **Advanced Feature Calculation**: Velocity, acceleration, angular velocity, spatial analysis
- **Comprehensive Behavioral Classification**: 13+ distinct behavioral states
- **Statistical Analysis**: Transition matrices, stability scores, entropy analysis

### Visualization Suite
- **Behavioral Timelines**: Frame-by-frame behavioral state visualization
- **Velocity Analysis**: Nose, tail, and center-of-mass velocity tracking
- **Distribution Plots**: Behavioral frequency and duration analysis
- **Correlation Matrices**: Feature relationship analysis
- **Transition Heatmaps**: Behavioral state transition probabilities

### Output Formats
- **CSV Data**: Comprehensive behavioral analysis results
- **Statistical Reports**: Detailed behavioral statistics and summaries
- **High-Resolution Plots**: Publication-ready visualizations (300 DPI)
- **Transition Matrices**: Behavioral state transition probabilities

## Quick Start

### Prerequisites
- Python 3.7+
- Required packages (see `requirements.txt`)

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd SimbaAlt
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare your data**:
   - Place your pose estimation CSV file in the `data/` directory
   - Ensure the CSV contains required columns: `track`, `frame_idx`, `nose.x`, `nose.y`, `t_base.x`, `t_base.y`

### Basic Usage

#### Option 1: Jupyter Notebook (Interactive)
```bash
jupyter notebook SIMBA_alternative.ipynb
```

#### Option 2: Command Line (Automated)
```bash
# Generate all plots and analysis
python scripts/download_plots.py

# Or use the shell script
chmod +x scripts/run_plot_download.sh
./scripts/run_plot_download.sh
```

## Project Structure

```
SimbaAlt/
├── README.md                           # This file
├── requirements.txt                     # Python dependencies
├── SIMBA_alternative.ipynb             # Main analysis notebook
├── data/                              # Data directory
│   └── labels.v003.000_2025-07-15 Boxes B3&B4_mjpg.analysis.csv
├── scripts/                           # Analysis scripts
│   ├── download_plots.py              # Automated plot generation
│   └── run_plot_download.sh           # Shell script for plot generation
├── analysis_results/                  # Generated analysis results
│   ├── saved_plots/                   # Generated plots directory
│   ├── enhanced_behavioral_analysis_*.csv
│   ├── behavioral_statistics_*.txt
│   ├── behavioral_transitions_*.csv
│   └── feature_correlations_*.csv
└── documentation/                     # Detailed documentation
    ├── README.md                      # Documentation index
    └── getting_started.md             # Getting started guide
```

## Analysis Pipeline

### 1. Data Loading and Validation
- Load pose estimation data from CSV
- Validate required columns and data integrity
- Sample data for efficiency (every 10th frame by default)

### 2. Feature Calculation
- **Velocity Calculation**: Nose, tail base, and center-of-mass velocities
- **Acceleration Analysis**: Second derivatives of position data
- **Spatial Features**: Body length, distance to center, movement direction
- **Angular Velocity**: Change in movement direction over time

### 3. Behavioral Classification
- **Basic Classification**: Velocity and body length-based states
- **Advanced Motif Detection**: Specific behavioral patterns
- **Comprehensive Classification**: Priority-based behavioral assignment

### 4. Statistical Analysis
- **Transition Analysis**: Behavioral state transition probabilities
- **Stability Analysis**: Self-transition probabilities
- **Entropy Analysis**: Behavioral unpredictability measures
- **Feature Correlations**: Relationship between behavioral features

### 5. Visualization and Output
- **Timeline Visualizations**: Frame-by-frame behavioral states
- **Distribution Plots**: Behavioral frequency analysis
- **Correlation Matrices**: Feature relationship heatmaps
- **Statistical Reports**: Comprehensive analysis summaries

## Behavioral States Detected

### Basic Behaviors
- **Sleeping**: Very low velocity, small body length
- **Resting**: Low velocity, extended body
- **Slow Movement**: Moderate velocity (1-5 px/frame)
- **Moderate Movement**: Higher velocity (5-15 px/frame)
- **Fast Movement**: High velocity (>15 px/frame)

### Advanced Motifs
- **Sniffing**: Low velocity, high angular velocity, small body length
- **Freezing**: Very low velocity, low acceleration, extended body
- **Grooming**: Moderate velocity, high angular velocity, variable body length
- **Rearing**: High nose velocity, low tail velocity, extended body
- **Exploration**: Moderate velocity, low angular velocity
- **Thigmotaxis**: Movement near walls (high distance to center)
- **Circling**: High angular velocity, moderate velocity
- **Jumping**: High acceleration, high velocity

## Output Files

### Data Files
- `analysis_results/enhanced_behavioral_analysis_*.csv`: Complete analysis results with all features
- `analysis_results/behavioral_transitions_*.csv`: Behavioral state transition matrix
- `analysis_results/feature_correlations_*.csv`: Feature correlation matrix

### Reports
- `analysis_results/behavioral_statistics_*.txt`: Comprehensive statistical summary
- `analysis_results/behavioral_transitions_*.csv`: Transition probability matrix

### Visualizations
- `analysis_results/saved_plots/01_simple_velocity_plots.png`: Basic velocity analysis
- `analysis_results/saved_plots/02_behavioral_timeline.png`: Frame-by-frame behavioral states
- `analysis_results/saved_plots/03_comprehensive_velocity_plots.png`: Advanced velocity analysis
- `analysis_results/saved_plots/04_focused_behavioral_visualization.png`: Key behavioral patterns
- `analysis_results/saved_plots/05_behavioral_distribution_charts.png`: Behavioral frequency analysis

## Configuration

### Data Sampling
By default, the analysis samples every 10th frame for efficiency. To change this:
```python
# In the notebook or script
df = df.iloc[::5].copy()  # Sample every 5th frame
```

### Behavioral Thresholds
Adjust behavioral classification thresholds in the classification functions:
```python
# Example: Adjust velocity thresholds
if nose_vel < 2.0:  # Changed from 1.0
    return 'Slow Movement'
```

### Visualization Settings
Modify plot appearance and save settings:
```python
# High-resolution output
fig.savefig(filename, dpi=300, bbox_inches='tight')
```

## Advanced Usage

### Custom Behavioral Classification
Add your own behavioral patterns by modifying the classification functions:

```python
def custom_behavior_classifier(row):
    # Your custom logic here
    if custom_condition:
        return 'CustomBehavior'
    # ... rest of classification
```

### Batch Processing
Process multiple datasets:
```python
import glob
csv_files = glob.glob('data/*.csv')
for csv_file in csv_files:
    # Process each file
    process_dataset(csv_file)
```

### Integration with Other Tools
The output CSV files can be easily integrated with:
- **R**: For advanced statistical analysis
- **MATLAB**: For signal processing
- **Python**: For machine learning pipelines
- **Excel**: For manual inspection and reporting

## Dependencies

### Core Libraries
- `numpy`: Numerical computations
- `pandas`: Data manipulation
- `matplotlib`: Plotting and visualization
- `seaborn`: Statistical visualization
- `scipy`: Scientific computing
- `scikit-learn`: Machine learning utilities

### Optional Libraries
- `jupyter`: Interactive notebook environment
- `tqdm`: Progress bars for long operations

## Troubleshooting

### Common Issues

1. **Missing Data Columns**:
   ```
   ValueError: Missing required columns: ['nose.x', 'nose.y']
   ```
   **Solution**: Ensure your CSV contains all required pose estimation columns.

2. **Memory Issues**:
   ```
   MemoryError: Unable to allocate array
   ```
   **Solution**: Increase sampling interval (e.g., `::20` instead of `::10`).

3. **Empty Results**:
   ```
   No valid velocity data for scatter plot
   ```
   **Solution**: Check data quality and ensure pose estimation confidence scores are adequate.

### Performance Optimization

- **Large Datasets**: Increase sampling interval
- **Memory Constraints**: Process tracks individually
- **Speed Optimization**: Use vectorized operations (already implemented)

## Documentation

- **[Getting Started](documentation/getting_started.md)**: Detailed setup and first analysis guide
- **[Analysis Pipeline](documentation/analysis_pipeline.md)**: Detailed workflow information
- **[Behavioral Classification](documentation/behavioral_classification.md)**: Understanding detected behaviors
- **[Output Formats](documentation/output_formats.md)**: Interpreting results
- **[Troubleshooting](documentation/troubleshooting.md)**: Common issues and solutions

