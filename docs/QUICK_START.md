# SIMBA Alternative - Quick Start Guide

## Getting Started in 5 Minutes

This guide will get you up and running with the SIMBA Alternative behavioral analysis toolkit quickly.

## Prerequisites

- Python 3.7 or higher
- Basic familiarity with command line
- Pose estimation data (CSV format)

## Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd SimbaAlt
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Verify Installation
```bash
python -c "import numpy, pandas, matplotlib; print('Installation successful!')"
```

## Quick Analysis

### Step 1: Prepare Your Data

Place your pose estimation CSV file in the `data/` directory. The file should contain columns:
- `track`: Animal identifier
- `frame_idx`: Frame number  
- `nose.x`, `nose.y`: Nose coordinates
- `t_base.x`, `t_base.y`: Tail base coordinates

### Step 2: Run Analysis

**Option A: Interactive Analysis (Recommended)**
```bash
jupyter notebook SIMBA_alternative.ipynb
```

**Option B: Command Line Analysis**
```bash
./scripts/run_plot_download.sh
```

### Step 3: View Results

Results will be saved in:
- `saved_plots/`: Visualization files
- `outputs/`: Analysis results and statistics

## Understanding Your Results

### Key Output Files

1. **Visualizations** (`saved_plots/`):
   - `01_simple_velocity_plots.png`: Basic movement analysis
   - `02_behavioral_timeline.png`: Behavioral sequences over time
   - `03_comprehensive_velocity_plots.png`: Detailed velocity analysis
   - `04_focused_behavioral_visualization.png`: Advanced behavioral plots
   - `05_behavioral_distribution_charts.png`: Behavioral distribution

2. **Data Files** (`outputs/`):
   - `enhanced_behavioral_analysis_*.csv`: Complete dataset with classifications
   - `behavioral_statistics_*.txt`: Statistical summary
   - `behavioral_transitions_*.csv`: Transition probability matrix

### Behavioral States Detected

The system automatically detects these behaviors:

**Basic States**:
- Sleeping, Resting, Slow Movement, Moderate Movement, Fast Movement

**Advanced Motifs**:
- Sniffing, Freezing, Grooming, Rearing, Exploration, Thigmotaxis, Circling, Jumping

## Common Use Cases

### 1. Basic Behavioral Analysis
```python
# Load your data
df = pd.read_csv('data/your_data.csv')

# Run basic analysis (see notebook for details)
# Results: Behavioral classifications and basic statistics
```

### 2. Comparative Analysis
```python
# Compare multiple animals
# Results: Track-specific behavioral profiles and comparisons
```

### 3. Temporal Analysis
```python
# Analyze behavioral changes over time
# Results: Transition matrices and stability analysis
```

## Troubleshooting

### Common Issues

**"Data file not found"**
- Ensure CSV file is in `data/` directory
- Check file permissions

**"Missing required columns"**
- Verify CSV has required columns: `track`, `frame_idx`, `nose.x`, `nose.y`, `t_base.x`, `t_base.y`
- Check column names match exactly

**"Memory error"**
- Reduce data sampling frequency
- Process smaller time windows
- Increase system RAM

**"No plots generated"**
- Check matplotlib backend
- Verify output directory permissions
- Ensure sufficient disk space

### Getting Help

1. Check the troubleshooting section in README.md
2. Review the example notebook
3. Consult the Analysis Methods documentation
4. Open an issue on GitHub

## Next Steps

### Advanced Analysis

1. **Customize Behavioral Thresholds**:
   - Modify classification parameters in the notebook
   - Adjust for your specific experimental conditions

2. **Add New Behaviors**:
   - Extend the classification system
   - Add species-specific behaviors

3. **Statistical Analysis**:
   - Import results into R or Python for further analysis
   - Perform group comparisons
   - Conduct time series analysis

### Integration

1. **Batch Processing**:
   - Process multiple files automatically
   - Create analysis pipelines

2. **Real-time Analysis**:
   - Adapt for live behavioral monitoring
   - Integrate with video analysis systems

3. **Publication**:
   - Use generated plots in papers
   - Export data for statistical software

## Example Workflow

```bash
# 1. Install and setup
git clone <repo-url>
cd SimbaAlt
pip install -r requirements.txt

# 2. Add your data
cp your_data.csv data/

# 3. Run analysis
jupyter notebook SIMBA_alternative.ipynb

# 4. View results
open saved_plots/
open outputs/
```

## Tips for Success

1. **Start Small**: Test with a subset of your data first
2. **Validate Results**: Check behavioral classifications manually
3. **Document Changes**: Keep track of parameter adjustments
4. **Save Intermediate Results**: Don't lose your work
5. **Use Version Control**: Track your analysis changes

## Support

- **Documentation**: Check `docs/` directory for detailed guides
- **Examples**: See the Jupyter notebook for code examples
- **Community**: Join discussions on GitHub
- **Issues**: Report bugs and request features

---

**Ready to analyze your behavioral data? Start with the Jupyter notebook and explore the comprehensive analysis capabilities!**
