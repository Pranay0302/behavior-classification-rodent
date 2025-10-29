# Analysis Methods Documentation

## Overview

This document describes the analytical methods used in the SIMBA Alternative behavioral analysis toolkit. The system employs a multi-layered approach combining velocity-based classification, advanced motif detection, and statistical analysis.

## Core Algorithms

### 1. Velocity Calculation

**Method**: Frame-to-frame displacement calculation
```python
velocity = sqrt((x[t+1] - x[t])² + (y[t+1] - y[t])²)
```

**Implementation**:
- Vectorized operations for efficiency
- Robust error handling for missing data
- NaN padding for edge cases

**Body Parts Analyzed**:
- Nose (primary movement indicator)
- Tail base (body movement)
- Center of mass (overall locomotion)
- Individual body parts (detailed analysis)

### 2. Acceleration Calculation

**Method**: Second derivative of position
```python
acceleration = sqrt((v[t+1] - v[t])²)
```

**Applications**:
- Jumping detection
- Sudden movement changes
- Behavioral transitions

### 3. Angular Velocity

**Method**: Change in movement direction
```python
angular_velocity = arctan2(dy, dx)
angular_change = diff(angular_velocity)
```

**Applications**:
- Circling behavior detection
- Direction change analysis
- Grooming pattern recognition

## Behavioral Classification

### Hierarchical Classification System

The system uses a priority-based approach:

1. **High Priority Behaviors** (Most Specific)
   - Jumping: High acceleration + high velocity
   - Rearing: High nose velocity + low tail velocity + extended body
   - Grooming: Moderate velocity + high angular velocity

2. **Medium Priority Behaviors**
   - Sniffing: Low velocity + high angular velocity + small body
   - Freezing: Very low velocity + low acceleration + extended body
   - Circling: High angular velocity + moderate velocity

3. **Low Priority Behaviors** (Fallback)
   - Velocity-based classification
   - Body length thresholds
   - General movement patterns

### Classification Thresholds

| Behavior | Nose Velocity | Body Length | Angular Velocity | Acceleration |
|----------|---------------|-------------|------------------|--------------|
| Sleeping | < 1.0 | < 60 | - | - |
| Resting | < 1.0 | ≥ 60 | - | - |
| Slow Movement | 1.0 - 5.0 | - | - | - |
| Moderate Movement | 5.0 - 15.0 | - | - | - |
| Fast Movement | > 15.0 | - | - | - |
| Sniffing | < 2.0 | < 80 | > 0.1 | - |
| Freezing | < 0.5 | > 70 | - | < 1.0 |
| Grooming | 1.0 - 8.0 | > 60 | > 0.2 | - |
| Rearing | > 5.0 | > 90 | - | - |
| Exploration | 2.0 - 10.0 | - | < 0.1 | - |
| Thigmotaxis | > 1.0 | - | - | - |
| Circling | 3.0 - 15.0 | - | > 0.3 | - |
| Jumping | > 10.0 | - | - | > 5.0 |

## Advanced Features

### 1. Spatial Analysis

**Distance to Center**:
```python
distance_to_center = sqrt((x - center_x)² + (y - center_y)²)
```

**Applications**:
- Thigmotaxis detection
- Arena exploration patterns
- Spatial preference analysis

### 2. Body Length Analysis

**Calculation**:
```python
body_length = sqrt((nose.x - tail_base.x)² + (nose.y - tail_base.y)²)
```

**Applications**:
- Posture analysis
- Stretching detection
- Body orientation

### 3. Movement Direction

**Calculation**:
```python
movement_direction = arctan2(dy, dx)
```

**Applications**:
- Directional preferences
- Movement patterns
- Behavioral sequences

## Statistical Analysis

### 1. Transition Matrix

**Purpose**: Analyze behavioral sequences and transitions

**Method**:
- Count transitions between behaviors
- Calculate transition probabilities
- Identify common behavioral sequences

**Output**: Probability matrix P(i,j) = P(behavior_j | behavior_i)

### 2. Stability Analysis

**Purpose**: Measure behavioral persistence

**Method**:
- Self-transition probability
- Higher values indicate more stable behaviors

**Formula**:
```python
stability = P(behavior_i | behavior_i)
```

### 3. Entropy Analysis

**Purpose**: Measure behavioral unpredictability

**Method**:
- Shannon entropy of transition probabilities
- Higher entropy = more unpredictable transitions

**Formula**:
```python
entropy = -Σ P(i,j) * log2(P(i,j))
```

### 4. Feature Correlations

**Purpose**: Understand relationships between behavioral features

**Method**:
- Pearson correlation coefficients
- Identify redundant or complementary features

## Performance Optimizations

### 1. Vectorized Operations

- NumPy vectorized calculations
- Pandas groupby operations
- Minimal Python loops

### 2. Memory Management

- Data sampling (every 10th frame)
- Chunked processing for large datasets
- Efficient data types

### 3. Error Handling

- Robust NaN handling
- Graceful degradation
- Comprehensive validation

## Validation Methods

### 1. Schema Validation

- Required column checking
- Data type validation
- Range validation

### 2. Statistical Validation

- Percentage sum validation (must equal 100%)
- Transition probability validation (must sum to 1)
- Correlation matrix validation

### 3. Cross-Validation

- Track-wise analysis
- Temporal consistency checks
- Behavioral plausibility validation

## Quality Metrics

### 1. Data Quality

- Percentage of valid velocity data
- Missing data patterns
- Outlier detection

### 2. Classification Quality

- Behavioral distribution validation
- Transition matrix consistency
- Feature correlation analysis

### 3. Performance Metrics

- Processing time per frame
- Memory usage
- Accuracy validation

## Limitations and Considerations

### 1. Data Requirements

- Minimum frame rate: 10 FPS
- Required body parts: nose, tail_base
- Recommended sampling: every 10th frame

### 2. Behavioral Assumptions

- Thresholds based on typical rodent behavior
- May need adjustment for different species
- Environmental factors not considered

### 3. Computational Limits

- Memory requirements scale with dataset size
- Processing time increases with number of body parts
- Real-time analysis may require optimization

## Future Enhancements

### 1. Machine Learning Integration

- Supervised learning for behavior classification
- Unsupervised clustering for behavior discovery
- Deep learning for complex pattern recognition

### 2. Real-time Analysis

- Streaming data processing
- Online behavioral classification
- Live visualization capabilities

### 3. Multi-species Support

- Species-specific behavioral templates
- Adaptive threshold adjustment
- Cross-species comparison tools
