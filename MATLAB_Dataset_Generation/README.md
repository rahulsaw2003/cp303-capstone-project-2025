# MATLAB Dataset Generation for Soil Moisture Modeling

## Overview
This repository contains MATLAB code for generating synthetic datasets for soil moisture modeling using Richards' equation and LSTM neural networks. The work focuses on simulating soil moisture dynamics under various irrigation conditions and creating training datasets for machine learning applications in precision agriculture and irrigation control systems.

## Project Context and Motivation

### Problem Statement
Traditional irrigation systems often operate on fixed schedules without considering real-time soil moisture conditions, leading to water waste and suboptimal crop growth. This project addresses this by:
1. **Modeling Soil-Water Dynamics**: Using physics-based models to understand how irrigation affects soil moisture
2. **Generating Training Data**: Creating synthetic datasets for machine learning models
3. **Predictive Control**: Enabling real-time irrigation decisions based on predicted soil moisture

### Research Applications
- **Precision Agriculture**: Optimize water usage for different crop types
- **Climate Change Adaptation**: Study soil moisture under varying weather patterns
- **Smart Irrigation Systems**: Develop automated irrigation controllers
- **Soil Science Research**: Validate numerical models against theoretical predictions

## Mathematical Foundation

### Richards' Equation
The core of this work is based on Richards' equation, which describes water flow in unsaturated porous media:

```
∂θ/∂t = ∇·[K(h)∇(h + z)] - S(h)
```

Where:
- `θ` is the volumetric water content
- `h` is the capillary pressure head
- `K(h)` is the hydraulic conductivity
- `S(h)` is the sink term (evapotranspiration)
- `z` is the vertical coordinate

### Soil Hydraulic Functions - Van Genuchten-Mualem Model

#### Soil Moisture Retention Curve θ(h)
```
θ(h) = θr + (θs - θr) / [1 + (-αh)^n]^(1-1/n)
```

#### Hydraulic Conductivity K(h)
```
K(h) = Ks × [1 + (-αh)^n]^(-1+1/n) × [1 - (1 + (-αh)^n)^(-1+1/n)]^(1-1/n)
```

#### Soil Moisture Capacity C(h)
```
C(h) = (θs - θr) × α × n × (1-1/n) × (-αh)^(n-1) × [1 + (-αh)^n]^(1/n-2)
```

## Code Components

### 1. Main Simulation Script (`main.m`)
- **Purpose**: Orchestrates the entire simulation process
- **Key Features**:
  - Generates random irrigation profiles with 65 steps
  - Solves the PDE system using ODE45
  - Implements boundary conditions
  - Saves results to CSV and MAT files

**Parameters**:
- `alpha = 7.5` (Van Genuchten parameter)
- `Ks = 1.23×10⁻⁵` m/s (Saturated hydraulic conductivity)
- `θs = 0.41` (Saturated water content)
- `θr = 0.538` (Residual water content)
- `n = 2` (Van Genuchten parameter)
- `delZ = 0.0192` m (Spatial discretization)

### 2. PDE Solver (`pde_model.m`)
- **Purpose**: Implements the finite difference scheme for Richards' equation
- **Features**:
  - 26-node spatial discretization
  - Implicit boundary conditions
  - Handles both internal nodes and boundary nodes
  - Incorporates evapotranspiration effects

### 3. Hydraulic Functions
- **`K_function.m`**: Computes hydraulic conductivity K(h)
- **`C_function.m`**: Computes soil moisture capacity C(h)
- **`calc_soil_moisture.m`**: Converts capillary head to moisture content

### 4. Irrigation Signal Generation (`gen_irrigation_signal.m`)
- **Purpose**: Creates realistic irrigation patterns
- **Features**:
  - Random irrigation rates between 10⁻⁷ and 10⁻⁵ m/s
  - Variable step durations (8-12 time units)
  - Step-wise constant irrigation rates

### 5. LSTM Models
- **`lsm_model.m`**: Surface node prediction model
- **`lstm_global_model.m`**: Global prediction model
- **Architecture**: 50 hidden units, sequence-to-sequence mapping
- **Training**: 70% training, 30% testing split

## Usage Instructions

### Prerequisites
- MATLAB R2018b or later
- Deep Learning Toolbox (for LSTM models)
- Optimization Toolbox (for fsolve)

### Running the Simulation
1. **Generate Dataset**:
   ```matlab
   main
   ```
   This will create `simulation_data.csv` and `simulation_data.mat`

2. **Train LSTM Model**:
   ```matlab
   lsm_model
   ```
   or
   ```matlab
   lstm_global_model
   ```

### Output Files
- **`simulation_data.csv`**: Irrigation rates and soil moisture profiles
- **`simulation_data.mat`**: MATLAB format with input/output matrices

## Dataset Characteristics

### Input Features
- **Irrigation Rate**: Time-varying irrigation signal (m/s)
- **Time Steps**: Variable duration steps with random irrigation values

### Output Variables
- **Soil Moisture**: Volumetric water content at surface node
- **Capillary Head**: Pressure head values at all 26 nodes

### Data Size
- **Samples**: 65 irrigation steps
- **Features**: 1 (irrigation rate)
- **Targets**: 1 (surface soil moisture)

## Key Features

### 1. Realistic Irrigation Patterns
- Random irrigation rates within physically meaningful ranges
- Variable step durations to simulate real-world conditions
- Smooth transitions between irrigation levels

### 2. Physical Consistency
- Richards' equation ensures physically realistic water flow
- Van Genuchten-Mualem model for soil hydraulic properties
- Proper boundary conditions and numerical stability

### 3. Machine Learning Ready
- Structured input-output pairs for training
- Multiple data formats (CSV, MAT)
- LSTM models for time series prediction

### 4. Modular Design
- Separate functions for each physical process
- Easy to modify parameters and boundary conditions
- Extensible for different soil types and conditions

## Applications

### 1. Irrigation Control
- Predict soil moisture response to irrigation
- Optimize irrigation scheduling
- Reduce water waste

### 2. Crop Management
- Monitor soil moisture levels
- Predict drought stress
- Optimize planting times

### 3. Research and Education
- Study soil-water interactions
- Validate numerical models
- Train machine learning models

## Extending the Work

### 1. Additional Soil Types
- Implement different Van Genuchten parameters
- Add layered soil profiles
- Include temperature effects

### 2. Advanced Irrigation Patterns
- Drip irrigation simulation
- Sprinkler system modeling
- Weather-dependent irrigation

### 3. Machine Learning Improvements
- Attention mechanisms in LSTM
- Transformer-based models
- Multi-task learning for multiple nodes

### 4. Real-time Control
- Model Predictive Control (MPC)
- Reinforcement learning for irrigation optimization
- Sensor data integration

## Performance Considerations

### 1. Computational Efficiency
- Vectorized operations for hydraulic functions
- Parallel processing for multiple simulations
- Optimized numerical solvers

### 2. Memory Management
- Single precision for large datasets
- Streaming for very large simulations
- Compressed data storage

### 3. Numerical Stability
- Adaptive time stepping in ODE45
- Proper boundary condition handling
- Convergence monitoring

## Troubleshooting

### Common Issues
1. **Convergence Problems**: Check boundary condition parameters and initial guesses
2. **Memory Issues**: Reduce number of time steps or use single precision
3. **Physical Unrealism**: Validate parameters against literature values

### Validation Steps
- Check mass conservation (input ≈ storage change)
- Verify soil moisture bounds (θr ≤ θ ≤ θs)
- Monitor numerical convergence

## References

1. Richards, L.A. (1931). "Capillary conduction of liquids through porous mediums"
2. Van Genuchten, M.T. (1980). "A closed-form equation for predicting the hydraulic conductivity of unsaturated soils"
3. Mualem, Y. (1976). "A new model for predicting the hydraulic conductivity of unsaturated porous media"
4. Celia, M.A., et al. (1990). "A general mass-conservative numerical solution for the unsaturated flow equation"
