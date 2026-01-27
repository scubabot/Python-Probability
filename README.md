# ML-Resilient Gaussian Process Modeling

This repository implements a machine learning framework for resilient Gaussian Process (GP) modeling under adversarial conditions in a multi-agent setting.

The project combines probabilistic modeling, active learning, and robust aggregation to maintain accurate environmental estimation even when some agents behave maliciously or provide corrupted observations.

## Key Features

- Gaussian Process regression for spatial field estimation  
- Uncertainty-driven active sampling (active learning)  
- Simulation of adversarial agents injecting corrupted observations  
- Robust aggregation using W-MSR–style resilient consensus  
- Comparative evaluation of resilient vs non-resilient learning  
- Visualization of estimation error and uncertainty evolution  

## Example Usage

```bash
python multi_agent_resilience.py
