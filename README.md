# Monarch Swarm Optimization (MSO)

A Python implementation of the Monarch Swarm Optimization algorithm, designed for solving binary optimization problems. MSO is inspired by the migration behavior of monarch butterflies and uses a novel approach combining swarm intelligence with gradient-based optimization.

## Features

- Binary optimization for various problem types
- Built-in command line interface
- Automatic result saving and history tracking
- Early stopping with known optimum
- Automatic progress reporting
- Custom problem file loading support
- Customizable algorithm parameters
- Built-in timeout mechanism
- Reproducible results with seed setting

## Installation

You can install MSO using pip:

```bash
pip install monarch-swarm-optimization
```

## Quick Start

Here's a simple example to get you started:

```python
from mso import MSO
import numpy as np

def simple_fitness(solution):
    """Example fitness function: maximize sum of elements."""
    return np.sum(solution)

# Run optimization
MSO.run(
    obj_func=simple_fitness,  # Your fitness function
    dim=20,                  # Problem dimension
    pop_size=50,            # Population size
    max_iter=100,           # Maximum iterations
    obj_type='max',         # Maximize the objective
    neighbour_count=3       # Number of neighbors
)
```

Run your script with command line options:
```bash
# Basic run
python your_script.py

# Save results
python your_script.py --save-results yes

# Specify results directory
python your_script.py --save-results yes --results-dir my_results

# Set random seed
python your_script.py --seed 42
```

## Using With Problem Files

For problems that require loading data from files (e.g., Multiple Knapsack Problem):

```python
from mso import MSO
import numpy as np
from dataclasses import dataclass

@dataclass
class ProblemData:
    """Your problem data structure."""
    dim: int
    weights: np.ndarray
    # ... other problem-specific data

def read_problem_file(filepath: str):
    """Your file reading function."""
    # Read and parse your problem file
    data = ProblemData(...)
    known_optimum = ...  # Optional
    return data, known_optimum

def calculate_fitness(solution, data):
    """Your fitness calculation."""
    # Calculate fitness using solution and problem data
    return fitness_value

# Run optimization
MSO.run(
    obj_func=calculate_fitness,
    dim=None,  # Will be set from problem file
    load_problem_file=read_problem_file,
    known_optimum=123.45,  # Optional
    tolerance=1e-6,        # Required if known_optimum is set
    pop_size=100,
    max_iter=500,
    obj_type='max'
)
```

## API Reference

### Main Parameters

Required:
- `obj_func`: Objective function to optimize
- `dim`: Problem dimension (or None if using problem file)
- `pop_size`: Population size (>0)
- `max_iter`: Maximum iterations (>0)
- `obj_type`: 'min' or 'max'
- `neighbour_count`: Number of neighbors (1 <= n < pop_size)

Optional:
- `load_problem_file`: Function to load problem data
- `gradient_strength`: Gradient influence (0-1, default=0.8)
- `base_learning_rate`: Learning rate (0-1, default=0.1)
- `known_optimum`: Known optimal value
- `tolerance`: Convergence tolerance
- `timeout`: Maximum runtime in seconds
- `seed`: Random seed for reproducibility

### Command Line Arguments

- `--save-results`: Whether to save results ('yes' or 'no', default='no')
- `--results-dir`: Directory to save results (default='results')
- `--seed`: Random seed for reproducibility

## Examples

The package includes several example implementations:

1. Basic binary optimization
2. Multiple Knapsack Problem (MKP)

Check the `examples/` directory in the repository for complete examples.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.