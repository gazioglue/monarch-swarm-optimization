# Monarch Swarm Optimization (MSO)

A Python implementation of the Monarch Swarm Optimization algorithm, designed for solving binary optimization problems. MSO is inspired by the migration behavior of monarch butterflies and uses a novel approach combining swarm intelligence with gradient-based optimization.

## Features

- Binary optimization support
- Custom fitness function support
- Problem file loading capability
- Early termination options
- Progress tracking and history
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

# Define your fitness function
def my_fitness_func(solution):
    return sum(solution)  # Example fitness function

# Create optimizer
optimizer = MSO(
    pop_size=100,      # Population size
    max_iter=500,      # Maximum iterations
    obj_type="min",    # Optimization type (min/max)
    neighbour_count=5, # Number of neighbors to consider
    obj_func=my_fitness_func,  # Your fitness function
    dim=100,           # Problem dimension
    gradient_strength=0.5,     # Optional parameter
    base_learning_rate=0.1,    # Optional parameter
)

# Run optimization
best_solution, best_fitness = optimizer.optimize()
print(f"Best fitness: {best_fitness}")
print(f"Best solution: {best_solution}")
```

## Advanced Usage

### Custom Problem Files

You can load problem data from custom files:

```python
def read_problem_file(filepath):
    # Your file reading logic here
    return problem_data, known_optimum

optimizer = MSO(
    # ... other parameters ...
    load_problem_file=read_problem_file,
    known_optimum=123456789,  # Optional
    tolerance=1e-6            # Required if known_optimum is set
)
```

### Setting Timeout and Seed

```python
optimizer = MSO(
    # ... other parameters ...
    timeout=3600,  # Maximum runtime in seconds
    seed=42       # Random seed for reproducibility
)
```

### Accessing Optimization History

```python
best_solution, best_fitness = optimizer.optimize()
history = optimizer.history  # Get optimization history
```

## API Reference

### Main Parameters

- `pop_size` (int): Population size (required, >0)
- `max_iter` (int): Maximum iterations (required, >0)
- `obj_type` (str): Optimization type, "min" or "max" (required)
- `neighbour_count` (int): Number of neighbors (required, 1<=neighbour_count<pop_size)
- `obj_func` (callable): Fitness calculation function (required)
- `load_problem_file` (callable): Problem file reader function (optional)
- `gradient_strength` (float): Gradient influence (optional, 0-1, default=0.8)
- `base_learning_rate` (float): Learning rate (optional, 0-1, default=0.1)
- `dim` (int): Problem dimension (required if no problem file is loaded)
- `known_optimum` (float): Known optimal value (optional)
- `tolerance` (float): Convergence tolerance (required if known_optimum is set)
- `timeout` (int): Maximum runtime in seconds (optional)
- `seed` (int): Random seed (optional)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
