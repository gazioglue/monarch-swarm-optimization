# MSO Usage Guide

## Basic Usage

The simplest way to use MSO is with a basic fitness function:

```python
from mso import MSO
import numpy as np

def fitness(solution):
    return np.sum(solution)  # Example: maximize sum of 1s

MSO.run(
    obj_func=fitness,
    dim=20,
    pop_size=50,
    max_iter=100,
    obj_type='max'
)
```

## Command Line Interface

Every MSO script automatically supports these arguments:

```bash
# Save results
python your_script.py --save-results yes

# Custom save directory
python your_script.py --save-results yes --results-dir custom_dir

# Set random seed
python your_script.py --seed 42
```

## Loading Problem Files

For problems that require data files:

```python
from dataclasses import dataclass
import numpy as np

@dataclass
class ProblemData:
    dim: int
    constraints: np.ndarray
    # ... other fields

def read_file(filepath="problem.txt"):
    # Read your file
    data = ProblemData(...)
    known_optimum = 123.45  # Optional
    return data, known_optimum

def fitness(solution, data):
    # Use data in fitness calculation
    return value

MSO.run(
    obj_func=fitness,
    load_problem_file=read_file,
    dim=None,  # Will be set from file
    known_optimum=123.45,
    tolerance=1e-6
)
```

## Result Files

When saving is enabled, MSO creates JSON files containing:
- Best solution found
- Best fitness value
- Full optimization history
- All parameters used
- Execution time
- Progress tracking

Example result file structure:
```json
{
    "best_fitness": 123.45,
    "best_solution": [1, 0, 1, ...],
    "parameters": {
        "pop_size": 50,
        "max_iter": 100,
        ...
    },
    "history": [
        {
            "iteration": 0,
            "best_fitness": 100.0,
            "time": 0.1
        },
        ...
    ]
}
```

## Early Stopping

MSO supports two types of early stopping:

1. Known optimum:
```python
MSO.run(
    ...,
    known_optimum=123.45,
    tolerance=1e-6
)
```

2. Timeout:
```python
MSO.run(
    ...,
    timeout=3600  # Stop after 1 hour
)
```

## Advanced Usage

### Custom Callback

You can provide a custom callback function:

```python
def my_callback(iteration, best_fitness, best_solution):
    print(f"Custom logging: iter={iteration}, fitness={best_fitness}")

optimizer = MSO(
    obj_func=fitness,
    dim=20,
    ...
)
optimizer.optimize(callback=my_callback)
```

### Alternative Parameter Settings

Different parameter combinations for different problems:

```python
# Exploration focused
MSO.run(
    obj_func=fitness,
    dim=20,
    gradient_strength=0.5,    # Lower gradient influence
    base_learning_rate=0.2,   # Higher learning rate
    neighbour_count=5         # More neighbors
)

# Exploitation focused
MSO.run(
    obj_func=fitness,
    dim=20,
    gradient_strength=0.9,    # Higher gradient influence
    base_learning_rate=0.05,  # Lower learning rate
    neighbour_count=2         # Fewer neighbors
)
```