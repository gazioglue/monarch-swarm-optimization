# Monarch Swarm Optimization (MSO)

A Python implementation of the Monarch Swarm Optimization algorithm.

## Installation

```bash
pip install monarch-swarm-optimization
## Quick Start

from mso import MSO

# Define your fitness function
def my_fitness_func(solution):
    return sum(solution)  # Example fitness function

# Create and run optimizer
optimizer = MSO(
    pop_size=100,
    max_iter=500,
    obj_type="min",
    neighbour_count=5,
    obj_func=my_fitness_func,
)

best_solution, best_fitness = optimizer.optimize()
## License

This project is licensed under the MIT License - see the LICENSE file for details.
