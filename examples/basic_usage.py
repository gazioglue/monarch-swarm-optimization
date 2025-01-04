"""
Basic usage example of the MSO algorithm.
This example demonstrates solving a simple binary optimization problem.
"""

from mso import MSO
import numpy as np

def simple_fitness(solution):
    """Simple fitness function: maximize the number of 1s while keeping first bit 0."""
    if solution[0] == 1:  # Constraint: first bit must be 0
        return -1000
    return np.sum(solution[1:])

def main():
    # Create optimizer
    optimizer = MSO(
        pop_size=50,          # Population size
        max_iter=100,         # Maximum iterations
        obj_type='max',       # Maximize the objective
        neighbour_count=3,    # Number of neighbors
        obj_func=simple_fitness,  # Our fitness function
        dim=20,               # 20-bit problem
        gradient_strength=0.8, # Default gradient strength
        base_learning_rate=0.1,# Default learning rate
        timeout=60,           # 60 seconds timeout
        seed=42              # For reproducibility
    )
    
    # Run optimization
    def callback(iteration, best_fitness, best_solution):
        if iteration % 10 == 0:  # Print every 10 iterations
            print(f"Iteration {iteration}: Best Fitness = {best_fitness}")
    
    best_solution, best_fitness = optimizer.optimize(callback=callback)
    
    # Print results
    print("\nOptimization completed!")
    print(f"Best solution: {best_solution}")
    print(f"Best fitness: {best_fitness}")
    
if __name__ == "__main__":
    main()