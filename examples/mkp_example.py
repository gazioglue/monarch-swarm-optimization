"""
Example of solving Multiple Knapsack Problem (MKP) using MSO.
This example shows how to use problem file loading and custom fitness calculation.
"""

from mso import MSO
from mso.utils import save_results
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class MKPData:
    """Class to hold MKP problem data."""
    n_items: int              # Number of items
    n_knapsacks: int         # Number of knapsacks
    profits: np.ndarray      # Profit of each item
    weights: np.ndarray      # Weight matrix [n_items x n_knapsacks]
    capacities: np.ndarray   # Capacity of each knapsack
    known_optimum: float     # Known optimal solution value
    dim: int                 # Problem dimension (same as n_items)

def read_mkp_file(filepath: str = "mkp_instance.txt") -> Tuple[MKPData, float]:
    """Read MKP instance from file."""
    with open(filepath, 'r') as f:
        # Read problem dimensions
        n_items, n_knapsacks = map(int, f.readline().split())
        
        # Read profits
        profits = np.array(list(map(float, f.readline().split())))
        
        # Read weights
        weights = np.zeros((n_items, n_knapsacks))
        for i in range(n_knapsacks):
            weights[:, i] = list(map(float, f.readline().split()))
            
        # Read capacities
        capacities = np.array(list(map(float, f.readline().split())))
        
        # Read known optimum if available
        known_optimum = float(f.readline())
        
        mkp_data = MKPData(
            n_items=n_items,
            n_knapsacks=n_knapsacks,
            profits=profits,
            weights=weights,
            capacities=capacities,
            known_optimum=known_optimum,
            dim=n_items
        )
        
        return mkp_data, known_optimum

def calculate_mkp_fitness(solution: np.ndarray, data: MKPData) -> float:
    """Calculate fitness for MKP solution."""
    # Calculate total profit
    total_profit = np.sum(solution * data.profits)
    
    # Check capacity constraints
    weights_sum = np.dot(solution, data.weights)
    if np.any(weights_sum > data.capacities):
        return -np.sum(np.maximum(0, weights_sum - data.capacities)) * 1000
        
    return total_profit

def main():
    # Create optimizer
    optimizer = MSO(
        pop_size=100,
        max_iter=500,
        obj_type='max',
        neighbour_count=5,
        obj_func=calculate_mkp_fitness,
        load_problem_file=read_mkp_file,
        gradient_strength=0.8,
        base_learning_rate=0.1,
        known_optimum=20.0,    # Known optimum değerini ekledik
        tolerance=1e-6,        # Tolerance değerini ekledik
        timeout=3600,          # 1 hour timeout
        seed=42
    )
    
    # Run optimization
    def callback(iteration, best_fitness, best_solution):
        print(f"Iteration {iteration}: Best Fitness = {best_fitness}")
        # Kontrol toleransı içindeyse
        if abs(best_fitness - optimizer.known_optimum) <= optimizer.tolerance:
            print(f"\nConverged to known optimum ({optimizer.known_optimum}) within tolerance!")
    
    best_solution, best_fitness = optimizer.optimize(callback=callback)
    
    # Print results
    print("\nOptimization completed!")
    print(f"Best solution: {best_solution}")
    print(f"Best fitness: {best_fitness}")
    print(f"Known optimum: {optimizer.known_optimum}")
    print(f"Gap: {abs(best_fitness - optimizer.known_optimum) / optimizer.known_optimum * 100:.2f}%")
    
    # Save results
    results = {
        'best_fitness': best_fitness,
        'best_solution': best_solution.tolist(),
        'parameters': {
            'pop_size': optimizer.pop_size,
            'max_iter': optimizer.max_iter,
            'obj_type': optimizer.obj_type,
            'neighbour_count': optimizer.neighbour_count,
            'gradient_strength': optimizer.gradient_strength,
            'base_learning_rate': optimizer.base_learning_rate
        },
        'history': optimizer.history,
        'known_optimum': optimizer.known_optimum,
        'total_time': optimizer.history[-1]['time'] if optimizer.history else None
    }
    
    output_dir = Path('results')
    saved_file = save_results(results, output_dir, prefix='mkp_example')
    print(f"\nResults saved to: {saved_file}")
    
if __name__ == "__main__":
    main()