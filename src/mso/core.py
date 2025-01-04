import numpy as np
import time
from typing import Callable, Optional, Tuple, Dict, Any, Union
from .exceptions import OptimizationError, ConvergenceError, TimeoutError
from .validators import validate_parameters

class MSO:
    """Monarch Swarm Optimization algorithm implementation."""
    
    def __init__(
        self,
        pop_size: int,
        max_iter: int,
        obj_type: str,
        neighbour_count: int,
        obj_func: Callable[[np.ndarray], float],
        dim: Optional[int] = None,
        load_problem_file: Optional[Callable] = None,
        gradient_strength: float = 0.8,
        base_learning_rate: float = 0.1,
        known_optimum: Optional[float] = None,
        tolerance: Optional[float] = None,
        timeout: Optional[int] = None,
        seed: Optional[int] = None
    ):
        """Initialize the MSO optimizer.
        
        Args:
            pop_size: Population size
            max_iter: Maximum number of iterations
            obj_type: Optimization type ('min' or 'max')
            neighbour_count: Number of neighbors to consider
            obj_func: Objective function to optimize
            dim: Problem dimension
            load_problem_file: Function to load problem data
            gradient_strength: Gradient field effect strength
            base_learning_rate: Base learning rate for neighbor influence
            known_optimum: Known optimal solution value
            tolerance: Convergence tolerance
            timeout: Maximum runtime in seconds
            seed: Random seed for reproducibility
        """
        # Store parameters
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.obj_type = obj_type.lower()
        self.neighbour_count = neighbour_count
        self.obj_func = obj_func
        self.gradient_strength = gradient_strength
        self.base_learning_rate = base_learning_rate
        self.known_optimum = known_optimum
        self.tolerance = tolerance
        self.timeout = timeout
        
        # Validate parameters
        validate_parameters(self)
        
        # Set random seed
        if seed is not None:
            np.random.seed(seed)
            
        # Load problem if file loader provided
        self.problem_data = None
        if load_problem_file:
            result = load_problem_file()
            if isinstance(result, tuple):
                self.problem_data, file_optimum = result
                if self.known_optimum is None:
                    self.known_optimum = file_optimum
            else:
                self.problem_data = result
                
        # Set dimension from problem data or parameter
        if dim is None:
            if self.problem_data is not None and hasattr(self.problem_data, 'dim'):
                self.dim = self.problem_data.dim
            else:
                raise ValueError("Problem dimension must be provided when no problem file is loaded")
        else:
            self.dim = dim
            
        # Initialize population and tracking variables
        self.population = self._initialize_population()
        self.fitness_values = np.zeros(pop_size)
        self.best_solution = None
        self.best_fitness = float('inf')
        self.gradient_field = np.zeros(dim)
        self.history = []
        
    def _initialize_population(self) -> np.ndarray:
        """Initialize binary population with smart initialization strategy."""
        population = np.zeros((self.pop_size, self.dim), dtype=int)
        
        # Random initialization
        num_random = self.pop_size // 3
        population[:num_random] = np.random.randint(2, size=(num_random, self.dim))
        
        # Half-filled solutions
        num_half = self.pop_size // 3
        half_filled = np.zeros((num_half, self.dim), dtype=int)
        for i in range(num_half):
            positions = np.random.choice(self.dim, size=self.dim//2, replace=False)
            half_filled[i, positions] = 1
        population[num_random:num_random+num_half] = half_filled
        
        # Dense solutions
        remaining = self.pop_size - (num_random + num_half)
        dense_filled = np.ones((remaining, self.dim), dtype=int)
        for i in range(remaining):
            positions = np.random.choice(self.dim, size=self.dim//4, replace=False)
            dense_filled[i, positions] = 0
        population[num_random+num_half:] = dense_filled
        
        return population
        
    def _is_better(self, fitness1: float, fitness2: float) -> bool:
        """Compare two fitness values based on optimization type."""
        if self.obj_type == 'min':
            return fitness1 < fitness2
        return fitness1 > fitness2
        
    def _calculate_fitness(self, solution: np.ndarray) -> float:
        """Calculate fitness using user-provided function."""
        if self.problem_data is not None:
            return self.obj_func(solution, self.problem_data)
        return self.obj_func(solution)
        
    def _update_gradient_field(self):
        """Update gradient field based on best solution."""
        if self.best_solution is not None:
            self.gradient_field = (self.best_solution - np.mean(self.population, axis=0))
            norm = np.linalg.norm(self.gradient_field)
            if norm > 0:
                self.gradient_field = self.gradient_field / norm
                
    def _update_position(self, current_pos: np.ndarray, gradient_influence: np.ndarray,
                        iteration: int, butterfly_idx: int) -> np.ndarray:
        """Update position of a butterfly."""
        new_pos = current_pos.copy()
        
        # Global best influence
        if self.best_solution is not None:
            gradient_diff = gradient_influence * (self.best_solution - current_pos)
            gradient_change_prob = np.abs(gradient_diff)
            gradient_mask = np.random.random(self.dim) < gradient_change_prob
            new_pos[gradient_mask] = self.best_solution[gradient_mask]
        
        # Neighbor influence
        k = min(self.neighbour_count, self.pop_size-1)
        distances = []
        
        for i in range(self.pop_size):
            if i != butterfly_idx:
                dist = np.sum(np.abs(current_pos - self.population[i]))
                distances.append((dist, i, self.fitness_values[i]))
        
        distances.sort()
        for _, neighbor_idx, neighbor_fitness in distances[:k]:
            if self._is_better(neighbor_fitness, self.fitness_values[butterfly_idx]):
                neighbor = self.population[neighbor_idx]
                learn_mask = np.random.random(self.dim) < self.base_learning_rate
                new_pos[learn_mask] = neighbor[learn_mask]
        
        return new_pos
        
    def _update_butterfly(self, butterfly_idx: int, iteration: int):
        """Update a single butterfly in the population."""
        butterfly = self.population[butterfly_idx].copy()
        gradient_influence = self.gradient_strength * self.gradient_field
        
        new_position = self._update_position(
            butterfly,
            gradient_influence,
            iteration,
            butterfly_idx
        )
        
        # Apply bounds and calculate fitness
        new_position = np.round(np.clip(new_position, 0, 1)).astype(int)
        new_fitness = self._calculate_fitness(new_position)
        
        # Update if better
        if self._is_better(new_fitness, self.fitness_values[butterfly_idx]):
            self.population[butterfly_idx] = new_position
            self.fitness_values[butterfly_idx] = new_fitness
            
            if self._is_better(new_fitness, self.best_fitness):
                self.best_fitness = new_fitness
                self.best_solution = new_position.copy()
                
    def optimize(self, callback: Optional[Callable] = None) -> Tuple[np.ndarray, float]:
        """Run the optimization process.
        
        Args:
            callback: Optional callback function called after each iteration
            
        Returns:
            Tuple of (best_solution, best_fitness)
        """
        start_time = time.time()
        
        # Calculate initial fitness values
        for i in range(self.pop_size):
            self.fitness_values[i] = self._calculate_fitness(self.population[i])
            
        # Initialize best solution
        best_idx = np.argmin(self.fitness_values) if self.obj_type == 'min' else np.argmax(self.fitness_values)
        self.best_solution = self.population[best_idx].copy()
        self.best_fitness = self.fitness_values[best_idx]
        
        # Main optimization loop
        try:
            for iteration in range(self.max_iter):
                # Check timeout
                if self.timeout and time.time() - start_time > self.timeout:
                    raise TimeoutError(f"Optimization exceeded timeout of {self.timeout} seconds")
                    
                # Update gradient field and butterflies
                self._update_gradient_field()
                for butterfly_idx in range(self.pop_size):
                    self._update_butterfly(butterfly_idx, iteration)
                    
                # Record history
                self.history.append({
                    'iteration': iteration,
                    'best_fitness': float(self.best_fitness),
                    'best_solution': self.best_solution.tolist(),
                    'time': time.time() - start_time
                })
                
                # Check for convergence
                if self.known_optimum is not None and self.tolerance is not None:
                    if abs(self.best_fitness - self.known_optimum) <= self.tolerance:
                        print(f"Converged to known optimum within tolerance at iteration {iteration}")
                        break
                        
                # Call callback if provided
                if callback:
                    callback(iteration, self.best_fitness, self.best_solution)
                    
        except Exception as e:
            raise OptimizationError(f"Optimization failed: {str(e)}")
            
        return self.best_solution, self.best_fitness