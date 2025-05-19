# AOA-Semester-Project:
import numpy as np

# Objective function to minimize
def sphere_function(x):
    return np.sum(x**2)

def initialize_forest(pop_size, dim, lb, ub):
    return np.random.uniform(lb, ub, (pop_size, dim))

def tree_growth_algorithm(obj_func, dim, lb, ub, pop_size=30, max_iter=100):
    forest = initialize_forest(pop_size, dim, lb, ub)
    fitness = np.apply_along_axis(obj_func, 1, forest)
    
    best_idx = np.argmin(fitness)
    best_tree = forest[best_idx].copy()
    best_score = fitness[best_idx]

    for t in range(max_iter):
        new_forest = []
        
        for i in range(pop_size):
            tree = forest[i]
            direction = best_tree - tree
            growth_step = 0.2 * np.random.rand() * direction
            new_tree = tree + growth_step
            new_tree = np.clip(new_tree, lb, ub)
            
            # Add some randomness (seed dispersal)
            if np.random.rand() < 0.3:
                random_vector = np.random.uniform(lb, ub, dim)
                new_tree = (new_tree + random_vector) / 2
            
            new_forest.append(new_tree)
        
        forest = np.array(new_forest)
        fitness = np.apply_along_axis(obj_func, 1, forest)

        # Update the best
        current_best_idx = np.argmin(fitness)
        current_best_score = fitness[current_best_idx]

        if current_best_score < best_score:
            best_score = current_best_score
            best_tree = forest[current_best_idx].copy()

        print(f"Iteration {t+1}/{max_iter}, Best Score: {best_score:.5f}")

    return best_tree, best_score

if __name__ == "__main__":
    dim = 30
    lb = -10
    ub = 10
    best_tree, best_val = tree_growth_algorithm(sphere_function, dim, lb, ub)
    print("Best Tree Position:", best_tree)
    print("Best Fitness Value:", best_val)
