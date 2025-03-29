import numpy as np

from utils import load_data
import StudentAgent

student_agent = StudentAgent.StudentAgent()  # Create an instance of the StudentAgent class

from joblib import Parallel, delayed

def compute_mae_with_weights(student_agent, weights, validation_data):
    predictions = []
    true_values = []
    
    for (state, true_eval) in validation_data:
        pred_eval = student_agent.evaluate(state, weights)
        predictions.append(pred_eval)
        true_values.append(true_eval)
    
    # Determine min/max across the entire dataset (for normalization)
    min_predicted = min(predictions)
    max_predicted = max(predictions)
    
    absolute_errors = []
    for i in range(len(true_values)):
        if max_predicted == min_predicted:
            normalized_evaluation = 0.0
        else:
            normalized_evaluation = 2 * ((predictions[i] - min_predicted) / (max_predicted - min_predicted)) - 1
        absolute_errors.append(abs(true_values[i] - normalized_evaluation))
    
    mean_abs_error = np.mean(absolute_errors)
    return mean_abs_error

def parallel_random_search(student_agent, validation_data, n_samples=50):
    best_mae = float('inf')
    best_weights = None
    print("Working...")
    possible_values = np.arange(0, 6.01, 0.1)

    # Generate candidates
    candidates = [np.random.choice(possible_values, size=11, replace=True) for _ in range(n_samples)]

    # Evaluate candidates in parallel using all available CPU cores
    results = Parallel(n_jobs=-1)(
        delayed(compute_mae_with_weights)(student_agent, candidate, validation_data)
        for candidate in candidates
    )
    
    # Find the best candidate
    for candidate, mae in zip(candidates, results):
        print("Used these weights:", candidate, "MAE:", mae)
        if mae < best_mae:
            best_mae = mae
            best_weights = candidate

    print("Best MAE (parallel random search):", best_mae)
    print("Best weights found:", best_weights)
    return best_weights, best_mae

# Main usage:
validation_data = load_data()  # load once
best_w, best_err = parallel_random_search(student_agent, validation_data, n_samples=20)