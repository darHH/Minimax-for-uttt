
# Run the following cell to import utilities
import numpy as np

from utils import State, Action, load_data
import StudentAgent
student_agent = StudentAgent.StudentAgent()  # Create an instance of the StudentAgent class


def validate_evaluation_function(student_agent):
    # Load the dataset
    validation_data = load_data()

    # Initialize error tracking
    absolute_errors = []
    signed_errors = []
    out_of_range_predictions = []
    
    # Iterate through the dataset
    for idx, (state, true_evaluation) in enumerate(validation_data):
        # Use your agent's evaluate method on the state
        predicted_evaluation = student_agent.evaluate(state)
        
        # Check for out of range predictions
        if not (-1 <= predicted_evaluation <= 1):
            out_of_range_predictions.append({
                'index': idx,
                'predicted': predicted_evaluation,
                'true_value': true_evaluation
            })
        
        # Calculate errors
        absolute_error = abs(predicted_evaluation - true_evaluation)
        signed_error = predicted_evaluation - true_evaluation
        absolute_errors.append(absolute_error)
        signed_errors.append(signed_error)

    # Calculate metrics
    metrics = {
        'validation_samples': len(validation_data),
        'out_of_range_count': len(out_of_range_predictions),
        'mean_absolute_error': np.mean(absolute_errors),
        'median_absolute_error': np.median(absolute_errors),
        'max_absolute_error': np.max(absolute_errors),
        'std_absolute_error': np.std(absolute_errors),
        'mean_signed_error': np.mean(signed_errors),
        # 'out_of_range_predictions': out_of_range_predictions
    }
    
    # Print results in a copyable format
    print("Validation Results:")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    
    return metrics

# Usage
validation_results = validate_evaluation_function(student_agent)