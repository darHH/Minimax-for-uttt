import numpy as np

from utils import State, Action, load_data
import StudentAgent

student_agent = StudentAgent.StudentAgent()  # Create an instance of the StudentAgent class

def validate_evaluation_function(student_agent):
    # Load the dataset
    validation_data = load_data()

    # Lists for storing predictions and true values
    predictions = []
    true_values = []
    normalized_values = []
    absolute_errors = []
    total = 0
    
    # Evaluate on each data point
    for (state, true_evaluation) in validation_data:
        predicted_evaluation = student_agent.evaluate(state)
        predictions.append(predicted_evaluation)
        true_values.append(true_evaluation)
    
    # Determine min/max across the entire dataset (for normalization)
    min_predicted = min(predictions)
    max_predicted = max(predictions)

    for i in range(len(true_values)):
        normalized_evaluation = 2 * ((predictions[i] - min_predicted) / (max_predicted - min_predicted) - 1)
        normalized_values.append(normalized_evaluation)
        print(true_values[i], (normalized_evaluation))
        absolute_errors.append(abs(true_values[i] - normalized_evaluation))
        total += 1

    print(total)
    print(min_predicted, max_predicted)
    print("mean_abs_err =", np.mean(absolute_errors))
    

validation_results = validate_evaluation_function(student_agent)