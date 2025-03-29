import numpy as np

from utils import load_data
import StudentAgentNN

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import MinMaxScaler

student_agent = StudentAgentNN.StudentAgent()  # Create an instance of the StudentAgent class
class WeightTuner(BaseEstimator, RegressorMixin):
    def __init__(self, weights=None):
        self.weights = weights if weights is not None else np.ones(11)
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
    
    def fit(self, X, y):
        return self
    
    def predict(self, X):
        # Collect predictions
        predictions = []
        
        # Collect all predicted values first to normalize
        raw_predictions = []
        for state in X:
            pred = self.student_agent.evaluate(state, self.weights)
            raw_predictions.append(pred)
        
        # Normalize predictions to [-1, 1] range
        if len(raw_predictions) > 1:
            # Use MinMaxScaler to normalize predictions
            normalized_predictions = self.scaler.fit_transform(
                np.array(raw_predictions).reshape(-1, 1)
            ).flatten()
        else:
            # If only one prediction, just use the prediction
            normalized_predictions = [0] if not raw_predictions else [raw_predictions[0]]
        
        return np.array(normalized_predictions)

def prepare_data(student_agent):
    # Load validation data
    validation_data = load_data()
    
    # Separate states and true evaluations
    X = [state for state, _ in validation_data]
    y = [true_eval for _, true_eval in validation_data]
    
    return X, y

def optimize_weights_sklearn(student_agent):
    # Prepare data
    X, y = prepare_data(student_agent)
    
    # Create the custom estimator
    tuner = WeightTuner()
    tuner.student_agent = student_agent  # Attach student agent
    
    # Define parameter grid
    param_grid = {
        'weights': [
            np.random.uniform(0, 5, 11) for _ in range(20)  # 20 random initial weight sets
        ]
    }
    
    # Set up GridSearchCV
    grid_search = GridSearchCV(
        estimator=tuner,
        param_grid=param_grid,
        scoring='neg_mean_absolute_error',  # Scikit-learn uses negative MAE
        cv=3,  # 3-fold cross-validation
        n_jobs=-1  # Use all available cores
    )
    
    # Fit the grid search
    grid_search.fit(X, y)
    
    # Best parameters and score
    print("Best weights:", grid_search.best_params_['weights'])
    print("Best MAE:", -grid_search.best_score_)
    
    return grid_search.best_params_['weights'], -grid_search.best_score_

def optimize_with_randomized_search(student_agent):
    # Prepare data
    X, y = prepare_data(student_agent)
    
    # Create the custom estimator
    tuner = WeightTuner()
    tuner.student_agent = student_agent  # Attach student agent
    
    # Define parameter distribution
    param_distributions = {
        'weights': [
            np.random.uniform(0, 5, 11) for _ in range(100)  # 100 random weight sets
        ]
    }
    
    # Set up RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=tuner,
        param_distributions=param_distributions,
        n_iter=50,  # Number of parameter settings sampled
        scoring='neg_mean_absolute_error',
        cv=3,
        n_jobs=-1,
        random_state=42
    )
    
    # Fit the random search
    random_search.fit(X, y)
    
    # Best parameters and score
    print("Best weights:", random_search.best_params_['weights'])
    print("Best MAE:", -random_search.best_score_)
    
    return random_search.best_params_['weights'], -random_search.best_score_