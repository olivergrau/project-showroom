import numpy as np

class EarlyStopping:
    """
    Early stopping to terminate training if the evaluation reward does not improve
    by at least min_delta for patience consecutive evaluations.
    """
    def __init__(self, patience=10, min_delta=1e-3, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.best_score = -np.inf
        self.counter = 0
        self.early_stop = False

    def step(self, current_score):
        # Check if there is an improvement by more than min_delta
        if current_score > self.best_score + self.min_delta:
            self.best_score = current_score
            self.counter = 0  # Reset counter if improvement is seen
            
            if self.verbose:
                print(f"EarlyStopping: Improved evaluation score to {self.best_score:.4f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
