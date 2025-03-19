import numpy as np

class EarlyStopping:
    """
    Early stopping to terminate training if the evaluation reward does not improve
    by at least min_delta for patience consecutive evaluations, or if the evaluation 
    reward is zero for zero_patience consecutive evaluations, or if the actor loss 
    increases (i.e. becomes less negative) for actor_patience consecutive evaluations.
    """
    def __init__(self, patience=10, min_delta=1e-3, verbose=True, zero_patience=3,
                 actor_patience=5, actor_min_delta=1e-3):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.zero_patience = zero_patience
        self.actor_patience = actor_patience
        self.actor_min_delta = actor_min_delta

        self.best_score = -np.inf
        self.counter = 0
        self.zero_counter = 0

        # For actor loss, lower (more negative) is better.
        self.best_actor_loss = None  # To be set on first call.
        self.actor_counter = 0

        self.early_stop = False

        print(f"EarlyStopping: Patience: {patience}, Min delta: {min_delta}, Zero patience: {zero_patience}, "
              f"Actor patience: {actor_patience}, Actor min delta: {actor_min_delta}")

    def stepAvgReward(self, current_score):
        # Check if the current evaluation reward is exactly zero.
        if current_score == 0:
            self.zero_counter += 1
            if self.verbose:
                print(f"EarlyStopping: Current score is zero. Zero counter: {self.zero_counter} out of {self.zero_patience}")
        else:
            self.zero_counter = 0

        # Check if current_score has improved.
        if current_score > self.best_score + self.min_delta:
            self.best_score = current_score
            self.counter = 0  # Reset if improvement is seen.
            if self.verbose:
                print(f"EarlyStopping: Improved evaluation score to {self.best_score:.4f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")

        # Trigger early stopping if any counter exceeds its threshold.
        if self.counter >= self.patience or self.zero_counter >= self.zero_patience:
            self.early_stop = True

        return self.early_stop

    def stepActorLoss(self, current_actor_loss):        
        # Now, check the actor loss criterion if provided.
        if current_actor_loss is not None:
            if self.best_actor_loss is None:
                self.best_actor_loss = current_actor_loss
            else:
                # For actor loss, lower (more negative) is better.
                # If current actor loss is greater (i.e. less negative) than the best by at least actor_min_delta,
                # then we consider that an increase.
                if current_actor_loss > self.best_actor_loss + self.actor_min_delta:
                    self.actor_counter += 1
                    if self.verbose:
                        print(f"EarlyStopping: Actor loss increased. Actor counter: {self.actor_counter} out of {self.actor_patience}")
                else:
                    self.actor_counter = 0
                    self.best_actor_loss = current_actor_loss

        # Trigger early stopping if any counter exceeds its threshold.
        if self.actor_counter >= self.actor_patience:
            self.early_stop = True

        return self.early_stop

