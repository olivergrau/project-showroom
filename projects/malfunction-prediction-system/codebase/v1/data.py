import pandas as pd

class ScalerTransform:
    def __init__(self, scaler):
        """
        Initializes the ScalerTransform with a pre-fitted scaler.
        
        Args:
            scaler (StandardScaler): A pre-fitted scaler instance.
        """
        self.scaler = scaler

    def transform(self, df):
        """
        Apply the pre-fitted scaler to all numerical columns in the DataFrame.
        
        Args:
            df (pd.DataFrame): Input DataFrame to transform.
        
        Returns:
            pd.DataFrame: Scaled DataFrame.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")
        
        # Apply scaling to all columns
        scaled_values = self.scaler.transform(df.values)
        return pd.DataFrame(scaled_values, columns=df.columns, index=df.index)