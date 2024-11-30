# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
- **Model Type**: Random Forest Classifier
- **Library Used**: scikit-learn
- **Task**: Binary Classification
- **Target Variable**: `salary`
    - Values: `<=50K` and `>50K`
- **Features**:
    - **Categorical**: `workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `native-country`
    - **Continuous**: `age`, `fnlgt`, `education-num`, `capital-gain`, `capital-loss`, `hours-per-week`

## Intended Use
The model predicts whether an individual's annual salary is greater than $50K (`>50K`) or less than or equal to $50K (`<=50K`) based on their demographic and work-related information. It is intended for educational purposes and exploratory data analysis, not for real-world financial decision-making.

## Training Data
- **Source**: Modified version of the UCI Adult Census Income Dataset.
- **Size**: ~32,561 rows in the full dataset.
- **Preprocessing**:
    - Categorical features were one-hot encoded.
    - Continuous features were left unchanged.
    - Labels (`salary`) were binarized as `0` for `<=50K` and `1` for `>50K`.
- **Splitting**: 80% of the data used for training.

## Evaluation Data
- **Source**: Same as training data (UCI Adult Census Income Dataset).
- **Size**: ~20% of the data held out for testing (~6,512 rows).
- **Preprocessing**:
    - Same preprocessing steps as the training data.
- **Slicing**: Model performance was evaluated on slices of the test data based on the unique values of categorical features.

## Metrics
- **Overall Performance**:
    - **Precision**: 88%
    - **Recall**: 77%
    - **F1 Score**: 82%

- **Performance on Selected Slices**:
    - `workclass=Private`:
        - Precision: 88%, Recall: 76%, F1 Score: 82%
    - `education=Bachelors`:
        - Precision: 88%, Recall: 89%, F1 Score: 88%
    - Additional slice metrics available in the slice testing output.

## Ethical Considerations
1. **Bias in Data**: The dataset may contain inherent biases related to gender, race, or socioeconomic status, which can influence the model's predictions.
2. **Fairness**: Disparities in performance across different demographic groups must be carefully monitored and mitigated.
3. **Real-World Implications**: Predictions from this model should not be used to make consequential decisions, such as hiring or salary adjustments, without additional checks for fairness and accuracy.
4. **Data Privacy**: The dataset contains anonymized data, but users should be cautious when applying the model to sensitive information.

## Caveats and Recommendations
- The model's performance is tied to the quality and representativeness of the training data. It may not generalize well to populations outside the dataset's scope.
- Ensure that fairness metrics are computed and monitored for all subgroups before using the model in real-world applications.
- Interpret the predictions cautiously, especially for underrepresented groups in the dataset.
- Future work could include bias mitigation techniques or reweighting data for fairness.
