# Refactored Churn Library with Logging

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
import logging
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import os

# Configure logging
logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def import_data(pth):
    """
    Author: oliver.grau@grausoft.net
    Date Created: 2024-22-11
    
    Returns dataframe for the csv found at pth.

    input:
        pth: a path to the csv
    output:
        df: pandas dataframe
    """
    try:
        df = pd.read_csv(pth)
        logging.info("Successfully loaded data from %s", pth)
        return df
    except Exception as e:
        logging.error("Error loading data from %s: %s", pth, str(e))
        raise


def perform_eda(df):
    """
    Author: oliver.grau@grausoft.net
    Date Created: 2024-22-11
    
    Perform EDA on df and save figures to images folder.

    input:
        df: pandas dataframe
    output:
        None
    """
    try:
        os.makedirs('./images/eda', exist_ok=True)
        logging.info("Performing EDA and saving plots...")

        # Churn histogram
        df['Attrition_Flag'].hist(figsize=(10, 6))
        plt.title('Churn Distribution')
        plt.savefig('./images/eda/1_churn_histogram.png')
        plt.close()

        # Total_Trans_Ct distribution
        plt.figure(figsize=(20, 10))
        sns.histplot(df['Total_Trans_Ct'], stat='density', kde=True)
        plt.savefig('./images/eda/2_total_trans_ct_histplot.png')
        plt.close()

        # Customer Age histogram
        df['Customer_Age'].hist(figsize=(10, 6))
        plt.title('Customer Age Distribution')
        plt.savefig('./images/eda/3_customer_age_histogram.png')
        plt.close()

        # Marital Status distribution
        df['Marital_Status'].value_counts(
            'normalize').plot(kind='bar', figsize=(10, 6))
        plt.title('Marital Status Distribution')
        plt.savefig('./images/eda/4_marital_status_distribution.png')
        plt.close()

        # Correlation heatmap
        df_corr = df.copy()
        df_corr = encoder_helper(
            df_corr,
            ['Marital_Status', 'Education_Level',
                'Income_Category', 'Card_Category']
        )
        numeric_df = df_corr.select_dtypes(include=['number'])
        plt.figure(figsize=(20, 16))
        sns.heatmap(numeric_df.corr(), annot=False,
                    cmap='coolwarm', linewidths=2)
        plt.title('Correlation Heatmap')
        plt.tight_layout()
        plt.savefig('./images/eda/5_correlation_heatmap.png')
        plt.close()

        logging.info("EDA completed successfully.")
    except Exception as e:
        logging.error("Error during EDA: %s", str(e))
        raise


def encoder_helper(df, category_lst, response='Churn', drop=False):
    """
    Author: oliver.grau@grausoft.net
    Date Created: 2024-22-11
    
    Helper function to turn each categorical column into a new column with
    proportion of churn for each category.

    input:
        df: pandas dataframe
        category_lst: list of columns that contain categorical features
        response: string of response name [optional argument for naming]

    output:
        df: pandas dataframe with new columns for encoded categories
    """
    try:
        logging.info("Encoding categorical columns: %s", category_lst)
        df['Churn'] = df['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1)
        for col in category_lst:
            col_churn_mean = df.groupby(col)[response].mean()
            df[f'{col}_Churn'] = df[col].map(col_churn_mean)
            if drop:
                df.drop(col, axis=1, inplace=True)
        logging.info("Encoding completed successfully.")
        return df
    except Exception as e:
        logging.error("Error during encoding: %s", str(e))
        raise


def perform_feature_engineering(df, response='Churn'):
    """
    Author: oliver.grau@grausoft.net
    Date Created: 2024-22-11
    
    Performs feature engineering and splits the dataframe into train and test sets.

    input:
        df: pandas dataframe
        response: string of response column name

    output:
        x_train: x training data
        x_test: x testing data
        y_train: y training data
        y_test: y testing data
    """
    try:
        logging.info("Performing feature engineering...")
        category_columns = ['Gender', 'Education_Level',
                            'Marital_Status', 'Income_Category', 'Card_Category']
        df = df.copy()
        df = encoder_helper(df, category_columns, response='Churn')

        keep_cols = [
            'Customer_Age', 'Dependent_count', 'Months_on_book',
            'Total_Relationship_Count', 'Months_Inactive_12_mon',
            'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
            'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
            'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
            'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
            'Income_Category_Churn', 'Card_Category_Churn'
        ]

        x = df[keep_cols]
        y = df[response]
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.3, random_state=42)
        logging.info("Feature engineering completed successfully.")
        return x_train, x_test, y_train, y_test
    except Exception as e:
        logging.error("Error during feature engineering: %s", str(e))
        raise


def classification_report_image(y_train, y_test, y_train_preds_lr, y_train_preds_rf, y_test_preds_lr, y_test_preds_rf):
    """
    Author: oliver.grau@grausoft.net
    Date Created: 2024-22-11
    
    Produces classification report for training and testing results and stores the report as an image.

    input:
        y_train: training response values
        y_test: test response values
        y_train_preds_lr: training predictions from logistic regression
        y_train_preds_rf: training predictions from random forest
        y_test_preds_lr: test predictions from logistic regression
        y_test_preds_rf: test predictions from random forest
    output:
        None
    """
    try:
        logging.info("Creating classification reports...")

        os.makedirs('./images/results', exist_ok=True)

        train_report = classification_report(y_train, y_train_preds_rf)
        plt.text(0.01, 1.25, 'Random Forest Train', fontproperties='monospace')
        plt.text(0.01, 0.05, train_report, fontproperties='monospace')
        plt.axis('off')
        plt.savefig('./images/results/2_rf_train_report.png')
        plt.close()

        test_report = classification_report(y_test, y_test_preds_rf)
        plt.text(0.01, 1.25, 'Random Forest Test', fontproperties='monospace')
        plt.text(0.01, 0.05, test_report, fontproperties='monospace')
        plt.axis('off')
        plt.savefig('./images/results/3_rf_test_report.png')
        plt.close()

        train_report = classification_report(y_train, y_train_preds_lr)
        plt.text(0.01, 1.25, 'Logistic Regression Train',
                 fontproperties='monospace')
        plt.text(0.01, 0.05, train_report, fontproperties='monospace')
        plt.axis('off')
        plt.savefig('./images/results/0_lr_train_report.png')
        plt.close()

        test_report = classification_report(y_test, y_test_preds_lr)
        plt.text(0.01, 1.25, 'Logistic Regression Test',
                 fontproperties='monospace')
        plt.text(0.01, 0.05, test_report, fontproperties='monospace')
        plt.axis('off')
        plt.savefig('./images/results/1_lr_test_report.png')
        plt.close()

        logging.info("Classification reports created successfully.")
    except Exception as e:
        logging.error("Error creating classification reports: %s", str(e))
        raise


def feature_importance_plot(model, x_data, output_pth):
    """
    Author: oliver.grau@grausoft.net
    Date Created: 2024-22-11
    
    Creates and stores the feature importances in pth.

    input:
        model: model object containing feature_importances_
        X_data: pandas dataframe of X values
        output_pth: path to store the figure

    output:
        None
    """
    try:
        logging.info("Creating feature importance plots...")
        os.makedirs(output_pth, exist_ok=True)

        # Create the SHAP explainer and calculate SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(x_data)

        # Generate the SHAP summary plot and save it
        # Set show=False to suppress immediate display
        shap.summary_plot(shap_values, x_data, plot_type="bar", show=False)
        plt.savefig(f"{output_pth}/8_shap_summary_plot.png", bbox_inches="tight",
                    format='png', dpi=300)  # Save the plot to a file

        plt.close()  # Close the figure to release memory

        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        names = [x_data.columns[i] for i in indices]

        plt.figure(figsize=(25, 10))
        plt.title('Feature Importances', fontsize=16)
        plt.bar(range(x_data.shape[1]), importances[indices], align="center")

        # Adjust x-axis labels
        plt.xticks(
            range(x_data.shape[1]),
            names,
            rotation=45,  # Use a smaller rotation angle for better readability
            ha="right",   # Align labels to the right for better spacing
            fontsize=12   # Increase font size for x-axis labels
        )

        plt.tight_layout()

        plt.savefig(f'{output_pth}/7_feature_importances.png', dpi=300)
        plt.close()

        logging.info("Feature importance plots created successfully.")
    except Exception as e:
        logging.error("Error creating feature importance plots: %s", str(e))
        raise


def train_models(x_train, x_test, y_train, y_test):
    """
    Author: oliver.grau@grausoft.net
    Date Created: 2024-22-11
    
    Train, store model results (images + scores), and save models.

    input:
        X_train: X training data
        X_test: X testing data
        y_train: y training data
        y_test: y testing data
    output:
        None
    """
    try:
        logging.info("Training models...")
        os.makedirs('./models', exist_ok=True)
        os.makedirs('./images/results', exist_ok=True)

        # Grid search
        rfc = RandomForestClassifier(random_state=42)
        lrc = LogisticRegression(solver='lbfgs', max_iter=3000)
        param_grid = {
            'n_estimators': [200, 500],
            'max_features': ['auto', 'sqrt'],
            'max_depth': [4, 5, 100],
            'criterion': ['gini', 'entropy']
        }
        cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
        cv_rfc.fit(x_train, y_train)
        lrc.fit(x_train, y_train)

        y_train_preds_rf = cv_rfc.best_estimator_.predict(x_train)
        y_test_preds_rf = cv_rfc.best_estimator_.predict(x_test)
        y_train_preds_lr = lrc.predict(x_train)
        y_test_preds_lr = lrc.predict(x_test)

        classification_report_image(
            y_train, y_test, y_train_preds_lr, y_train_preds_rf, y_test_preds_lr, y_test_preds_rf)

        joblib.dump(cv_rfc.best_estimator_, './models/rf_model.pkl')
        joblib.dump(lrc, './models/lr_model.pkl')

        feature_importance_plot(cv_rfc.best_estimator_,
                                x_test, './images/results')
        logging.info("Models trained and results saved successfully.")
    except Exception as e:
        logging.error("Error training models: %s", str(e))
        raise
