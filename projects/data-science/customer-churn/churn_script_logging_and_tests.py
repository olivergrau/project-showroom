import os
import logging
import churn_library as cls

"""
Author: oliver.grau@grausoft.net 
Date Created: 2021-22-11

This modules tests different functions from churn_library module and
log any INFO or ERROR into churn_library.log file. The tests functions
are written to prevent the functions from churn_library module from breaking.
"""

if not os.path.exists('./logs'):
    os.makedirs('./logs')

# Configure logging to both a file and the console
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# FileHandler for logging to a file
file_handler = logging.FileHandler('./logs/churn_library_tests.log')
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)

# StreamHandler for console output
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
stream_handler.setFormatter(stream_formatter)

# Add handlers to the logger
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

if not logger.handlers:  # Prevent adding handlers multiple times
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

def test_import():
    """
    Author: oliver.grau@grausoft.net
    Date Created: 2024-22-11
    
    Tests the data import.
    """
    try:
        df = cls.import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_data: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda():
    """
    Author: oliver.grau@grausoft.net
    Date Created: 2024-22-11
    
    Tests the perform_eda function.
    """
    try:
        df = cls.import_data("./data/bank_data.csv")
        cls.perform_eda(df)
        logging.info("Testing perform_eda: SUCCESS")
    except Exception as err:
        logging.error("Testing perform_eda: FAILED")
        raise err

    # Check if the expected files are created
    try:
        eda_images = [
            "./images/eda/1_churn_histogram.png",
            "./images/eda/2_total_trans_ct_histplot.png",
            "./images/eda/3_customer_age_histogram.png",
            "./images/eda/4_marital_status_distribution.png",
            "./images/eda/5_correlation_heatmap.png",
        ]
        for image_path in eda_images:
            assert os.path.exists(image_path)
        logging.info(
            "Testing perform_eda: All EDA images created successfully")
    except AssertionError as err:
        logging.error(
            "Testing perform_eda: Not all EDA images were created")
        raise err


def test_encoder_helper():
    """
    Author: oliver.grau@grausoft.net
    Date Created: 2024-22-11
    
    Tests the encoder_helper function.
    """
    try:
        df = cls.import_data("./data/bank_data.csv")
        category_lst = ['Gender', 'Education_Level', 'Marital_Status',
                        'Income_Category', 'Card_Category']
        df_encoded = cls.encoder_helper(df, category_lst)
        logging.info("Testing encoder_helper: SUCCESS")
    except Exception as err:
        logging.error("Testing encoder_helper: FAILED")
        raise err

    # Check that new columns are created
    try:
        for category in category_lst:
            new_col = f"{category}_Churn"
            assert new_col in df_encoded.columns
        logging.info(
            "Testing encoder_helper: All new columns created successfully")
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: Not all new columns were created")
        raise err


def test_perform_feature_engineering():
    """
    Author: oliver.grau@grausoft.net
    Date Created: 2024-22-11
    
    Tests the perform_feature_engineering function.
    """
    try:
        df = cls.import_data("./data/bank_data.csv")
        x_train, x_test, y_train, y_test = cls.perform_feature_engineering(df)
        logging.info("Testing perform_feature_engineering: SUCCESS")
    except Exception as err:
        logging.error("Testing perform_feature_engineering: FAILED")
        raise err

    # Check that the outputs are not empty and have the correct shapes
    try:
        assert len(x_train) > 0
        assert len(x_test) > 0
        assert len(y_train) > 0
        assert len(y_test) > 0
        logging.info(
            "Testing perform_feature_engineering: Data split successfully")
    except AssertionError as err:
        logging.error(
            "Testing perform_feature_engineering: Data split failed")
        raise err


def test_train_models():
    """
    Author: oliver.grau@grausoft.net
    Date Created: 2024-22-11
    
    Tests the train_models function.
    """
    try:
        df = cls.import_data("./data/bank_data.csv")
        x_train, x_test, y_train, y_test = cls.perform_feature_engineering(df)
        cls.train_models(x_train, x_test, y_train, y_test)
        logging.info("Testing train_models: SUCCESS")
    except Exception as err:
        logging.error("Testing train_models: FAILED")
        raise err

    # Check that models are saved
    try:
        assert os.path.exists("./models/rf_model.pkl")
        assert os.path.exists("./models/lr_model.pkl")
        logging.info("Testing train_models: Models saved successfully")
    except AssertionError as err:
        logging.error("Testing train_models: Models not saved")
        raise err

    # Check that the images are created
    result_images = [
        "./images/results/0_lr_train_report.png",
        "./images/results/1_lr_test_report.png",
        "./images/results/2_rf_train_report.png",
        "./images/results/3_rf_test_report.png",
        "./images/results/7_feature_importances.png",
        "./images/results/8_shap_summary_plot.png",
    ]
    try:
        for image_path in result_images:
            assert os.path.exists(image_path)
        logging.info(
            "Testing train_models: All result images created successfully")
    except AssertionError as err:
        logging.error(
            "Testing train_models: Not all result images were created")
        raise err


if __name__ == "__main__":
    pass
