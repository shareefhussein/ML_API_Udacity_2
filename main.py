"""
This module is designed to execute a machine learning pipeline which includes data cleaning, 
model training and testing, and model scoring checks based on specified parameters.
"""

import argparse
import logging

from src.cleaning_data import clean_data
from src.train_test_model import train_test_model
from src.check_scoring import scoring_check


def execute_pipeline(parameter):
    """
    Execute the full pipeline based on the specified parameter.
    
    :param parameter: A string specifying which part of the pipeline to run. 
                      Can be 'cleaning_data', 'train_test_model', 'check_scoring', or 'all'.
    """
    logging.basicConfig(level=logging.INFO)

    if parameter in ('all', 'cleaning_data'):
        logging.info("Basic cleaning_data procedure is running...")
        clean_data()

    if parameter in ('all', 'train_test_model'):
        logging.info("Training and testing the model procedure is running...")
        train_test_model()

    if parameter in ('all', 'check_scoring'):
        logging.info("Check model scoring procedure is running...")
        scoring_check()


if __name__ == "__main__":
    # The main entry point of the script
    parser = argparse.ArgumentParser(description="Run the ML pipeline with specified parameters.")
    parser.add_argument(
        "--parameter",
        type=str,
        required=True,
        choices=["cleaning_data", "train_test_model", "check_scoring", "all"],
        help="Specify the ML pipeline action to perform"
    )

    args = parser.parse_args()
    execute_pipeline(args.parameter)
