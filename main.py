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

    actions = {
        'cleaning_data': clean_data,
        'train_test_model': train_test_model,
        'check_scoring': scoring_check
    }

    if parameter == 'all':
        for action in actions.values():
            action()
    else:
        if parameter in actions:
            logging.info(f"{parameter.replace('_', ' ').capitalize()} procedure is running...")
            actions[parameter]()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the ML pipeline with specified parameters."
    )
    parser.add_argument(
        "--parameter",
        type=str,
        required=True,
        choices=["cleaning_data", "train_test_model", "check_scoring", "all"],
        help="Specify the ML pipeline action to perform"
    )

    args = parser.parse_args()
    execute_pipeline(args.parameter)
