import argparse
import logging

from src.cleaning_data import clean_data
from src.train_test_model import train_test_model
from src.check_scoring import scoring_check

def execute_pipeline(args):
    """
    this function executes the full pipeline
    """

    logging.basicConfig(level=logging.INFO)

    if args.parameter == "all" or args.parameter == "cleaning_data":
        logging.info("Basic cleaning_data procedure is running...")
        clean_data()

    if args.parameter == "all" or args.parameter == "train_test_model":
        logging.info("training and testing the model procedure is running...")
        train_test_model()

    if args.parameter == "all" or args.parameter == "check_scoring":
        logging.info("check model scoring procedure is running...")
        scoring_check()

if __name__ == "__main__":
    """
    the main Entrypoint
    """
    parser = argparse.ArgumentParser(description="ML pipeline")

    parser.add_argument(
        "--parameter",
        type=str,
        required=True,
        choices=["cleaning_data", "train_test_model", "check_scoring", "all"],
        help="ML pipeline actions"
    )

    args = parser.parse_args()

    execute_pipeline(args)
