import argparse
import importlib
import inspect
import os
import sys
from os import path

FAIL_COLOR = '\033[91m'
OK_COLOR = '\033[92m'
WARN_COLOR = '\033[93m'


def run_sanity_check(test_dir):
    print('This script will perform a sanity test to ensure your code meets the criteria in the rubric.\n')
    print('Please enter the path to the file that contains your test cases for the GET() and POST() methods')
    print('The path should be something like abc/def/test_xyz.py')
    filepath = input('> ')

    assert path.exists(filepath), f"File {filepath} does not exist."
    sys.path.append(path.dirname(filepath))

    module_name = path.splitext(path.basename(filepath))[0]
    module = importlib.import_module(module_name)

    test_function_names = [
        x for x in dir(module)
        if inspect.isfunction(getattr(module, x)) and not x.startswith('__')
    ]

    test_functions_for_get = [
        x for x in test_function_names
        if '.get(' in inspect.getsource(getattr(module, x))
    ]
    test_functions_for_post = [
        x for x in test_function_names
        if '.post(' in inspect.getsource(getattr(module, x))
    ]

    print("\n============= Sanity Check Report ===========")
    sanity_test_passing = True
    warning_count = 1

    # GET()
    test_for_get_method_response_code = False
    test_for_get_method_response_body = False
    if not test_functions_for_get:
        print(FAIL_COLOR + f"[{warning_count}]")
        warning_count += 1
        print(FAIL_COLOR + "No test cases were detected for the GET() method.")
        print(
            FAIL_COLOR +
            "\nPlease make sure you have a test case for the GET method. "
            "This MUST test both the status code as well as the contents of the request object.\n"
        )
        sanity_test_passing = False
    else:
        for func in test_functions_for_get:
            source = inspect.getsource(getattr(module, func))
            if '.status_code' in source:
                test_for_get_method_response_code = True
            if '.json' in source or 'json.loads' in source:
                test_for_get_method_response_body = True

        if not test_for_get_method_response_code:
            print(FAIL_COLOR + f"[{warning_count}]")
            warning_count += 1
            print(
                FAIL_COLOR +
                "Your test case for GET() does not seem to be testing the response code.\n"
            )

        if not test_for_get_method_response_body:
            print(FAIL_COLOR + f"[{warning_count}]")
            warning_count += 1
            print(
                FAIL_COLOR +
                "Your test case for GET() does not seem to be testing the CONTENTS of the response.\n"
            )

    # POST()
    test_for_post_method_response_code = False
    test_for_post_method_response_body = False
    count_post_method_test_for_inference_result = 0

    if not test_functions_for_post:
        print(FAIL_COLOR + f"[{warning_count}]")
        warning_count += 1
        print(FAIL_COLOR + "No test cases were detected for the POST() method.")
        print(
            FAIL_COLOR +
            "Please make sure you have TWO test cases for the POST() method. "
            "One test case for EACH of the possible inferences (results/outputs) of the ML model.\n"
        )
        sanity_test_passing = False
    else:
        if len(test_functions_for_post) == 1:
            print(FAIL_COLOR + f"[{warning_count}]")
            warning_count += 1
            print(
                FAIL_COLOR +
                "Only one test case was detected for the POST() method. "
                "Please make sure you have two test cases for the POST() method. "
                "One test case for EACH of the possible inferences (results/outputs) of the ML model.\n"
            )
            sanity_test_passing = False

        for func in test_functions_for_post:
            source = inspect.getsource(getattr(module, func))
            if '.status_code' in source:
                test_for_post_method_response_code = True
            if '.json' in source or 'json.loads' in source:
                test_for_post_method_response_body = True
                count_post_method_test_for_inference_result += 1

        if not test_for_post_method_response_code:
            print(FAIL_COLOR + f"[{warning_count}]")
            warning_count += 1
            print(
                FAIL_COLOR +
                "One or more of your test cases for POST() do not seem to be testing the response code.\n"
            )
        if not test_for_post_method_response_body:
            print(FAIL_COLOR + f"[{warning_count}]")
            warning_count += 1
            print(
                FAIL_COLOR +
                "One or more of your test cases for POST() do not seem to be testing the contents of the response.\n"
            )

        if len(test_functions_for_post) >= 2 and count_post_method_test_for_inference_result < 2:
            print(FAIL_COLOR + f"[{warning_count}]")
            warning_count += 1
            print(
                FAIL_COLOR +
                "You do not seem to have TWO separate test cases, one for each possible prediction that your model can make."
            )

    sanity_test_passing = (
        sanity_test_passing and
        test_for_get_method_response_code and
        test_for_get_method_response_body and
        test_for_post_method_response_code and
        test_for_post_method_response_body and
        count_post_method_test_for_inference_result >= 2
    )

    if sanity_test_passing:
        print(OK_COLOR + "Your test cases look good!")

    print(
        WARN_COLOR +
        "This is a heuristic-based sanity testing and cannot guarantee the correctness of your code."
    )
    print(
        WARN_COLOR +
        "You should still check your work against the rubric to ensure you meet the criteria."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'test_dir',
        metavar='test_dir',
        nargs='?',
        default='tests',
        help='Name of the directory that has test files.')
    args = parser.parse_args()
    run_sanity_check(args.test_dir)
