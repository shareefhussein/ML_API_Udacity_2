### Overview of Third project: Implementing a Machine Learning Model Using FastAPI on a Cloud Platform

For my latest educational venture, I have uploaded a machine learning deployment project on GitHub. You can explore the details of this project through my GitHub repository:(https://github.com/shareefhussein/ML_API_Udacity/).

### Initial Setup Using Provided Code

Udacity has supplied the initial framework for this project, available at [Udacity Starter Code](https://github.com/udacity/nd0821-c3-starter-code). Following the course guidelines, I have set up a new project directory and initialized it with Git, using the provided starter code as a foundation.

### Acknowledgements

All intellectual rights for the foundational code belong to the course authors.

### Project Licensing

For licensing information, please refer to the [Udacity License](https://github.com/udacity/nd0821-c3-starter-code/blob/master/LICENSE.txt).

### Setup Instructions

```bash
# Create a new conda environment
conda create -n project "python=3.8" --file ML_API_Udaicty/requirements.txt -c conda-forge

# Activate the environment
conda activate project3

Usage Guidelines
Testing: Run the tests with the following command

python -m pytest ML_API_Udaicty -vv --log-cli-level=DEBUG

Sanity Check: Verify the setup by running:

python -m ML_API_Udaicty.src.sanitycheck


Model Training and Metrics Calculation:

# Train the model
python -m ML_API_Udaicty.src.train_test_model

# Calculate metrics
python -m ML_API_Udaicty.src.model.slice_output.txt

Local Server: Start the application locally using:

uvicorn ML_API_Udaicty.src.main:app --host 127.0.0.1 --port 8000
Check the documentation at: http://127.0.0.1:8000/docs

Model Details
See the model card for more details here: Model Card

Github Actions
Github Actions are triggered on changes within this project, with an exception: if a tag is pushed, Github Actions are also called.

