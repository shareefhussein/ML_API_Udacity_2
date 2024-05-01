## Model Card Overview

**Reference**: For additional background information, refer to the [Model Card research paper](https://arxiv.org/pdf/1810.03993.pdf).

### Model Details
- **Creator**: Shareef Mansour
- **Date**: April 24th 2024 
- **Type**: Gradient Boosting Classifier.

### Intended Use
- **Application**: This model is designed to predict an individual's salary based on various financial attributes.

### Data Sources
- **Training Data**: The model is trained with 80% of the data sourced from the [UCI Census Income Dataset](https://archive.ics.uci.edu/ml/datasets/census+income).
- **Evaluation Data**: Evaluation utilizes the remaining 20% of the data from the same [UCI Census Income Dataset](https://archive.ics.uci.edu/ml/datasets/census+income).

### Performance Metrics
- **Accuracy**: The model is evaluated based on the accuracy with value: 0.85

### Ethical Considerations
- **Inherent Bias**: The inclusion of sensitive attributes such as race, gender, and country of origin in the dataset can lead to inherent biases in the model's predictions. These biases could perpetuate or even exacerbate existing societal inequalities, particularly in employment-related decision-making.

- **Impact on Individuals**: The model’s predictions could significantly impact individuals' livelihoods. Misclassifications or biased predictions could affect job prospects, promotions, and overall economic stability.

