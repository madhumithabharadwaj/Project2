# Project 2

# Team Members

- MADHUMITHA BHARADWAJ GUDMELLA, A20544762
- BHAVANA POLAKALA, A20539792
- SHREYA JATLING, A20551616
- DEVESH PATEL, A20548274

# Overview

This project focuses on implementing a Gradient Boosting Tree Classifier from first principles, inspired by Sections 10.9–10.10 of The Elements of Statistical Learning (2nd Edition). The aim is to build a binary classification model that incrementally fits simple decision stumps to the gradient of the log-loss function. The model is implemented without the use of external machine learning libraries like scikit-learn or XGBoost.

Bonus features include early stopping, training accuracy tracking, and visualizations such as predicted labels and accuracy-vs-estimators curve to enhance model interpretability and control over training dynamics.

# Features:

- Gradient Boosting with decision stumps

- Pure NumPy implementation (no sklearn or external ML libraries)

- Binary classification with log-loss optimization

- Early stopping mechanism for regularization

- In-memory synthetic data generation and CSV export

- Data visualization and performance tracking

# Project Structure

- `run_demo.py` – Main script to generate data, train model, and visualize results
- `model/BoostingTree.py` – Core boosting model (fit/predict interface)
- `tests/test_boosting_tree.py` – Unit test verifying model correctness
- `generated_data.csv` – CSV file of the generated dataset
- `requirements.txt` – Lists the dependencies required for the project.

# Implementation:

1. Data Processing

The dataset is synthetically generated using NumPy and consists of randomly drawn features with binary labels. The generator applies logistic transformation on linear combinations of features, simulating a real-world binary classification task.

run_demo.py: generates the dataset in memory and saves it to generated_data.csv

Pandas is used for saving the dataset

matplotlib is used for visualization

2. Model Implementation

model/BoostingTree.py: Implements Gradient Boosting logic from scratch using:

Decision stumps (depth-1 trees)

Log-loss gradient descent

Sigmoid function for classification

A new weak learner is added in each round to fit the negative gradient

Early stopping is supported based on accuracy stagnation

3. Testing and Validation

tests/test_boosting_tree.py: Automatically verifies that the model reaches at least 80% accuracy using Pytest

Training accuracy and predictions are printed in run_demo.py

plots include:

Accuracy vs. number of estimators

Predicted vs. true labels scatter plot

# Installation

1. Clone the repository.

2. Navigate to the project directory:
  
   cd Project2-main
 
3. Create and activate a virtual environment (optional but recommended):
  
   python -m venv venv
   source venv/bin/activate
   
4. Install dependencies:

   pip install -r requirements.txt
   

# Usage

- Model Training & Evaluation:
Run run_demo.py to generate synthetic classification data, train the Gradient Boosting model, evaluate accuracy, and visualize predictions.

- Hyperparameter Tuning:
Modify parameters such as n_estimators, learning_rate, and early_stopping_rounds in run_demo.py to observe their effects on accuracy and convergence behavior.

- Visualization & Diagnostics:
The script automatically produces:

 -> A scatter plot of predicted class labels

 -> An accuracy vs. number of estimators curve These visualizations help analyze learning dynamics and decision boundaries.

- Unit Testing:
Run tests/test_boosting_tree.py using pytest to validate that the model achieves acceptable performance (≥80% accuracy) on controlled synthetic data.

- To run:
  Run the full demo:

  python run_demo.py

- This will:

  Generate and save data to generated_data.csv

  Train a boosting model

  Print accuracy and predictions

  Show plots for visual validation

  Trigger early stopping if enabled

- Run tests:

  python -m pytest -v tests/test_boosting_tree.py

# Dependencies

numpy

pandas

matplotlib

pytest

The required dependencies are listed in `requirements.txt`. Install them using:

pip install -r requirements.txt

# Acknowledgments

- Libraries such as NumPy, Pytest and Pandas are utilized.

# References

- ESL II: Hastie, Tibshirani, Friedman. Sections 10.9–10.10 (Gradient Boosting Machines)

- https://explained.ai/gradient-boosting/index.html (for understanding visualizations)

# Questions 

1) What does the model you have implemented do and when should it be used?

- This project implements a Gradient Boosting Tree Classifier from first principles, based on Sections 10.9–10.10 of The Elements of Statistical Learning. The model builds an ensemble of simple decision stumps (trees with depth = 1), where each new stump corrects the errors made by the previous ones. It does this by fitting to the negative gradient of the loss function, which in this case is the log-loss used for binary classification. Each round fits a new weak learner to correct the residuals (gradient) of the current ensemble’s prediction using log-loss optimization.

- This model should be used when:

We want a high-performing model for binary classification tasks

We want full control over boosting logic (no scikit-learn)

We are exploring ensemble methods for classification from first principles

We want to analyze interpretability and overfitting control mechanisms like early stopping


2) How did you test your model to determine if it is working reasonably correctly?

- We generated a synthetic binary classification dataset (generated_data.csv) with two separable clusters.

We tested the model using a small script (run_demo.py) that loads the data, trains the classifier, and evaluates accuracy.

Additionally, We created a unit test file (test_boosting_tree.py) to verify that the model.

- The model was tested through:

A unit test (test_boosting_tree.py) which verifies accuracy ≥ 80% using Pytest

Visual validation: scatter plots of true labels vs. predicted labels

Performance curve: accuracy plotted against number of boosting rounds

Saved training accuracy and predictions in run_demo.py

Synthetic dataset generation ensures controlled testing scenarios

Early stopping logs are printed if the model detects no improvement over several rounds.


3) What parameters have you exposed to users of your implementation in order to tune performance? (Also perhaps provide some basic usage examples.)

- The following parameters can be passed when creating a BoostingTree classifier:

n_estimators: number of boosting rounds (trees)

learning_rate: learning shrinkage to avoid overfitting

max_depth: maximum depth of each tree (currently 1)

early_stopping_rounds: if specified, stops training after N rounds with no accuracy improvement

- Usage example:

  from model.BoostingTree import BoostingTree
  model = BoostingTree(n_estimators=100, learning_rate=0.1, early_stopping_rounds=5)

  model.fit(X, y)
  predictions = model.predict(X)


4) Are there specific inputs that your implementation has trouble with? Given more time, could you work around these or is it fundamental?

- Limitations:

  -> It currently supports only binary classification

  -> Deep trees are not implemented (only stumps)

  -> No support for missing values or categorical variables

  -> Very noisy datasets may reduce accuracy

  -> No early-stopping based on validation loss (only training accuracy)

- Possible Improvements:

  -> Add deeper trees and tree pruning

  -> Extend to multiclass classification using one-vs-all

  -> Add support for validation sets and loss-based early stopping

  -> Implement shrinkage, feature subsampling, and momentum boosting
