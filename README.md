# Sonar Rock Vs. Mine Prediction

### Overview
This project aims to predict whether an object detected by sonar is a rock or a mine using machine learning techniques. The project includes a Graphical User Interface (GUI) for user interaction.
### Features
Graphical User Interface (GUI): The project features a user-friendly GUI that allows users to input sonar data and obtain predictions.

Logistic Regression Model: The machine learning model employed is logistic regression, a classification algorithm suitable for binary classification tasks like this.

Feature Scaling: Features are scaled using the StandardScaler to ensure that all features contribute equally to the model's learning process. This prevents features with larger scales from dominating the model.

Hyperparameter Tuning: Grid search is employed to tune the hyperparameters of the logistic regression model, specifically the regularization parameter C and the solver algorithm. This helps optimize the model's performance by finding the best combination of hyperparameters.
### Requirements
Python 3.x

Required Python libraries: numpy, pandas, scikit-learn, tkinter
### Usage
Clone or download the repository to your local machine.

Install the required Python libraries if they are not already installed:
