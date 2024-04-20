import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Load data
solar_data = pd.read_csv('Copy of sonar data.csv', header=None)

# Prepare data
X = solar_data.drop(columns=60, axis=1)
y = solar_data[60]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=1)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train and evaluate model
model = LogisticRegression(max_iter=1000)  # Increase max_iter for convergence if needed

# Hyperparameter tuning
param_grid = {
    'C': np.logspace(-4, 4, 20),
    'solver': ['liblinear', 'lbfgs']  # Suitable solvers for small datasets
}
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)

# Best model
best_model = grid_search.best_estimator_

# Evaluations
training_prediction = best_model.predict(X_train_scaled)
print("Training Accuracy:", accuracy_score(training_prediction, y_train))
test_prediction = best_model.predict(X_test_scaled)
print("Test Accuracy:", accuracy_score(test_prediction, y_test))

# //////////////////////////////////////////////////////////////////////////

# Create Streamlit App:
st.title("Sonar Rock VS Mine Prediction")
# Text input for user to enter data
input_data = st.text_input('Enter comma-separated values here')
# Predict and show result on button click
if st.button('Predict'):
    # Prepare input data
    input_data_np_array = np.asarray(input_data.split(','), dtype=float)
    reshaped_input = input_data_np_array.reshape(1, -1)
    # Predict and show result
    prediction = model.predict(reshaped_input)
    if prediction[0] == 'R':
        st.write('This Object is Rock')
    else:
        st.write('The Object is Mine')
