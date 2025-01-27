from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import pickle

# Load data
data = load_iris()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define hyperparameter grid
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 5, 10],
    'min_samples_split': [2, 5, 10]
}

# Initialize and fit GridSearchCV
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Extract results
results = pd.DataFrame(grid_search.cv_results_)

# Best parameters and model
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

# Test set evaluation
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Save the best model
with open('model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

# Generate report
report = f"""
Hyperparameter Tuning Report
============================

1. Hyperparameter Grid
----------------------
{param_grid}

2. Best Parameters
------------------
{best_params}

3. CV Results (Top 5 Configurations)
------------------------------------
{results[['mean_test_score', 'std_test_score', 'params']].sort_values(by='mean_test_score', ascending=False).head()}

4. Test Set Evaluation
----------------------
Accuracy: {accuracy:.4f}

Classification Report:
{classification_rep}
"""

# Save the report to a text file
with open("hyperparameter_tuning_report.txt", "w") as file:
    file.write(report)

# Visualize results
plt.figure(figsize=(10, 6))
for param in param_grid['n_estimators']:
    subset = results[results['param_n_estimators'] == param]
    plt.plot(subset['param_max_depth'], subset['mean_test_score'], label=f'n_estimators={param}')

plt.title("Mean Test Scores Across Max Depths")
plt.xlabel("Max Depth")
plt.ylabel("Mean Test Score")
plt.legend()
plt.grid()
plt.savefig("hyperparameter_tuning_plot.png")
plt.show()

print("Report generated and saved as 'hyperparameter_tuning_report.txt'")
print("Plot saved as 'hyperparameter_tuning_plot.png'")

