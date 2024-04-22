# 1. Simple Linear Regression

import numpy as np
import matplotlib.pyplot as plt

def estimate_coef(x, y):
  # number of observations/points
  n = np.size(x)

  # mean of x and y vector
  m_x = np.mean(x)
  m_y = np.mean(y)

  # calculating cross-deviation and deviation about x
  SS_xy = np.sum(y*x) - n*m_y*m_x
  SS_xx = np.sum(x*x) - n*m_x*m_x

  # calculating regression coefficients
  b_1 = SS_xy / SS_xx
  b_0 = m_y - b_1*m_x

  return (b_0, b_1)

def plot_regression_line(x, y, b):
  # plotting the actual points as scatter plot
  plt.scatter(x, y, color = "m",
        marker = "o", s = 30)

  # predicted response vector
  y_pred = b[0] + b[1]*x

  # plotting the regression line
  plt.plot(x, y_pred, color = "g")

  # putting labels
  plt.xlabel('x')
  plt.ylabel('y')


def main():
    # observations / data
    x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    y = np.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12])

    # estimating coefficients
    b = estimate_coef(x, y)

    # plotting the regression line
    plot_regression_line(x, y, b)

    # displaying the plot
    plt.show()

    print("Estimated coefficients:\nb_0 = {} \
       \nb_1 = {}".format(b[0], b[1]))

if __name__ == "__main__":
    main()


#2. multiple linear regression : predict student exam scores using multiple factors like study hours previous scores and attendance
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(42)

# Assuming you have a dataset with columns: 'study_hours', 'previous_scores', 'attendance', and 'exam_scores'
# Replace this with your actual dataset
data = {
    'study_hours': np.random.randint(1, 10, 100),
    'previous_scores': np.random.randint(40, 100, 100),
    'attendance': np.random.randint(0, 2, 100),  # Assuming binary attendance (0 or 1)
}

data['exam_scores'] = 0.5 * data['study_hours'] + 0.3 * data['previous_scores'] + 10 * data['attendance'] + np.random.normal(0, 5, 100)

df = pd.DataFrame(data)

# Split the data into features (X) and target variable (y)
X = df[['study_hours', 'previous_scores', 'attendance']]
y = df['exam_scores']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = metrics.r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse:.2f}')
print(f'Root Mean Squared Error: {rmse:.2f}')
print(f'R-squared: {r2:.2f}')

# Plotting the actual vs predicted values with different colors
plt.scatter(y_test, y_pred, label='Actual', color='blue', marker='o')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red', label='Perfect Prediction')
plt.xlabel('Actual Exam Scores')
plt.ylabel('Predicted Exam Scores')
plt.title('Actual vs Predicted Exam Scores')
plt.legend()
plt.show()


# 3 Liner Regression: Predict house prices 
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt

# Load the California Housing dataset
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['PRICE'] = housing.target

# Select features and target variable
X = df[['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']]
y = df['PRICE']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = metrics.r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse:.2f}')
print(f'Root Mean Squared Error: {rmse:.2f}')
print(f'R-squared: {r2:.2f}')

# Plotting the actual vs predicted house prices
plt.plot(y_test, label='Actual Prices', marker='o', linestyle='', alpha=0.7)
plt.plot(y_pred, label='Predicted Prices', marker='x', linestyle='', alpha=0.7)
plt.xlabel('Samples')
plt.ylabel('House Prices')
plt.title('Actual vs Predicted House Prices')
plt.legend()
plt.show()


#  5. Build a simple decision tree with a single attribute and two branches
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Select a single attribute (e.g., sepal length)
selected_attribute_index = 0
X_selected = X[:, selected_attribute_index].reshape(-1, 1)

# Build a decision tree model
decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X_selected, y)

# Plot the decision tree
plt.figure(figsize=(10, 6))
plot_tree(decision_tree, filled=True, feature_names=[iris.feature_names[selected_attribute_index]])
plt.title("Decision Tree - Single Attribute")
plt.show()



#8. loan approval decison tree: Create a decision tree to predict wether a loan application will be approved or not based on factors like income, credit score and emplyment status
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

data = {
    'income': [50000, 80000, 120000, 70000, 100000, 60000, 90000, 110000, 95000, 85000, 78000, 60000, 123000, 95000, 87000, 105000, 98000, 80000, 70000, 60000] * 5,
    'credit_score': [650, 720, 800, 690, 750, 620, 700, 780, 700, 680, 750, 620, 800, 720, 690, 780, 760, 720, 710, 650] * 5,
    'employment_status': ['Employed', 'Unemployed', 'Employed', 'Employed', 'Self-employed', 'Unemployed', 'Employed', 'Self-employed', 'Employed', 'Self-employed', 'Unemployed', 'Employed', 'Self-employed', 'Unemployed', 'Employed', 'Self-employed', 'Unemployed', 'Employed', 'Unemployed', 'Self-employed'] * 5,
    'loan_approval': [1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0] * 5  # 1 for approved, 0 for not approved
}

df = pd.DataFrame(data)

# Convert categorical features to numerical using one-hot encoding
df_encoded = pd.get_dummies(df, columns=['employment_status'], drop_first=True)

# Split the dataset into features (X) and target variable (y)
X = df_encoded.drop('loan_approval', axis=1)
y = df_encoded['loan_approval']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a decision tree model
decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X_train, y_train)

# Make predictions on the test set
y_pred = decision_tree.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", classification_rep)

# Plot the decision tree
plt.figure(figsize=(15, 8))
plot_tree(decision_tree, filled=True, feature_names=X.columns, class_names=df['loan_approval'].astype(str))
plt.title("Decision Tree Classifier - Loan Approval Prediction")
plt.show()
