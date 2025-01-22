# train.py
"""
This script trains a RandomForestClassifier model on the Titanic dataset.
The dataset is processed, missing values are handled, and categorical variables
are mapped before training the model. The trained model is saved as a .pkl file.

Modules used:
- pandas
- scikit-learn
- joblib
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load the Titanic dataset
data = pd.read_csv('data/Titanic-Dataset.csv')

# Preprocess the dataset
# Keep only the useful features
data = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Survived']]

# Map categorical variables
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
data['Embarked'] = data['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

# Handle missing values
data['Age'] = data['Age'].fillna(data['Age'].mean())
data['Fare'] = data['Fare'].fillna(data['Fare'].mean())
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])  # Use mode for categorical values

# Validate that no missing values remain
if data.isnull().any().any():
    raise ValueError("Data contains NaN values after preprocessing!")

# Features and target
X = data.drop('Survived', axis=1)
y = data['Survived']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy}")

# Save the model
joblib.dump(model, 'models/titanic_model.pkl')
