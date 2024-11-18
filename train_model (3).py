import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Load the dataset
data_path = 'survey lung cancer.csv'  # Update with your actual dataset path
data = pd.read_csv(data_path)

# Check if the expected columns are in the dataset
expected_columns = ['GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE', 'CHRONIC DISEASE', 'LUNG_CANCER']
missing_columns = [col for col in expected_columns if col not in data.columns]

if missing_columns:
    print(f"Warning: The following columns are missing: {missing_columns}")
    # Optionally exit the script or handle the situation
    exit(1)

# Define features and target variable
X = data[['GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE', 'CHRONIC DISEASE']]
y = data['LUNG_CANCER']

# Encode categorical variables if needed (example using pd.get_dummies)
X = pd.get_dummies(X, drop_first=True)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save the model if needed
joblib.dump(model, 'lung_cancer_model.pkl')
