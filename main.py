import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# STEP 1: Load training dataset only
data = pd.read_csv("data/data_set_ALL_AML_train.csv")

# STEP 2: Transpose
data = data.T

# STEP 3: Fix headers
data.columns = data.iloc[0]
data = data[1:]

# STEP 4: Reset index
data = data.reset_index(drop=True)

# STEP 5: Convert to numeric
data = data.apply(pd.to_numeric, errors='coerce')

# STEP 6: Fill missing values
data = data.fillna(data.mean())

# STEP 7: Load labels
labels = pd.read_csv("data/actual.csv")
labels = labels.drop("patient", axis=1)

y = labels["cancer"].map({'ALL': 0, 'AML': 1})

# Ensure data and labels have the same number of samples
data = data.iloc[:len(y)]

# STEP 8: Train-test split (THIS FIXES EVERYTHING)
X_train, X_test, y_train, y_test = train_test_split(
    data, y, test_size=0.2, random_state=42
)

# STEP 9: Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# STEP 10: Predict
y_pred = model.predict(X_test)

# STEP 11: Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# STEP 12: Print unique values of the label column
original_labels = labels["cancer"].unique()
print(f"Unique values in the label column: {original_labels}")


from sklearn.svm import SVC

# STEP 12: Train SVM model
svm_model = SVC()
svm_model.fit(X_train, y_train)

# STEP 13: Predict using SVM
y_pred_svm = svm_model.predict(X_test)

# STEP 14: Evaluate SVM
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(f"SVM Accuracy: {accuracy_svm * 100:.2f}%")