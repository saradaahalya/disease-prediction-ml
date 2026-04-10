from sklearn.decomposition import PCA
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

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

# STEP 8: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    data, y, test_size=0.2, random_state=42
)

# 🔥 STEP 9: APPLY PCA (KEY STEP)
pca = PCA(n_components=10)

X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

print("Explained variance:", pca.explained_variance_ratio_.sum())

# ================= RANDOM FOREST =================
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy (PCA): {accuracy_rf * 100:.2f}%")

# ================= SVM =================
svm_model = SVC()
svm_model.fit(X_train, y_train)

y_pred_svm = svm_model.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(f"SVM Accuracy (PCA): {accuracy_svm * 100:.2f}%")

# ================= KNN =================
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

y_pred_knn = knn_model.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f"KNN Accuracy (PCA): {accuracy_knn * 100:.2f}%")

# STEP: Print labels
original_labels = labels["cancer"].unique()
print(f"Unique values in the label column: {original_labels}")