import os
import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import joblib

# Folder where CSV files are saved
DATA_DIR = "sign_data"
X = []
y = []

# Load data
for file in os.listdir(DATA_DIR):
    if file.endswith(".csv"):
        label = file.split("_")[1].replace(".csv", "")
        with open(os.path.join(DATA_DIR, file), 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) == 42:  # 21 keypoints x 2 (x, y)
                    X.append([float(val) for val in row])
                    y.append(label)

# Convert to NumPy arrays
X = np.array(X)
y = np.array(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train KNN model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Evaluate accuracy
accuracy = model.score(X_test, y_test)
print(f"✅ Model trained with accuracy: {accuracy * 100:.2f}%")

# Save the model
joblib.dump(model, "sign_model_knn.pkl")
print("💾 Model saved as sign_model_knn.pkl")
