import warnings

import matplotlib
import xgboost as xgb
import pandas as pd
import glob
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Force a backend that works (TkAgg, Agg, or Qt5Agg)
matplotlib.use('TkAgg')

warnings.filterwarnings('ignore')

# Load dataset.csv
df_simulation = pd.read_csv(r"C:\Users\Najihah Azman\PycharmProjects\pythonProject6\data\MITM dataset\combined_mitm_data_simulation_experiments.csv")# Adjust path if needed
df_real = pd.read_csv(r"C:\Users\Najihah Azman\PycharmProjects\pythonProject6\data\MITM dataset\combined_mitm_data_real_experiments.csv")# Adjust path if needed


print(df_real.columns)
print(df_simulation.columns)
df_combined = pd.concat([df_real, df_simulation], ignore_index=True)
print(df_combined.head())
print(df_combined.shape)

# rename the columns
col_names = ['IRTT', 'TTOC', 'MITR', 'MATR', 'NROC', 'ARP poisoning indicator','Label']
df_combined.columns = col_names
print(df_combined.columns)
#
print(df_combined['Label'].value_counts())
# check missing variables
print(df_combined.isnull().sum())
df_cleaned = df_combined.dropna(axis=0)


# Load dataset (assuming it's already loaded as 'df')
X = df_cleaned.drop(['Label'], axis=1)  # Features (exclude 'Label' column)
y = df_cleaned['Label']  # Target variable (Labels)

# Encode target variable to numeric labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  # Convert labels to numeric values (0, 1, 2, 3...)

# Split the dataset into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=0)

# Define parameters for multi-class classification
params = {
    "objective": "multi:softmax",  # Multi-class classification
    "num_class": len(label_encoder.classes_),  # Set number of classes
    "colsample_bytree": 0.3,  # Randomly sample columns for building trees
    "learning_rate": 0.1,  # Step size
    "max_depth": 5,  # Max depth of each tree
    "alpha": 10,  # L1 regularization term
    "eval_metric": "mlogloss"  # Evaluation metric for multi-class classification
}

# Instantiate the XGBClassifier with parameters
xgb_clf = XGBClassifier(**params)

# Fit the classifier to the training data
xgb_clf.fit(X_train, y_train)

# Make predictions on test data
y_pred = xgb_clf.predict(X_test)

# Check accuracy score
print(f'XGBoost model accuracy score: {accuracy_score(y_test, y_pred):0.4f}')

# Plot feature importance graph
xgb.plot_importance(xgb_clf)
plt.rcParams['figure.figsize'] = [6, 4]
plt.show()

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Compute classification report
report = classification_report(y_test, y_pred)

# Print classification report
print("Classification Report:\n", report)

# Plot confusion matrix using Seaborn heatmap
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()


# Print the Confusion Matrix and slice it into four pieces

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print('Confusion matrix\n\n', cm)

