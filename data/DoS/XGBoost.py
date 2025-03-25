import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Force a backend that works (TkAgg, Agg, or Qt5Agg)
matplotlib.use('TkAgg')

import warnings

from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')
# import dataset
data = r'C:\Users\Najihah Azman\PycharmProjects\pythonProject6\data\DoS\Flooding_Based_DDoS_Multi-class_Dataset.csv'

df = pd.read_csv(data)
print(df.shape)
print(df.head())
print(df.describe())
print(df.columns)

X = df.drop('Label', axis=1)

y = df['Label']

print(X.head())
print(y.head())


# import XGBoost
import xgboost as xgb

from sklearn.preprocessing import LabelEncoder

# Encode target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  # Convert labels to numeric value

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=0)

# Define data_dmatrix (optional, used for boosting)
data_dmatrix = xgb.DMatrix(data=X, label=y_encoded)
# import XGBClassifier
from xgboost import XGBClassifier

# declare parameters
params = {
    'objective': 'binary:logistic',
    'max_depth': 4,
    'alpha': 10,
    'learning_rate': 1.0,
    'n_estimators': 100
}

# instantiate the classifier
xgb_clf = XGBClassifier(**params)

# fit the classifier to the training data
print(xgb_clf.fit(X_train, y_train))

# make predictions on test data
y_pred = xgb_clf.predict(X_test)

# check accuracy score
from sklearn.metrics import accuracy_score

print('XGBoost model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

from xgboost import cv

# Get number of unique classes
num_classes = len(set(y))

# Update parameters
params = {"objective":"multi:softmax", "num_class": num_classes, 'colsample_bytree': 0.3,
                'learning_rate': 0.1, 'max_depth': 5, 'alpha': 10}

# Cross-validation
xgb_cv = xgb.cv(dtrain=data_dmatrix, params=params, nfold=3,
                num_boost_round=50, early_stopping_rounds=10, metrics="mlogloss", as_pandas=True, seed=123)

print(xgb_cv.head())

# plot graph of the importance
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


