import warnings

import matplotlib
import numpy as np
import pandas as pd
import sns as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt

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
# Drop rows with missing values
df_cleaned = df_combined.dropna(axis=0)
print(df_cleaned.isnull().sum())


# Load dataset (assuming it's already loaded as 'df')
X = df_cleaned.drop(['Label'], axis=1)  # Features (exclude 'Label' column)
y = df_cleaned['Label']  # Target variable (Labels)

# Encode target variable to numeric labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  # Convert labels to numeric values (0, 1, 2, 3...)

# Split the dataset into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=0)


# instantiate the classifier

rfc = RandomForestClassifier(random_state=0)



# fit the model

rfc.fit(X_train, y_train)



# Predict the Test set results

y_pred = rfc.predict(X_test)



# Check accuracy score

from sklearn.metrics import accuracy_score

print('Model accuracy score with 10 decision-trees : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))


# instantiate the classifier with n_estimators = 100

rfc_100 = RandomForestClassifier(n_estimators=100, random_state=0)



# fit the model to the training set

rfc_100.fit(X_train, y_train)



# Predict on the test set results

y_pred_100 = rfc_100.predict(X_test)



# Check accuracy score

print('Model accuracy score with 100 decision-trees : {0:0.4f}'. format(accuracy_score(y_test, y_pred_100)))


from sklearn.ensemble import RandomForestClassifier
# create the classifier with n_estimators = 100

clf = RandomForestClassifier(n_estimators=100, random_state=0)

# fit the model to the training set
clf.fit(X_train, y_train)
print(clf.get_params())

# view the feature scores

feature_scores = pd.Series(clf.feature_importances_, index=X_train.columns).sort_values(ascending=False)

# print(feature_scores)



# Creating a seaborn bar plot
sns.barplot(x=feature_scores, y=feature_scores.index)

# Add labels to the graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')

# Add title to the graph
plt.title("Visualizing Important Features")

# Visualize the graph
plt.show()


# Print the Confusion Matrix and slice it into four pieces
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print('Confusion matrix\n\n', cm)

# classification report
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))


plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Print the Confusion Matrix and slice it into four pieces

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print('Confusion matrix\n\n', cm)



