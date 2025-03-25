import pickle
import warnings
import matplotlib
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

matplotlib.use('TkAgg')  # Set the backend to TkAgg
# Load the split data from pickle file
with open(r'C:\Users\Najihah Azman\PycharmProjects\pythonProject6\split_data.pkl', 'rb') as f:
    X_train, X_test, y_train, y_test = pickle.load(f)


# import DecisionTreeClassifier

from sklearn.tree import DecisionTreeClassifier
# instantiate the DecisionTreeClassifier model with criterion gini index

clf_gini = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=0)


# fit the model
clf_gini.fit(X_train, y_train)

# predict using criterion gini index
y_pred_gini = clf_gini.predict(X_test)

from sklearn.metrics import accuracy_score

print('Model accuracy score with criterion gini index: {0:0.4f}'. format(accuracy_score(y_test, y_pred_gini)))

y_pred_train_gini = clf_gini.predict(X_train)

y_pred_train_gini

print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train_gini)))

# print the scores on training and test set

print('Training set score: {:.4f}'.format(clf_gini.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(clf_gini.score(X_test, y_test)))

# instantiate the DecisionTreeClassifier model with criterion entropy

clf_en = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)

# fit the model
clf_en.fit(X_train, y_train)
# predict using criterion entropy
y_pred_en = clf_en.predict(X_test)

#accuracy score using criterion entropy
from sklearn.metrics import accuracy_score

print('Model accuracy score with criterion entropy: {0:0.4f}'. format(accuracy_score(y_test, y_pred_en)))

y_pred_train_en = clf_en.predict(X_train)

y_pred_train_en

print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train_en)))
# print the scores on training and test set

print('Training set score: {:.4f}'.format(clf_en.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(clf_en.score(X_test, y_test)))

# Print the Confusion Matrix and slice it into four pieces

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred_en)

print('Confusion matrix\n\n', cm)


from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred_en))

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Encode target variable (only needed if not done before)
label_encoder = LabelEncoder()
label_encoder.fit(y_train)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

import numpy as np
import matplotlib.pyplot as plt


# Updated Classification report data
labels = ['Class 0', 'Class 1', 'Class 2']
precision = [1.00, 0.97, 0.99]
recall = [0.84, 1.00, 1.00]
f1_score = [0.91, 0.99, 0.99]

x = np.arange(len(labels))  # Label positions
width = 0.2  # Bar width

fig, ax = plt.subplots(figsize=(8, 5))
rects1 = ax.bar(x - width, precision, width, label='Precision', color='royalblue')
rects2 = ax.bar(x, recall, width, label='Recall', color='orange')
rects3 = ax.bar(x + width, f1_score, width, label='F1-score', color='green')

# Labels, title, and legend
ax.set_xlabel('Classes')
ax.set_ylabel('Scores')
ax.set_title('Classification Report Metrics')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim([0, 1.1])  # Set y-axis limit
ax.legend()

# Show values on top of bars
for rect in rects1 + rects2 + rects3:
    height = rect.get_height()
    ax.annotate(f'{height:.2f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom')

plt.show()

# --------------------------
# Accuracy, Macro Avg, Weighted Avg Visualization
# --------------------------

# Updated metrics and values
metrics = ['Accuracy', 'Macro Avg', 'Weighted Avg']
values = [0.97, 0.94, 0.96]  # Accuracy, macro avg F1-score, weighted avg F1-score

plt.figure(figsize=(6, 4))
plt.bar(metrics, values, color=['blue', 'orange', 'green'])

# Labels and title
plt.xlabel("Metrics")
plt.ylabel("Scores")
plt.title("Classification Report Summary")
plt.ylim(0.85, 1.0)  # Adjust y-axis range

# Display values on top of bars
for i, v in enumerate(values):
    plt.text(i, v + 0.005, f"{v:.2f}", ha='center', fontsize=12)

plt.show()
