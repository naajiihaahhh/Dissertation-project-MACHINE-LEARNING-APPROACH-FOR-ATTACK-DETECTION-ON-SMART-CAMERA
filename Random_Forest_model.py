import pickle
import warnings
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV


warnings.filterwarnings('ignore')
matplotlib.use('TkAgg')  # Set backend

# --------------------------
# Load data
# --------------------------
with open(r'split_data.pkl', 'rb') as f:
    X_train, X_test, y_train, y_test = pickle.load(f)

# Fit label encoder
label_encoder = LabelEncoder()
label_encoder.fit(y_train)


# --------------------------
# Evaluation Function
# --------------------------
def evaluate_model(y_true, y_pred, label_encoder, title="Model Evaluation"):
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)

    # Print reports
    print(f'\nConfusion Matrix ({title})\n', cm)
    print(f'\nClassification Report ({title})\n', classification_report(y_true, y_pred))

    # Confusion Matrix Plot
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix ({title})')
    plt.show()

    # Bar plot of metrics
    labels = list(report.keys())[:-3]
    precision = [report[label]['precision'] for label in labels]
    recall = [report[label]['recall'] for label in labels]
    f1 = [report[label]['f1-score'] for label in labels]

    x = np.arange(len(labels))
    width = 0.2
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width, precision, width, label='Precision', color='royalblue')
    ax.bar(x, recall, width, label='Recall', color='orange')
    ax.bar(x + width, f1, width, label='F1-score', color='green')
    ax.set_xlabel('Classes')
    ax.set_ylabel('Scores')
    ax.set_title(f'Classification Metrics ({title})')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim([0.8, 1.02])
    ax.legend()
    for bar in ax.containers:
        ax.bar_label(bar, fmt='%.2f', padding=3)
    plt.show()

    # Summary metrics
    summary_labels = ['Accuracy', 'Macro Avg', 'Weighted Avg']
    summary_values = [
        report['accuracy'],
        report['macro avg']['f1-score'],
        report['weighted avg']['f1-score']
    ]
    plt.figure(figsize=(6, 4))
    plt.bar(summary_labels, summary_values, color=['blue', 'orange', 'green'])
    plt.ylim(0.8, 1.0)
    plt.xlabel("Metrics")
    plt.ylabel("Scores")
    plt.title(f"Summary Metrics ({title})")
    for i, v in enumerate(summary_values):
        plt.text(i, v + 0.005, f"{v:.2f}", ha='center', fontsize=12)
    plt.show()


# --------------------------
# Training Random Forest Model
# --------------------------
rfc_default = RandomForestClassifier(random_state=0)
rfc_default.fit(X_train, y_train)
y_pred_default = rfc_default.predict(X_test)
print("Accuracy (Default RF): {:.4f}".format(accuracy_score(y_test, y_pred_default)))
evaluate_model(y_test, y_pred_default, label_encoder, title="Random Forest - Default")

def perform_rf_grid_search(X_train, y_train):
    """
    Perform GridSearchCV to find the best hyperparameters for Random Forest classifier.
    """
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['gini', 'entropy'],
        'max_features': ['sqrt', 'log2']
    }

    rf = RandomForestClassifier(random_state=0)

    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=3,
        n_jobs=-1,
        verbose=1,
        scoring='accuracy'
    )

    grid_search.fit(X_train, y_train)
    print("Best Hyperparameters (Random Forest):", grid_search.best_params_)
    return grid_search.best_estimator_

# --------------------------
# Tuned RF Model
# --------------------------
tuned_rf = RandomForestClassifier(
    criterion='gini',
    max_depth=10,
    max_features='sqrt',
    min_samples_leaf=1,
    min_samples_split=2,
    n_estimators=100,
    random_state=0
)
tuned_rf.fit(X_train, y_train)
y_pred_tuned = tuned_rf.predict(X_test)
print("Accuracy (Tuned RF): {:.4f}".format(accuracy_score(y_test, y_pred_tuned)))
evaluate_model(y_test, y_pred_tuned, label_encoder, title="Random Forest - Tuned Model")

# Save tuned Random Forest model
with open('Random_Forest_tuned.pkl', 'wb') as file:
    pickle.dump(tuned_rf, file)

