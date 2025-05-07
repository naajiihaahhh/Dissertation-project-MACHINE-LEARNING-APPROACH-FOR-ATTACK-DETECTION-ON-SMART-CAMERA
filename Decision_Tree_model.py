import pickle
import warnings
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')
matplotlib.use('TkAgg')  # Set backend

# Load data
with open(r'C:\Users\Najihah Azman\PycharmProjects\pythonProject6\split_data.pkl', 'rb') as f:
    X_train, X_test, y_train, y_test = pickle.load(f)

# Fit label encoder
label_encoder = LabelEncoder()
label_encoder.fit(y_train)


# ---------------------------------------------------
# Define evaluation function for usage after training
# ---------------------------------------------------
def evaluate_model(y_true, y_pred, label_encoder, title="Model Evaluation"):
    """
    Evaluate the model using confusion matrix, classification report,
    and bar plots for precision, recall, and F1-score.

    """
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)

    # Print raw results
    print(f'\nConfusion Matrix ({title})\n', cm)
    print(f'\nClassification Report ({title})\n', classification_report(y_true, y_pred))

    # Plot Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix ({title})')
    plt.show()

    # Classification Report Metrics
    labels = list(report.keys())[:-3]  # Exclude 'accuracy', etc.
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
    ax.set_title(f'Classification Report Metrics ({title})')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim([0, 1.1])
    ax.legend()

    for bar in ax.containers:
        ax.bar_label(bar, fmt='%.2f', padding=3)

    plt.show()

    # Summary metrics
    summary_metrics = ['Accuracy', 'Macro Avg', 'Weighted Avg']
    values = [
        report['accuracy'],
        report['macro avg']['f1-score'],
        report['weighted avg']['f1-score']
    ]

    plt.figure(figsize=(6, 4))
    plt.bar(summary_metrics, values, color=['blue', 'orange', 'green'])
    plt.xlabel("Metrics")
    plt.ylabel("Scores")
    plt.title(f"Classification Summary ({title})")
    plt.ylim(0.85, 1.0)

    for i, v in enumerate(values):
        plt.text(i, v + 0.005, f"{v:.2f}", ha='center', fontsize=12)
    plt.show()


# --------------------------
# Gini Model training
# --------------------------
clf_gini = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=0)
clf_gini.fit(X_train, y_train)
y_pred_gini = clf_gini.predict(X_test)

print('Model accuracy (Gini): {:.4f}'.format(accuracy_score(y_test, y_pred_gini)))
evaluate_model(y_test, y_pred_gini, label_encoder, title="Gini Criterion")

# --------------------------
# Entropy Model training
# --------------------------
clf_entropy = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
clf_entropy.fit(X_train, y_train)
y_pred_entropy = clf_entropy.predict(X_test)

print('Model accuracy (Entropy): {:.4f}'.format(accuracy_score(y_test, y_pred_entropy)))
evaluate_model(y_test, y_pred_entropy, label_encoder, title="Entropy Criterion")

def perform_grid_search(X_train, y_train):
    """
    Perform GridSearchCV to find the best hyperparameters for a Decision Tree classifier.

    """
    from sklearn.model_selection import GridSearchCV

    # Define the hyperparameter grid
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [3, 5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Base Decision Tree model
    dt = DecisionTreeClassifier(random_state=0)

    # Initialize Grid Search with 5-fold cross-validation
    grid_search = GridSearchCV(
        estimator=dt,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        verbose=1,
        scoring='accuracy'
    )

    # Fit Grid Search to training data
    grid_search.fit(X_train, y_train)

    print("Best hyperparameters:", grid_search.best_params_)

    return grid_search.best_estimator_


# --------------------------
# Tuned Model (based on GridSearch)
# --------------------------
best_model = DecisionTreeClassifier(
    criterion='gini',
    max_depth=5,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=0
)
best_model.fit(X_train, y_train)
y_pred_best = best_model.predict(X_test)

print("Accuracy of best tuned model: {:.4f}".format(accuracy_score(y_test, y_pred_best)))
evaluate_model(y_test, y_pred_best, label_encoder, title="Tuned Gini Model")
