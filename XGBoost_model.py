import pickle
import warnings
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier

# Set backend and suppress warnings
warnings.filterwarnings('ignore')
matplotlib.use('TkAgg')

# Load data
with open('split_data.pkl', 'rb') as f:
    X_train, X_test, y_train, y_test = pickle.load(f)

# Fit label encoder
label_encoder = LabelEncoder()
label_encoder.fit(y_train)
num_classes = len(label_encoder.classes_)

# -------------------------------------
# Evaluation Function
# -------------------------------------
def evaluate_model(y_true, y_pred, label_encoder, title="Model Evaluation"):
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)

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

    # Bar plots for Precision, Recall, F1
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
    ax.set_title(f'Classification Report Metrics ({title})')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim([0, 1.1])
    ax.legend()

    for bar in ax.containers:
        ax.bar_label(bar, fmt='%.2f', padding=3)

    plt.show()

    # Summary metrics bar
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

# -------------------------------------
# Initial XGBoost Training (Before Tuning)
# -------------------------------------
def train_initial_xgboost(X_train, y_train, num_classes):
    params = {
        'objective': 'multi:softmax',
        'num_class': num_classes,
        'max_depth': 4,
        'alpha': 10,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'use_label_encoder': False,
        'eval_metric': 'mlogloss'
    }

    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    return model

# Train and evaluate the initial model
initial_xgb_model = train_initial_xgboost(X_train, y_train, num_classes)
y_pred_initial = initial_xgb_model.predict(X_test)

print("Accuracy of Initial XGBoost Model: {:.4f}".format(accuracy_score(y_test, y_pred_initial)))
evaluate_model(y_test, y_pred_initial, label_encoder, title="Initial XGBoost Model")

# -------------------------------------
# RandomizedSearchCV for XGBoost
# -------------------------------------
def perform_xgb_random_search(X_train, y_train, num_classes):
    param_dist = {
        'max_depth': [4, 6, 8, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'n_estimators': [100, 200, 300],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'gamma': [0, 0.1, 0.3, 0.5],
        'reg_alpha': [0, 0.1, 1, 10],
        'reg_lambda': [0, 1, 10]
    }

    xgb_model = XGBClassifier(
        objective='multi:softmax',
        num_class=num_classes,
        use_label_encoder=False,
        eval_metric='mlogloss',
        random_state=42
    )

    random_search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_dist,
        scoring='accuracy',
        n_iter=50,
        cv=3,
        verbose=2,
        n_jobs=-1,
        random_state=42
    )

    random_search.fit(X_train, y_train)
    print("Best Hyperparameters (XGBoost):", random_search.best_params_)
    return random_search.best_estimator_

# --------------------------
# Tuned XGBoost Model
# --------------------------
tuned_xgb = XGBClassifier(
    objective='multi:softmax',
    num_class=num_classes,
    use_label_encoder=False,
    eval_metric='mlogloss',
    subsample=1.0,
    reg_lambda=0,
    reg_alpha=1,
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    gamma=0.1,
    colsample_bytree=0.6,
    random_state=42
)

tuned_xgb.fit(X_train, y_train)
y_pred_tuned_xgb = tuned_xgb.predict(X_test)

print("Accuracy (Tuned XGBoost): {:.4f}".format(accuracy_score(y_test, y_pred_tuned_xgb)))
evaluate_model(y_test, y_pred_tuned_xgb, label_encoder, title="XGBoost - Tuned Model")

