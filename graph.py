import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Force a backend that works (TkAgg, Agg, or Qt5Agg)
matplotlib.use('TkAgg')

# Set global font size
plt.rcParams.update({'font.size': 14})

# Data
categories = ['XGBoost', 'Random Forest', 'Decision Tree']
group1 = [0.99, 0.99, 0.97]
group2 = [0.98, 0.98, 0.94]
group3 = [0.99, 0.99, 0.96]

# Set positions for each bar (this is for the clustered effect)
x = np.arange(len(categories))  # The label locations
width = 0.25  # Width of each bar (smaller width to fit 3 bars)

# Create the plot
fig, ax = plt.subplots()

# Create bars for all three groups with the proper x-axis offset and improved color contrast
bars1 = ax.bar(x - width, group1, width, label='Accuracy', color='#1f77b4')  # Blue
bars2 = ax.bar(x, group2, width, label='Macro Avg Metrics', color='#ff7f0e')  # Orange
bars3 = ax.bar(x + width, group3, width, label='Weighted Avg', color='#2ca02c')  # Green

# Add labels, title, and custom x-axis tick labels
ax.set_xlabel('ML models')
ax.set_ylabel('Score')
ax.set_title('Accuracy')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()

# Set the y-axis range from 0.5 to 1.2
ax.set_ylim(0.90, 1.0)
# Add value labels above the bars
def add_value_labels(bars):
    for bar in bars:
        yval = bar.get_height()  # Get the height of the bar (value)
        ax.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2),
                ha='center', va='bottom', fontsize=14)  # Position the label above the bar

# Call the function to add value labels for each set of bars
add_value_labels(bars1)
add_value_labels(bars2)
add_value_labels(bars3)

# Display the plot
plt.show()
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Force a backend that works (TkAgg, Agg, or Qt5Agg)
matplotlib.use('TkAgg')

# Models
models = ['XGBoost', 'Random Forest', 'Decision Tree']

# Averaged Scores for each model
precision_avg = [np.mean([1.00, 0.99, 0.99]),  # XGBoost
                 np.mean([0.98, 0.99, 0.99]),  # Random Forest
                 np.mean([1.00, 0.97, 0.99])]  # Decision Tree

recall_avg = [np.mean([0.93, 1.00, 0.99]),  # XGBoost
              np.mean([0.94, 1.00, 1.00]),  # Random Forest
              np.mean([0.84, 1.00, 1.00])]  # Decision Tree

f1_avg = [np.mean([0.96, 0.99, 0.99]),  # XGBoost
          np.mean([0.96, 0.99, 0.99]),  # Random Forest
          np.mean([0.91, 0.99, 0.99])]  # Decision Tree

# Set positions for each bar (clustered effect)
x = np.arange(len(models))  # The label locations
width = 0.25  # Width of each bar

# Create the plot
fig, ax = plt.subplots(figsize=(8, 5))

# Create bars for each metric
bars1 = ax.bar(x - width, precision_avg, width, label='Precision', color='#1f77b4')  # Blue
bars2 = ax.bar(x, recall_avg, width, label='Recall', color='#ff7f0e')  # Orange
bars3 = ax.bar(x + width, f1_avg, width, label='F1 Score', color='#2ca02c')  # Green

# Add labels, title, and custom x-axis tick labels
ax.set_xlabel('ML Models')
ax.set_ylabel('Average Score')
ax.set_title('Average Precision, Recall, and F1 Score for Each Model')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()

# Set y-axis range for better visualization
ax.set_ylim(0.9, 1.0)


# Function to add value labels with custom decimal formatting
def add_value_labels(bars, decimals=2, specific_bars={}):
    """
    Adds value labels to bars with custom decimal formatting.

    :param bars: Matplotlib bar container
    :param decimals: Default number of decimal places
    :param specific_bars: Dictionary where keys are bar indices and values are specific decimal places
    """
    for i, bar in enumerate(bars):
        yval = bar.get_height()

        # Use custom decimal places for specific bars, else default to 2
        decimal_places = specific_bars.get(i, decimals)
        formatted_yval = f"{yval:.{decimal_places}f}"

        ax.text(bar.get_x() + bar.get_width() / 2, yval, formatted_yval,
                ha='center', va='bottom', fontsize=14)


# Define specific bars to have 3 decimal places (All Precision bars: XGBoost, Random Forest, Decision Tree)
specific_bars_mapping = {0: 3, 1: 3, 2: 3}  # Index 0, 1, and 2 for Precision bars

# Apply function with custom decimal formatting
add_value_labels(bars1, decimals=2, specific_bars=specific_bars_mapping)  # Precision bars
add_value_labels(bars2)  # Recall bars (default 2 decimals)
add_value_labels(bars3)  # F1 Score bars (default 2 decimals)

# Display the plot
plt.tight_layout()
plt.show()
