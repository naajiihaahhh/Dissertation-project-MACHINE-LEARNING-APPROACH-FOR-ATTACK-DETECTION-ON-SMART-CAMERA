# Preprocessing.py
import pickle

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load your combined dataset (assuming it is already combined)
combined_df_multi_classes = pd.read_csv(r'C:\Users\Najihah Azman\PycharmProjects\pythonProject6\combined_dataset_with_multi-classes.csv')

# Drop the old label column (if it's not used)
combined_df_multi_classes = combined_df_multi_classes.drop(columns=['Attack_Label'])

# Separate features (X) and labels (y)
X = combined_df_multi_classes.drop(columns=['Combined_Attack_Label'])
y = combined_df_multi_classes['Combined_Attack_Label']

print(X.head())
print(y.head())
print(combined_df_multi_classes.shape)

# import XGBoost
import xgboost as xgb

from sklearn.preprocessing import LabelEncoder

# Encode target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  # Convert labels to numeric value
# Print the mapping of labels
label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print("Label Mapping:", label_mapping)  # <-- Add this line here to see the mapping


# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=0)

# Define data_dmatrix (optional, used for boosting)
data_dmatrix = xgb.DMatrix(data=X, label=y_encoded)

# Save the split data (X_train, X_test, y_train, y_test) using pickle
with open('split_data.pkl', 'wb') as f:
    pickle.dump([X_train, X_test, y_train, y_test], f)

print("Data has been saved to 'split_data.pkl'")