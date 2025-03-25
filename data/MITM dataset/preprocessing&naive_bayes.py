import warnings

import pandas as pd
import sns

warnings.filterwarnings('ignore')


# Load dataset.csv
df_simulation = pd.read_csv(r"C:\Users\Najihah Azman\PycharmProjects\pythonProject6\data\MITM dataset\combined_mitm_data_simulation_experiments.csv")# Adjust path if needed
df_real = pd.read_csv(r"C:\Users\Najihah Azman\PycharmProjects\pythonProject6\data\MITM dataset\combined_mitm_data_real_experiments.csv")# Adjust path if needed



df_combined = pd.concat([df_real, df_simulation], ignore_index=True)
print(df_combined.head())
print(df_combined.shape)
print(df_combined.info())

# rename the columns
col_names = ['IRTT', 'TTOC', 'MITR', 'MATR', 'NROC', 'ARP poisoning indicator','Label']
df_combined.columns = col_names
print(df_combined.columns)



# find categorical variables

categorical = [var for var in df_combined.columns if df_combined[var].dtype=='O']

print('There are {} categorical variables\n'.format(len(categorical)))

print('The categorical variables are :\n\n', categorical)

# find numerical variables

numerical = [var for var in df_combined.columns if df_combined[var].dtype!='O']

print('There are {} numerical variables\n'.format(len(numerical)))

print('The numerical variables are :', numerical)

# replace missing values

df_combined[numerical] = df_combined[numerical].fillna(df_combined[numerical].mean())  # or df[numerical].median()
# check missing values in numerical variables

print(df_combined[numerical].isnull().sum())
df_cleaned = df_combined.dropna(axis=0)


X = df_cleaned.drop(['Label'], axis=1)

y = df_cleaned['Label'].astype(int)

# split X and y into training and testing sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
# check the shape of X_train and X_test

print(X.shape, X_test.shape)
print(X_train.dtypes)

# Store column names before scaling
cols = X_train.columns

from sklearn.preprocessing import RobustScaler

# Initialize and apply RobustScaler
scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert back to DataFrame with correct column names
X_train = pd.DataFrame(X_train, columns=cols)
X_test = pd.DataFrame(X_test, columns=cols)



# train a Gaussian Naive Bayes classifier on the training set
from sklearn.naive_bayes import GaussianNB


# instantiate the model
gnb = GaussianNB()


# fit the model
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)

from sklearn.metrics import accuracy_score

print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

y_pred_train = gnb.predict(X_train)
print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))

# print the scores on training and test set

print('Training set score: {:.4f}'.format(gnb.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(gnb.score(X_test, y_test)))

# Print the Confusion Matrix and slice it into four pieces

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print('Confusion matrix\n\n', cm)



