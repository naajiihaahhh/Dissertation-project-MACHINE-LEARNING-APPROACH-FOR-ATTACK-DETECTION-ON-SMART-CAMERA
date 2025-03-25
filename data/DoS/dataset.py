import pandas as pd
import pyshark
import warnings
from sklearn.model_selection import train_test_split


warnings.filterwarnings('ignore')

# Load dataset.csv
df = pd.read_csv(r"C:\Users\Najihah Azman\PycharmProjects\pythonProject6\data\DoS\APA-DDoS-Dataset.csv")# Adjust path if needed

print(df.shape)
# Display first 5 rows
print(df.head())

# Show basic information
print(df.info())

# rename the columns
col_names = ['ip.src', 'ip.dst', 'tcp.srcport', 'tcp.dstport', 'ip.proto', 'frame.len', 'tcp.flags.syn',
             'tcp.flags.reset', 'tcp.flags.push', 'tcp.flags.ack', 'ip.flags.mf', 'ip.flags.df', 'ip.flags.rb', 'tcp.seq',
             'tcp.ack', 'frame.time', 'Packets', 'Bytes', 'Tx Packets', 'Tx Bytes', 'Rx Packets', 'Rx Bytes', 'Label']

df.columns = col_names
print(df.columns)

for col in col_names:
    print(df[col].value_counts())


print(df['Label'].value_counts())
# check missing variables
# print(df.isnull().sum())

X = df.drop(['Label'], axis=1)
y = df['Label']
print(X.head())  # Check first few rows
print(y.head())  # Check first few labels


# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

print(X_train.columns)
# check the shape of X_train and X_test
print(X_train.shape, X_test.shape)

# check data types in X_train
# print(X_train.dtypes)

# print(X_train.head())

# import category encoders
import category_encoders as ce
# encode categorical variables with ordinal encoding
encoder = ce.OrdinalEncoder(cols=['ip.src', 'ip.dst', 'frame.time'])

from sklearn.preprocessing import LabelEncoder
# encode Label separately
label_encoder = LabelEncoder()
df['Label'] = label_encoder.fit_transform(df['Label'])  # Convert to numbers


X_train = encoder.fit_transform(X_train)

X_test = encoder.transform(X_test)

print(X_train.head())
print(X_test.head())


