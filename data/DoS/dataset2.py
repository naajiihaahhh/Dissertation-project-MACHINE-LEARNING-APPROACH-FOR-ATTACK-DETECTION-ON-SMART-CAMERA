import warnings

import pandas as pd
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

# Load dataset.csv
df = pd.read_csv(r"C:\Users\Najihah Azman\PycharmProjects\pythonProject6\data\DoS\Flooding_Based_DDoS_Multi-class_Dataset.csv")# Adjust path if needed

print(df.info())
print(df.head())




# rename the columns
col_names = ['Avg_syn_flag', 'Avg_urg_flag', 'Avg_fin_flag', 'Avg_ack_flag',
       'Avg_psh_flag', 'Avg_rst_flag', 'Avg_DNS_pkt', 'Avg_TCP_pkt',
       'Avg_UDP_pkt', 'Avg_ICMP_pkt', 'Duration_window_flow', 'Avg_delta_time',
       'Min_delta_time', 'Max_delta_time', 'StDev_delta_time',
       'Avg_pkts_lenght', 'Min_pkts_lenght', 'Max_pkts_lenght',
       'StDev_pkts_lenght', 'Avg_small_payload_pkt', 'Avg_payload',
       'Min_payload', 'Max_payload', 'StDev_payload', 'Avg_DNS_over_TCP',
       'Label']

df.columns = col_names
print(df.columns)

for col in col_names:
    print(df[col].value_counts())


print(df['Label'].value_counts())
# check missing variables
print(df.isnull().sum())


X = df.drop(['Label'], axis=1)
y = df['Label']
print(X.head())  # Check first few rows
print(y.head())  # Check first few labels


# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

print(X_train.columns)
# check the shape of X_train and X_test
print(X_train.shape, X_test.shape)
 

from sklearn.preprocessing import LabelEncoder

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Fit and transform the 'Label' column
df['Label'] = label_encoder.fit_transform(df['Label'])

# Check encoding results
print(df['Label'].unique())


