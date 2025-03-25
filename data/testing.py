import pandas as pd

df_dos= pd.read_csv(r"C:\Users\Najihah Azman\PycharmProjects\pythonProject6\data\DoS\Flooding_Based_DDoS_Multi-class_Dataset.csv")
df_mitm = pd.read_csv(r"C:\Users\Najihah Azman\PycharmProjects\pythonProject6\data\MITM dataset\combined_mitm_data_real_experiments.csv")

print("DOS dataset :", df_dos.columns)
print("MITM dataset :", df_mitm.columns)

# Find common columns
common_columns = list(set(df_mitm.columns) & set(df_dos.columns))

# Keep only common columns
df_mitm = df_mitm[common_columns]
df_dos = df_dos[common_columns]