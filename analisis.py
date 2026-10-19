import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

print("=== MEMULAI PROSES ANALISIS ===")

# KODE INI AKAN MENCARI FILE DENGAN NAMA PERSIS SEPERTI DI FOLDER
FILE = "P8. UTS_Business Intelligence.xlsx" 
df_raw = pd.read_excel(FILE, sheet_name="Data_Raw")

print("\n=== SOAL 2: PROFIL DATA AWAL ===")
print(f"Jumlah Baris: {df_raw.shape[0]}, Kolom: {df_raw.shape[1]}")
df_raw['Transaction_Date'] = pd.to_datetime(df_raw['Transaction_Date'], errors='coerce')
print(f"Rentang Tanggal: {df_raw['Transaction_Date'].min()} s.d. {df_raw['Transaction_Date'].max()}")

print("\n=== SOAL 3: PREPROCESSING ===")
df_clean = df_raw.copy()
baris_awal = len(df_clean)
df_clean = df_clean.drop_duplicates()
kondisi_kotor = (
    (df_clean['Quantity'] <= 0) | (df_clean['Quantity'] >= 999) | 
    (df_clean['Unit_Price'] <= 0) | (df_clean['Unit_Price'] >= 999999)
)
df_clean = df_clean[~kondisi_kotor]
df_clean['Category'] = df_clean['Category'].astype(str).str.strip().str.title()
df_clean['Payment_Method'] = df_clean['Payment_Method'].astype(str).str.strip().str.title()
for col in ['Customer_Type', 'Region']:
    df_clean[col] = df_clean[col].fillna('Unknown')
df_clean = df_clean.dropna(subset=['Customer_ID', 'Product_Name'])
df_clean['Line_Total'] = df_clean['Quantity'] * df_clean['Unit_Price'] * (1 - df_clean['Discount_Rate'])

print(f"Baris Sebelum: {baris_awal}, Sesudah: {len(df_clean)}")
df_clean.to_excel("Data_Clean.xlsx", index=False)
print("Data_Clean.xlsx berhasil dibuat!")

print("\n=== SOAL 5: APRIORI ===")
basket = (df_clean.groupby(['Transaction_ID', 'Product_Name'])['Quantity'].sum().unstack().reset_index().fillna(0).set_index('Transaction_ID'))
basket_sets = basket.map(lambda x: 1 if x > 0 else 0)
frequent_itemsets = apriori(basket_sets, min_support=0.01, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.3)
rules = rules.sort_values(by='lift', ascending=False)
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(5).to_string())

print("\n=== SOAL 7: K-MEANS ===")
tanggal_acuan = df_clean['Transaction_Date'].max() + pd.Timedelta(days=1)
rfm = df_clean.groupby('Customer_ID').agg({
    'Transaction_Date': lambda x: (tanggal_acuan - x.max()).days,
    'Transaction_ID': 'nunique',
    'Line_Total': 'sum'
}).reset_index()
rfm.columns = ['Customer_ID', 'Recency', 'Frequency', 'Monetary']
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
profil = rfm.groupby('Cluster').agg({'Customer_ID': 'count', 'Recency': 'mean', 'Frequency': 'mean', 'Monetary': 'mean'})
print(profil)
rfm.to_excel("Data_RFM.xlsx", index=False)

print("\n=== ANALISIS SELESAI! ===")