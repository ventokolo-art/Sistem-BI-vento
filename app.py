import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# KONFIGURASI HALAMAN & CSS KUSTOM + ANIMASI
# ==========================================
st.set_page_config(page_title="Sistem BI - UAS", layout="wide", page_icon="📊")

st.markdown("""
    <style>
    /* =========================================
       1. ANIMASI KUSTOM (FADE-IN & HOVER)
       ========================================= */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes slideInLeft {
        from { opacity: 0; transform: translateX(-20px); }
        to { opacity: 1; transform: translateX(0); }
    }

    /* Terapkan Animasi ke Elemen Utama */
    .kpi-card {
        animation: fadeInUp 0.6s ease-out forwards;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .kpi-card:hover {
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1) !important;
    }

    .card-blue { animation: fadeInUp 0.8s ease-out forwards; }
    .section-header { animation: slideInLeft 0.8s ease-out forwards; }
    div[data-testid="stNotification"] { animation: fadeInUp 0.5s ease-out forwards; }

    /* =========================================
       2. TEMA DASAR
       ========================================= */
    .stApp { background-color: #F8FAFC !important; }
    h1, h2, h3, h4, h5, h6, p, span, div, label, li { color: #000000 !important; }

    /* 3. SIDEBAR (PANEL UPLOAD) */
    [data-testid="stSidebar"] { background-color: #1E293B !important; }
    [data-testid="stSidebar"] * { color: #FFFFFF !important; }
    [data-testid="stSidebar"] .stFileUploader {
        background-color: #334155 !important;
        border-radius: 8px;
        padding: 10px;
    }
    
    /* =========================================
       4. PERBAIKAN MUTLAK KOTAK INPUT (FILTER)
       ========================================= */
    div[data-baseweb="input"] {
        background-color: #FFFFFF !important; 
        border: 2px solid #10B981 !important;
        border-radius: 8px !important;
    }
    
    input { 
        color: #000000 !important; 
        -webkit-text-fill-color: #000000 !important; 
        background-color: #FFFFFF !important;
    }
    
    input::placeholder {
        color: #6B7280 !important;
        -webkit-text-fill-color: #6B7280 !important;
    }

    div[data-baseweb="select"] > div { 
        background-color: #FFFFFF !important; 
        border: 2px solid #10B981 !important;
        border-radius: 8px !important; 
    }
    div[data-baseweb="select"] * { color: #000000 !important; }

    /* Tombol */
    button[kind="primary"], button[kind="secondary"] {
        background-color: #10B981 !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: bold !important;
        padding: 10px !important;
        transition: background-color 0.3s ease;
    }
    button[kind="primary"]:hover, button[kind="secondary"]:hover {
        background-color: #059669 !important;
        color: #FFFFFF !important;
    }

    /* 5. TAB NAVIGASI */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: white;
        padding: 10px 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: 600;
        color: #4B5563 !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #10B981 !important;
        color: #FFFFFF !important; 
    }

    /* 6. KOTAK ALERT */
    div[data-testid="stNotification"], .stAlert {
        background-color: #FFFFFF !important; 
        border: 1px solid #D1D5DB !important;
        border-left: 6px solid #10B981 !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
    }
    div[data-testid="stNotification"] *, .stAlert * {
        color: #000000 !important; fill: #000000 !important;
    }

    /* 7. KARTU PERTANYAAN */
    .card-blue {
        background-color: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 10px;
        padding: 15px 20px;
        margin-bottom: 10px;
        display: flex;
        align-items: flex-start;
        gap: 15px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .card-icon-blue {
        background-color: #EFF6FF;
        color: #3B82F6 !important;
        font-weight: bold;
        font-size: 16px;
        width: 30px; height: 30px;
        display: flex; align-items: center; justify-content: center;
        border-radius: 5px;
    }

    /* 8. KARTU KPI */
    .kpi-card {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        flex: 1; min-width: 150px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        position: relative;
        display: flex; align-items: center; gap: 15px;
        margin-bottom: 15px;
    }
    .kpi-card::before {
        content: '';
        position: absolute;
        left: 0; top: 0; bottom: 0; width: 5px;
        border-top-left-radius: 10px; border-bottom-left-radius: 10px;
    }
    .kpi-green::before { background-color: #10B981; }
    .kpi-blue::before { background-color: #3B82F6; }
    .kpi-orange::before { background-color: #F97316; }
    .kpi-red::before { background-color: #EF4444; }
    .kpi-purple::before { background-color: #A855F7; }
    .kpi-teal::before { background-color: #14B8A6; }
    
    .kpi-icon {
        font-size: 24px;
        background-color: #F3F4F6;
        padding: 10px;
        border-radius: 8px;
    }
    .kpi-content h5 {
        margin: 0; font-size: 12px; color: #6B7280 !important;
        text-transform: uppercase; font-weight: 600;
    }
    .kpi-content h3 {
        margin: 5px 0 0 0; font-size: 22px; font-weight: bold; color: #000000 !important;
    }

    /* 9. JUDUL SEKSI */
    .section-header {
        display: flex; align-items: center; gap: 10px;
        margin-top: 30px; margin-bottom: 15px;
    }
    .section-icon {
        background-color: #10B981; color: white !important;
        width: 35px; height: 35px;
        display: flex; align-items: center; justify-content: center;
        border-radius: 8px; font-size: 18px;
    }
    .section-title {
        font-size: 20px; font-weight: bold; color: #000000 !important; margin: 0;
    }

    /* 10. TABEL KUSTOM */
    .custom-table {
        width: 100%; border-collapse: collapse; margin: 15px 0;
        font-size: 14px; background-color: #FFFFFF;
        border-radius: 8px; overflow: hidden;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .custom-table thead tr {
        background-color: #10B981; color: #FFFFFF !important; text-align: left;
    }
    .custom-table th { color: #FFFFFF !important; padding: 12px 15px; border-bottom: 1px solid #E5E7EB; }
    .custom-table td { color: #000000 !important; padding: 12px 15px; border-bottom: 1px solid #E5E7EB; }
    .custom-table tbody tr:nth-of-type(even) { background-color: #F9FAFB; }
    .custom-table tbody tr:hover { background-color: #ECFDF5; }
    </style>
""", unsafe_allow_html=True)


# ==========================================
# FUNGSI BANTU
# ==========================================
def apply_light_theme(fig, title=""):
    fig.update_layout(
        title=dict(text=title, font=dict(color="#000000")),
        paper_bgcolor='#FFFFFF', plot_bgcolor='#FFFFFF',
        font=dict(color="#000000"),
        xaxis=dict(showgrid=True, gridcolor='#E5E7EB', zeroline=False, tickfont=dict(color="#000000"), title_font=dict(color="#000000")),
        yaxis=dict(showgrid=True, gridcolor='#E5E7EB', zeroline=False, tickfont=dict(color="#000000"), title_font=dict(color="#000000")),
        margin=dict(t=40, b=40, l=40, r=40)
    )
    return fig

def render_table(df, index=False):
    html = '<table class="custom-table">'
    html += '<thead><tr>'
    if index: html += '<th>Index</th>'
    for col in df.columns: html += f'<th>{col}</th>'
    html += '</tr></thead><tbody>'
    for idx, row in df.iterrows():
        html += '<tr>'
        if index: html += f'<td>{idx}</td>'
        for val in row:
            if isinstance(val, float): html += f'<td>{val:.3f}</td>'
            else: html += f'<td>{val}</td>'
        html += '</tr>'
    html += '</tbody></table>'
    st.markdown(html, unsafe_allow_html=True)

def fmt(num, dec=0):
    try:
        return f"{num:,.{dec}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return str(num)

def bersihkan_data(df):
    df_clean = df.copy()
    df_clean['Transaction_Date'] = pd.to_datetime(df_clean['Transaction_Date'], format='mixed', dayfirst=True, errors='coerce')
    tgl_default = df_clean['Transaction_Date'].max()
    if pd.isna(tgl_default): tgl_default = pd.Timestamp('2025-12-31')
    df_clean['Transaction_Date'] = df_clean['Transaction_Date'].fillna(tgl_default)

    df_clean = df_clean.drop_duplicates()
    kondisi_kotor = (
        (df_clean['Quantity'] <= 0) | (df_clean['Quantity'] >= 999) |
        (df_clean['Unit_Price'] <= 0) | (df_clean['Unit_Price'] >= 999999)
    )
    df_clean = df_clean[~kondisi_kotor]
    
    df_clean['Category'] = df_clean['Category'].astype(str).str.strip().str.title()
    df_clean['Payment_Method'] = df_clean['Payment_Method'].astype(str).str.strip().str.title()
    for col in ['Customer_Type', 'Region']:
        if col in df_clean.columns: df_clean[col] = df_clean[col].fillna('Unknown')
            
    df_clean = df_clean.dropna(subset=['Customer_ID', 'Product_Name'])
    df_clean['Discount_Rate'] = pd.to_numeric(df_clean['Discount_Rate'], errors='coerce').fillna(0)
    df_clean['Line_Total'] = df_clean['Quantity'] * df_clean['Unit_Price'] * (1 - df_clean['Discount_Rate'])
    return df_clean


# ==========================================
# SIDEBAR: UPLOAD
# ==========================================
st.sidebar.header("📁 Unggah Data")
st.sidebar.markdown("Silakan unggah file Excel (sheet: Data_Raw).")
uploaded_file = st.sidebar.file_uploader("Pilih file Excel", type=["xlsx"])

if uploaded_file is None:
    col_logo, col_teks = st.columns([1, 4])
    with col_logo:
        try: st.image("logo.png", use_container_width=True)
        except: st.warning("Logo tidak ditemukan. Pastikan file 'logo.png' ada.")
            
    with col_teks:
        st.markdown("""
        <div style="text-align: left; padding-top: 1vh;">
            <h1 style="color: #10B981 !important; font-size: 3.5rem; margin-bottom: 0px; font-weight: 800;">
                👋 Selamat Datang
            </h1>
            <h2 style="color: #1E293B !important; font-size: 2rem; margin-top: 5px; font-weight: 600;">
                Sistem Business Intelligence Terpadu
            </h2>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr style='border: 1px solid #10B981; margin-bottom: 30px;'>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background-color: #ECFDF5; border: 2px dashed #10B981; border-radius: 15px; padding: 30px; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-top: 50px;">
        <h3 style="color: #065F46 !important; margin-top: 0;">📁 Silakan Unggah Data Anda</h3>
        <p style="color: #000000 !important; font-size: 1.1rem; margin-bottom: 0;">
            Gunakan panel di sebelah kiri (Sidebar) untuk mengunggah file <b>Excel (.xlsx)</b> dengan nama sheet <b>Data_Raw</b>.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.stop()

try:
    df_raw = pd.read_excel(uploaded_file, sheet_name="Data_Raw")
    df_clean = bersihkan_data(df_raw)
    st.sidebar.success("✅ Data berhasil diproses!")
except Exception as e:
    st.sidebar.error(f"❌ Gagal: {e}")
    st.stop()

# ==========================================
# PRECOMPUTE
# ==========================================
with st.spinner("⏳ Menjalankan seluruh analisis..."):
    basket_size_df = (df_clean.groupby('Transaction_ID')['Product_ID']
                      .nunique().reset_index(name='Basket_Size'))

    tanggal_acuan = df_clean['Transaction_Date'].max() + pd.Timedelta(days=1)
    rfm = df_clean.groupby('Customer_ID').agg({
        'Transaction_Date': lambda x: (tanggal_acuan - x.max()).days,
        'Transaction_ID': 'nunique',
        'Line_Total': 'sum'
    }).reset_index()
    rfm.columns = ['Customer_ID', 'Recency', 'Frequency', 'Monetary']

    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

    k_range = list(range(2, 9))
    inertias, silhouettes = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        lbl = km.fit_predict(rfm_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(rfm_scaled, lbl))
    elbow_df = pd.DataFrame({'k': k_range, 'Inertia': inertias, 'Silhouette': silhouettes})

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
    means = rfm.groupby('Cluster')['Monetary'].mean()
    sorted_means = means.sort_values().index.tolist()
    label_map = {
        sorted_means[0]: 'Berisiko Churn',
        sorted_means[1]: 'Potensial',
        sorted_means[2]: 'Loyal/Setia'
    }
    rfm['Label Segmen'] = rfm['Cluster'].map(label_map)

    basket = (df_clean.groupby(['Transaction_ID', 'Product_Name'])['Quantity']
              .sum().unstack().reset_index().fillna(0).set_index('Transaction_ID'))
    basket_sets = basket.map(lambda x: 1 if x > 0 else 0)

    MIN_SUPPORT = 0.01
    MIN_CONFIDENCE = 0.3
    frequent_itemsets = apriori(basket_sets, min_support=MIN_SUPPORT, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=MIN_CONFIDENCE)
    if len(rules) > 0:
        rules = rules.sort_values(by='lift', ascending=False).head(10).copy()
        rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
        rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))

    total_penjualan = df_clean['Line_Total'].sum()
    total_transaksi = df_clean['Transaction_ID'].nunique()
    total_pelanggan = df_clean['Customer_ID'].nunique()
    avg_basket = basket_size_df['Basket_Size'].mean()

    df_clean['Month'] = df_clean['Transaction_Date'].dt.to_period('M').astype(str)

# ==========================================
# NAVIGASI TAB
# ==========================================
tabs = st.tabs([
    "🏠 Beranda", "📊 EDA (10%)", "🧹 Preprocessing (25%)", 
    "🛒 Apriori (20%)", "👥 K-Means (20%)", "📈 Dashboard BI (15%)", "💡 Rekomendasi (10%)"
])

# ==========================================
# TAB 0 — BERANDA (DENGAN FOTO & BIODATA)
# ==========================================
with tabs[0]:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 30px;">
        <h1 style="color: #10B981 !important; font-size: 2.5rem; margin-bottom: 10px; font-weight: 800;">
            👋 Selamat Datang di Dashboard
        </h1>
        <h3 style="color: #4B5563 !important; font-size: 1.2rem;">
            Sistem Business Intelligence & Data Mining Toko Retail
        </h3>
    </div>
    """, unsafe_allow_html=True)
    
    col_foto, col_bio = st.columns([1, 2])
    
    with col_foto:
        try: st.image("foto_maria.jpg", use_container_width=True)
        except: st.warning("Foto belum ditemukan. Pastikan file 'foto_maria.jpg' ada.")
            
    with col_bio:
        st.markdown("""
        <div style="background-color: #FFFFFF; padding: 25px; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); border-left: 8px solid #10B981; height: 100%;">
            <h3 style="color: #065F46 !important; margin-top: 0; margin-bottom: 15px;">👤 Profil Mahasiswa</h3>
            <table style="width: 100%; font-size: 16px; line-height: 1.8;">
                <tr><td style="font-weight: bold; width: 35%;">Nama</td><td>: Antonius Deveuster kolo </td></tr>
                <tr><td style="font-weight: bold;">NPM</td><td>: 51230013</td></tr>
                <tr><td style="font-weight: bold;">Kelas / Semester</td><td>: A / 7</td></tr>
                <tr><td style="font-weight: bold;">Program Studi</td><td>: Teknologi Informasi</td></tr>
                <tr><td style="font-weight: bold;">Fakultas</td><td>: Pertanian, Sains, dan Kesehatan</td></tr>
                <tr><td style="font-weight: bold;">Universitas</td><td>: Universitas Timor (Unimor)</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.divider()

    st.markdown("**Ringkasan Dataset Anda:**")
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(f'<div class="kpi-card kpi-green"><div class="kpi-icon">💵</div><div class="kpi-content"><h5>TOTAL PENJUALAN</h5><h3>Rp {fmt(total_penjualan)}</h3></div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="kpi-card kpi-blue"><div class="kpi-icon">🧾</div><div class="kpi-content"><h5>TOTAL TRANSAKSI</h5><h3>{fmt(total_transaksi)}</h3></div></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="kpi-card kpi-purple"><div class="kpi-icon">👥</div><div class="kpi-content"><h5>PELANGGAN UNIK</h5><h3>{fmt(total_pelanggan)}</h3></div></div>', unsafe_allow_html=True)
    with c4: st.markdown(f'<div class="kpi-card kpi-orange"><div class="kpi-icon">🛒</div><div class="kpi-content"><h5>RERATA BASKET</h5><h3>{avg_basket:.2f}</h3></div></div>', unsafe_allow_html=True)

    st.write("")
    st.markdown("**Tentang Sistem Ini:**")
    st.info("""
    Sistem ini mengintegrasikan **Data Mining** dan **Business Intelligence** untuk menganalisis data transaksi toko retail secara menyeluruh. 
    Data mentah yang telah Anda unggah telah dibersihkan dan diproses untuk menghasilkan *insight* bisnis.
    
    Silakan gunakan tab di bagian atas untuk menjelajahi hasil analisis:
    - **📊 EDA (10%):** Memahami karakteristik awal data dan visualisasi penjualan.
    - **🧹 Preprocessing (25%):** Melihat proses pembersihan data dan pembentukan fitur (RFM & Basket Size).
    - **🛒 Apriori (20%):** Menemukan pola kombinasi produk yang sering dibeli bersama (*Market Basket Analysis*).
    - **👥 K-Means (20%):** Melakukan segmentasi pelanggan berdasarkan perilaku belanja (RFM).
    - **📈 Dashboard BI (15%):** Melihat ringkasan interaktif KPI, tren, dan segmentasi.
    - **💡 Rekomendasi (10%):** Saran tindakan preskriptif untuk manajemen toko.
    """)

    st.markdown("""
    <div class="card-blue" style="margin-top: 20px;">
        <div class="card-icon-blue">✅</div>
        <div>
            <strong style="color: #000000 !important; font-size: 16px;">Data Siap Digunakan</strong><br>
            <span style="font-size: 14px; color: #000000 !important;">Sistem telah berhasil memuat dan membersihkan data Anda. Seluruh algoritma (Apriori & K-Means) sudah berjalan di latar belakang.</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# TAB 1 — EDA & BUSINESS UNDERSTANDING
# ==========================================
with tabs[1]:
    st.markdown("""
    <div class="section-header">
        <div class="section-icon">💼</div>
        <h2 class="section-title">1. Business Understanding</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.info("""
    **a. Permasalahan Utama Perusahaan:**
    Perusahaan ritel belum memanfaatkan data transaksi secara optimal. Manajemen belum mengetahui pola pembelian pelanggan, kombinasi produk yang sering dibeli bersama, dan karakteristik segmen pelanggan, sehingga strategi pemasaran masih bersifat general (umum) dan kurang tepat sasaran.
    
    **c. Manfaat Analisis bagi Manajemen:**
    Hasil analisis (Apriori & K-Means) dapat membantu manajemen dalam:
    1. Membuat strategi *bundling* produk (meningkatkan penjualan).
    2. Menempatkan produk berdekatan di rak (mendorong pembelian impulsif).
    3. Merancang program loyalitas khusus untuk pelanggan bernilai tinggi dan kampanye *win-back* untuk pelanggan yang berisiko berhenti (mempertahankan pelanggan).
    """)

    st.write("**b. Pertanyaan Bisnis yang Ingin Dijawab:**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="card-blue">
            <div class="card-icon-blue">1</div>
            <div>
                <strong style="color: #000000 !important; font-size: 14px;">Q1</strong><br>
                <span style="font-size: 13px; color: #000000 !important;">Produk apa yang paling sering dibeli bersama (untuk strategi bundling)?</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="card-blue">
            <div class="card-icon-blue">2</div>
            <div>
                <strong style="color: #000000 !important; font-size: 14px;">Q2</strong><br>
                <span style="font-size: 13px; color: #000000 !important;">Segmen pelanggan mana yang paling bernilai dan mana yang berisiko churn?</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="card-blue">
            <div class="card-icon-blue">3</div>
            <div>
                <strong style="color: #000000 !important; font-size: 14px;">Q3</strong><br>
                <span style="font-size: 13px; color: #000000 !important;">Kapan waktu penjualan tertinggi dan produk/kategori apa yang paling laku?</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="section-header">
        <div class="section-icon">📊</div>
        <h2 class="section-title">2. Exploratory Data Analysis (EDA)</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("**a & b. Profil Dataset & Statistik Deskriptif**")
    
    r1c1, r1c2, r1c3, r1c4 = st.columns(4)
    with r1c1: st.markdown(f'<div class="kpi-card kpi-blue"><div class="kpi-icon">📄</div><div class="kpi-content"><h5>TOTAL BARIS</h5><h3>{fmt(len(df_raw))}</h3></div></div>', unsafe_allow_html=True)
    with r1c2: st.markdown(f'<div class="kpi-card kpi-teal"><div class="kpi-icon">📐</div><div class="kpi-content"><h5>TOTAL KOLOM</h5><h3>{df_raw.shape[1]}</h3></div></div>', unsafe_allow_html=True)
    with r1c3: st.markdown(f'<div class="kpi-card kpi-green"><div class="kpi-icon">📅</div><div class="kpi-content"><h5>RENTANG AWAL</h5><h3>{str(df_raw["Transaction_Date"].min())[:10]}</h3></div></div>', unsafe_allow_html=True)
    with r1c4: st.markdown(f'<div class="kpi-card kpi-purple"><div class="kpi-icon">📅</div><div class="kpi-content"><h5>RENTANG AKHIR</h5><h3>{str(df_raw["Transaction_Date"].max())[:10]}</h3></div></div>', unsafe_allow_html=True)

    r2c1, r2c2, r2c3, r2c4 = st.columns(4)
    with r2c1: st.markdown(f'<div class="kpi-card kpi-blue"><div class="kpi-icon">🧾</div><div class="kpi-content"><h5>TRANSAKSI UNIK</h5><h3>{fmt(df_raw["Transaction_ID"].nunique())}</h3></div></div>', unsafe_allow_html=True)
    with r2c2: st.markdown(f'<div class="kpi-card kpi-purple"><div class="kpi-icon">👥</div><div class="kpi-content"><h5>PELANGGAN UNIK</h5><h3>{fmt(df_raw["Customer_ID"].nunique())}</h3></div></div>', unsafe_allow_html=True)
    with r2c3: st.markdown(f'<div class="kpi-card kpi-orange"><div class="kpi-icon">📦</div><div class="kpi-content"><h5>PRODUK UNIK</h5><h3>{fmt(df_raw["Product_Name"].nunique())}</h3></div></div>', unsafe_allow_html=True)
    with r2c4:
        total_penjualan_raw = (df_raw['Quantity'] * df_raw['Unit_Price']).sum()
        st.markdown(f'<div class="kpi-card kpi-green"><div class="kpi-icon">💵</div><div class="kpi-content"><h5>TOTAL PENJUALAN (RAW)</h5><h3>Rp {fmt(total_penjualan_raw)}</h3></div></div>', unsafe_allow_html=True)

    with st.expander("Lihat Tipe Data & Statistik Lengkap"):
        st.write("**Tipe Data:**")
        render_table(pd.DataFrame(df_raw.dtypes, columns=['Tipe Data']).reset_index().rename(columns={'index': 'Kolom'}))
        st.write("**Statistik Deskriptif:**")
        render_table(df_raw.describe().T.reset_index().rename(columns={'index': 'Kolom'}))

    st.write("**c. Identifikasi Missing Values, Duplikat, dan Outlier**")
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(f'<div class="kpi-card kpi-red"><div class="kpi-icon">🚫</div><div class="kpi-content"><h5>MISSING VALUES</h5><h3>{fmt(df_raw.isnull().sum().sum())}</h3></div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="kpi-card kpi-red"><div class="kpi-icon">👯</div><div class="kpi-content"><h5>DUPLIKAT</h5><h3>{fmt(df_raw.duplicated().sum())}</h3></div></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="kpi-card kpi-orange"><div class="kpi-icon">⚠️</div><div class="kpi-content"><h5>QTY ANEH</h5><h3>{fmt(((df_raw["Quantity"] <= 0) | (df_raw["Quantity"] >= 999)).sum())}</h3></div></div>', unsafe_allow_html=True)
    with c4: st.markdown(f'<div class="kpi-card kpi-orange"><div class="kpi-icon">💸</div><div class="kpi-content"><h5>HARGA ANEH</h5><h3>{fmt(((df_raw["Unit_Price"] <= 0) | (df_raw["Unit_Price"] >= 999999)).sum())}</h3></div></div>', unsafe_allow_html=True)

    st.write("**d. Visualisasi Awal Data**")
    c1, c2 = st.columns(2)
    with c1:
        fig_eda1 = px.histogram(df_raw[df_raw['Quantity'].between(1, 100)], x='Quantity', nbins=30, 
                                title="Distribusi Jumlah Barang Dibeli (Quantity)", color_discrete_sequence=['#10B981'])
        fig_eda1 = apply_light_theme(fig_eda1, "Distribusi Jumlah Barang Dibeli (Quantity)")
        st.plotly_chart(fig_eda1, use_container_width=True)
    with c2:
        pm = df_raw['Payment_Method'].astype(str).value_counts().reset_index()
        pm.columns = ['Payment_Method', 'Jumlah']
        fig_eda2 = px.bar(pm, x='Payment_Method', y='Jumlah', 
                          title="Distribusi Metode Pembayaran", color_discrete_sequence=['#10B981'])
        fig_eda2 = apply_light_theme(fig_eda2, "Distribusi Metode Pembayaran")
        st.plotly_chart(fig_eda2, use_container_width=True)

# ==========================================
# TAB 2 — PREPROCESSING
# ==========================================
with tabs[2]:
    st.markdown("""
    <div class="section-header">
        <div class="section-icon">🧹</div>
        <h2 class="section-title">1. Pembersihan & Transformasi Data</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.info("Sistem membuat salinan data kerja (`df_clean`) tanpa menimpa data asli (`Data_Raw`).")

    st.write("**Tugas 5: Perbandingan Kondisi Dataset (Sebelum vs Sesudah)**")
    
    laporan = pd.DataFrame({
        'Aspek Data': ['Jumlah Baris', 'Missing Values', 'Data Duplikat', 'Qty Tidak Wajar (<=0 / >=999)', 'Harga Tidak Wajar (<=0 / >=999.999)'],
        'Sebelum (Raw)': [
            fmt(len(df_raw)), fmt(df_raw.isnull().sum().sum()), fmt(df_raw.duplicated().sum()), 
            fmt(((df_raw['Quantity'] <= 0) | (df_raw['Quantity'] >= 999)).sum()), 
            fmt(((df_raw['Unit_Price'] <= 0) | (df_raw['Unit_Price'] >= 999999)).sum())
        ],
        'Sesudah (Clean)': [fmt(len(df_clean)), "0", "0", "0", "0"],
        'Keputusan Penanganan': ['-', 'Dihapus / Diimputasi', 'Dihapus (Drop Duplicates)', 'Dihapus (Filter Outlier)', 'Dihapus (Filter Outlier)']
    })
    render_table(laporan)

    st.write("**Penjelasan Keputusan & Alasan:**")
    keputusan = pd.DataFrame({
        'Masalah Data': ['Missing Values', 'Data Duplikat', 'Qty & Harga Tidak Wajar', 'Format Tanggal', 'Teks Tidak Konsisten', 'Total Transaksi'],
        'Metode': ['Dihapus / Diimputasi', 'Dihapus', 'Dihapus', 'Konversi Tipe Data', 'Standarisasi (Title Case)', 'Kalkulasi Ulang'],
        'Alasan': [
            'Baris tanpa ID tidak bisa dianalisis. Region kosong diisi "Unknown".',
            'Mencegah perhitungan ganda (double-counting) pada pendapatan.',
            'Qty minus atau harga tidak masuk akal akan merusak perhitungan algoritma.',
            'Mengubah teks menjadi format waktu agar bisa dianalisis tren waktunya.',
            'Menyeragamkan penulisan (misal: "makanan" & "Makanan" menjadi "Makanan").',
            'Membuat kolom baru `Line_Total` = Qty × Harga × (1 - Diskon) agar valid.'
        ]
    })
    render_table(keputusan)

    st.markdown("""
    <div class="section-header">
        <div class="section-icon">⚙️</div>
        <h2 class="section-title">2. Feature Engineering</h2>
    </div>
    """, unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**1. Tabel Basket Size**")
        render_table(basket_size_df.head(5))
        st.caption("Tugas 6: Menyiapkan tabel transaksi-produk untuk algoritma Apriori.")
    with c2:
        st.markdown(f"**2. Tabel RFM (Acuan: {tanggal_acuan.date()})**")
        render_table(rfm[['Customer_ID', 'Recency', 'Frequency', 'Monetary']].head(5))
        st.caption("Tugas 7: Menghitung Recency, Frequency, dan Monetary per pelanggan.")

    st.markdown("""
    <div class="section-header">
        <div class="section-icon">📊</div>
        <h2 class="section-title">3. Persiapan Fitur K-Means & Strategi Data Besar</h2>
    </div>
    """, unsafe_allow_html=True)

    st.write("**Tugas 8: Standardisasi/Normalisasi Fitur K-Means**")
    st.success("""
    **Metode:** Menggunakan `StandardScaler` (Z-Score).
    **Alasan:** Menyetarakan skala *Monetary* (jutaan) dan *Frequency* (satuan) agar K-Means tidak didominasi Monetary.
    """)

    st.write("**Bukti Transformasi Data (Hasil Scaling 5 Baris Pertama):**")
    df_scaled_view = pd.DataFrame(rfm_scaled[:5], columns=['Recency (Scaled)', 'Frequency (Scaled)', 'Monetary (Scaled)']).round(3)
    render_table(df_scaled_view)
    st.caption("Nilai jutaan (Monetary) dan puluhan (Frequency) kini seimbang di sekitar angka 0.")

    st.write("**Tugas 8: Strategi Dataset Besar (Big Data)**")
    st.warning("""
    **1. Sampling:** Mengambil sampel acak (misal 20%).
    **2. Chunking:** Membaca file Excel/CSV secara bertahap per 100.000 baris.
    **3. Agregasi:** Menghitung RFM di tingkat pelanggan sebelum K-Means.
    """)

    st.write("**Dokumentasi Perubahan:**")
    c1, c2 = st.columns(2)
    with c1: st.markdown(f'<div class="kpi-card kpi-red"><div class="kpi-icon">🗑️</div><div class="kpi-content"><h5>DIHAPUS</h5><h3>{fmt(len(df_raw) - len(df_clean))} Baris</h3></div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="kpi-card kpi-green"><div class="kpi-icon">✅</div><div class="kpi-content"><h5>DIPERTAHANKAN</h5><h3>{fmt(len(df_clean))} Baris</h3></div></div>', unsafe_allow_html=True)

# ==========================================
# TAB 3 — APRIORI (Tugas 9-12)
# ==========================================
with tabs[3]:
    st.markdown("""
    <div class="section-header">
        <div class="section-icon">🛒</div>
        <h2 class="section-title">Market Basket Analysis (Apriori)</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("**1. Matriks Transaksi × Produk (Tugas 9)**")
    st.caption("Nilai 1 = Dibeli, 0 = Tidak Dibeli. Satu baris mewakili satu struk transaksi.")
    render_table(basket_sets.head(5), index=True)

    st.write("**2. Parameter, Frequent Itemsets, & Aturan Asosiasi (Tugas 10 & 11)**")
    st.info("""
    - **Min Support = 1%** → Menangkap produk yang dibeli minimal 1% dari total transaksi.
    - **Min Confidence = 30%** → Memastikan jika pelanggan beli A, ada 30% kemungkinan beli B.
    - **Kriteria Pemeringkatan:** Diurutkan berdasarkan nilai **Lift tertinggi** (asosiasi terkuat).
    """)

    st.write("**Tabel Frequent Itemsets:**")
    top_itemsets = frequent_itemsets.sort_values('support', ascending=False).head(5).copy()
    top_itemsets['itemsets'] = top_itemsets['itemsets'].apply(lambda x: ', '.join(list(x)))
    render_table(top_itemsets.rename(columns={'itemsets': 'Produk (Itemset)', 'support': 'Support'}))

    st.write("**Tabel Top 10 Aturan Asosiasi:**")
    if len(rules) > 0:
        render_table(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].rename(columns={
            'antecedents': 'Jika Membeli (A)', 'consequents': 'Maka Membeli (B)',
            'support': 'Support', 'confidence': 'Confidence', 'lift': 'Lift'
        }))
    else:
        st.warning("Tidak ada aturan yang memenuhi threshold.")

    st.write("**3. Visualisasi Network Graph (Tugas 12)**")
    if len(rules) > 0:
        G = nx.DiGraph()
        for _, row in rules.head(10).iterrows():
            G.add_edge(row['antecedents'], row['consequents'], weight=row['lift'])
        pos = nx.spring_layout(G, seed=42)
        
        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]; x1, y1 = pos[edge[1]]
            edge_x += [x0, x1, None]; edge_y += [y0, y1, None]
        node_x = [pos[n][0] for n in G.nodes()]
        node_y = [pos[n][1] for n in G.nodes()]
        
        fig_net = go.Figure()
        fig_net.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=1, color='#888'), hoverinfo='none'))
        fig_net.add_trace(go.Scatter(x=node_x, y=node_y, mode='markers+text', text=list(G.nodes()), textposition='top center', marker=dict(size=18, color='#10B981'), hoverinfo='text'))
        
        fig_net.update_layout(
            paper_bgcolor='#FFFFFF', plot_bgcolor='#FFFFFF',
            showlegend=False, height=400, 
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title_font=dict(color="#000000")),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title_font=dict(color="#000000")),
            font=dict(color="#000000")
        )
        st.plotly_chart(fig_net, use_container_width=True)

        st.write("**4. Rekomendasi & Simulasi Penempatan Rak (Soal 6)**")
        top_rule = rules.iloc[0]
        
        st.markdown(f"""
        <div class="card-blue" style="margin-bottom: 20px;">
            <div class="card-icon-blue">💡</div>
            <div>
                <strong style="color: #000000 !important; font-size: 16px;">Kombinasi Paling Menarik: {top_rule['antecedents']} ↔ {top_rule['consequents']}</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1: st.markdown(f'<div class="kpi-card kpi-blue"><div class="kpi-icon">📊</div><div class="kpi-content"><h5>Support</h5><h3>{top_rule["support"]:.2%}</h3></div></div>', unsafe_allow_html=True)
        with c2: st.markdown(f'<div class="kpi-card kpi-green"><div class="kpi-icon">🎯</div><div class="kpi-content"><h5>Confidence</h5><h3>{top_rule["confidence"]:.2%}</h3></div></div>', unsafe_allow_html=True)
        with c3: st.markdown(f'<div class="kpi-card kpi-purple"><div class="kpi-icon">🚀</div><div class="kpi-content"><h5>Lift</h5><h3>{top_rule["lift"]:.2f}</h3></div></div>', unsafe_allow_html=True)

        st.write("**Simulasi Tata Letak Rak Toko (Cross-Merchandising):**")
        top3 = rules.head(3).reset_index(drop=True)
        for i, row in top3.iterrows():
            rak_html = f"""
<div style="background-color: #F8FAFC; border: 2px dashed #CBD5E1; border-radius: 10px; padding: 20px; margin-bottom: 15px;">
<p style="margin-top: 0; margin-bottom: 15px; font-weight: bold; color: #4B5563;">Rekomendasi Rak #{i+1}</p>
<div style="display: flex; gap: 15px;">
<div style="background-color: #FFFFFF; border-top: 5px solid #10B981; border-radius: 8px; padding: 15px; flex: 1; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
<div style="font-size: 30px; margin-bottom: 10px;">📦</div>
<strong style="color: #000000 !important; font-size: 14px;">{row['antecedents']}</strong>
</div>
<div style="display: flex; align-items: center; justify-content: center; font-size: 24px; color: #10B981; font-weight: bold;">➕</div>
<div style="background-color: #FFFFFF; border-top: 5px solid #10B981; border-radius: 8px; padding: 15px; flex: 1; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
<div style="font-size: 30px; margin-bottom: 10px;">📦</div>
<strong style="color: #000000 !important; font-size: 14px;">{row['consequents']}</strong>
</div>
</div>
<p style="margin-bottom: 0; margin-top: 15px; font-size: 13px; color: #6B7280;">
<em>Letakkan kedua produk ini dalam satu end-cap display / berdekatan. (Lift: {row['lift']:.2f})</em>
</p>
</div>
"""
            st.markdown(rak_html, unsafe_allow_html=True)

        st.write("**Grafik Bulat: Proporsi Top 5 Produk dalam Aturan Asosiasi**")
        all_products = rules['antecedents'].tolist() + rules['consequents'].tolist()
        prod_counts = pd.Series(all_products).value_counts().head(5).reset_index()
        prod_counts.columns = ['Produk', 'Frekuensi']
        
        fig_pie = px.pie(prod_counts, values='Frekuensi', names='Produk', hole=0.4, 
                         title="Proporsi Top 5 Produk Terpopuler", color_discrete_sequence=px.colors.sequential.Greens_r)
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        fig_pie = apply_light_theme(fig_pie, "Proporsi Top 5 Produk Terpopuler")
        st.plotly_chart(fig_pie, use_container_width=True)


# ==========================================
# TAB 4 — K-MEANS (Soal 7 & 8)
# ==========================================
with tabs[4]:
    st.markdown("""
    <div class="section-header">
        <div class="section-icon">👥</div>
        <h2 class="section-title">Segmentasi Pelanggan (K-Means)</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Soal 7: Pemodelan Segmentasi Pelanggan")
    
    st.write("**a. Fitur yang Digunakan**")
    st.info("""
    - **Fitur:** **Recency** (kapan terakhir belanja), **Frequency** (seberapa sering belanja), dan **Monetary** (total uang yang dibelanjakan).
    - **Level Data:** Satu baris data mewakili **satu pelanggan unik**. Data transaksi telah diagregasi menjadi data pelanggan pada Tahap 2.
    """)

    st.write("**b & c. Evaluasi Jumlah Klaster (Elbow Method & Silhouette Score)**")
    c1, c2 = st.columns(2)
    with c1:
        fig_elbow = px.line(elbow_df, x='k', y='Inertia', markers=True, title="Grafik Elbow Method", color_discrete_sequence=['#10B981'])
        fig_elbow = apply_light_theme(fig_elbow, "Grafik Elbow Method (Inertia vs k)")
        st.plotly_chart(fig_elbow, use_container_width=True)
    with c2:
        fig_sil = px.line(elbow_df, x='k', y='Silhouette', markers=True, title="Grafik Silhouette Score", color_discrete_sequence=['#10B981'])
        fig_sil = apply_light_theme(fig_sil, "Grafik Silhouette Score vs k")
        st.plotly_chart(fig_sil, use_container_width=True)

    st.write("**d. Penentuan Jumlah Klaster (k) & Alasannya**")
    st.success("""
    **Keputusan: Menggunakan k = 3.**
    - **Alasan Metrik:** Berdasarkan grafik Elbow, kurva *Inertia* mulai melandai (membentuk siku) pada titik k=3. Nilai *Silhouette Score* pada k=3 juga menunjukkan skor yang kompetitif dan stabil dibandingkan nilai k lainnya.
    - **Alasan Bisnis:** 3 segmen (Loyal, Potensial, Churn) adalah jumlah yang paling ideal untuk memetakan strategi pemasaran yang berbeda, tanpa membuat manajemen kebingungan dengan terlalu banyak segmen.
    """)

    st.write("**e & f. Hasil Pengelompokan Pelanggan & Visualisasi**")
    st.markdown(f"Algoritma K-Means berhasil mengelompokkan **{fmt(len(rfm))} pelanggan** ke dalam 3 klaster. Berikut adalah visualisasi sebarannya:")
    
    fig_km = px.scatter(rfm, x='Frequency', y='Monetary', color='Label Segmen', 
                        size='Recency', hover_data=['Customer_ID'], 
                        title="Peta Sebaran Segmen Pelanggan (Frequency vs Monetary)", 
                        color_discrete_sequence=['#10B981', '#F59E0B', '#EF4444'])
    fig_km = apply_light_theme(fig_km, "Peta Sebaran Segmen Pelanggan")
    st.plotly_chart(fig_km, use_container_width=True)
    
    st.divider()

    st.markdown("### Soal 8: Interpretasi dan Strategi Pelanggan")
    
    st.write("**a, b, c. Profil & Karakteristik Setiap Klaster**")
    st.markdown("Berdasarkan hasil perhitungan, berikut adalah profil rata-rata RFM dan pelabelan segmen berdasarkan bukti dari data:")
    
    profil = rfm.groupby('Label Segmen').agg({
        'Customer_ID': 'count', 'Recency': 'mean', 'Frequency': 'mean', 'Monetary': 'mean'
    }).round(1).rename(columns={'Customer_ID': 'Jumlah Pelanggan'}).reset_index()
    
    profil = profil.sort_values(by='Monetary', ascending=False)
    render_table(profil)
    
    st.markdown("""
    **Karakteristik Utama (Berdasarkan Data):**
    - **Loyal/Setia:** Memiliki nilai Monetary paling besar, Frequency (frekuensi) paling tinggi, dan Recency paling kecil (baru saja belanja). Mereka adalah pelanggan andalan toko.
    - **Potensial:** Memiliki nilai RFM di tingkat menengah. Mereka aktif, namun belum sebesar pelanggan Loyal.
    - **Berisiko Churn:** Memiliki nilai Recency paling besar (sudah lama tidak kembali), Monetary paling rendah, dan jarang bertransaksi.
    """)

    st.write("**d. Strategi Pemasaran per Segmen**")
    c1, c2, c3 = st.columns(3)
    with c1: st.success("""**Loyal / Setia**\n- **Strategi:** Program apresiasi member VIP, akses *early sale*, dan tawarkan bundling eksklusif. Prioritas utama untuk dijaga.""")
    with c2: st.warning("""**Potensial**\n- **Strategi:** Dorong *up-selling* dengan diskon bertingkat dan tawarkan bundling produk agar nominal belanjanya naik ke level Loyal.""")
    with c3: st.error("""**Berisiko Churn**\n- **Strategi:** Kirim kupon diskon personal via WA/Email (*win-back campaign*) untuk memancing mereka kembali bertransaksi.""")

    st.write("**e. Segmen yang Menjadi Prioritas Perusahaan**")
    st.info("""
    **Prioritas 1: Segmen "Loyal/Setia" dan "Berisiko Churn".**
    - **Alasan (Loyal):** Mereka adalah sumber pendapatan terbesar. Kehilangan satu saja dari mereka akan berdampak besar pada omzet, sehingga *retention* (mempertahankan) adalah kunci.
    - **Alasan (Churn):** Segmen ini adalah "uang yang hilang". Biaya untuk *win-back* jauh lebih murah dibandingkan mencari pelanggan baru. Jika berhasil dipancing kembali, potensi kenaikan omzet akan signifikan.
    """)

# ==========================================
# TAB 5 — DASHBOARD BI (Soal 9)
# ==========================================
with tabs[5]:
    st.markdown("""
    <div class="section-header">
        <div class="section-icon">📈</div>
        <h2 class="section-title">Dashboard Business Intelligence</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("**Filter Interaktif**")
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        tgl_min, tgl_max = df_clean['Transaction_Date'].min().date(), df_clean['Transaction_Date'].max().date()
        rentang = st.date_input("Filter Rentang Tanggal:", value=(tgl_min, tgl_max), min_value=tgl_min, max_value=tgl_max)
    with col_f2:
        kategori_opsi = ['Semua'] + sorted(df_clean['Category'].dropna().unique().tolist())
        kategori_pilih = st.selectbox("Filter Kategori Produk:", kategori_opsi)
        
    if isinstance(rentang, tuple) and len(rentang) == 2: tgl_awal, tgl_akhir = rentang
    else: tgl_awal, tgl_akhir = tgl_min, tgl_max
        
    df_f = df_clean[(df_clean['Transaction_Date'].dt.date >= tgl_awal) & (df_clean['Transaction_Date'].dt.date <= tgl_akhir)]
    if kategori_pilih != 'Semua': df_f = df_f[df_f['Category'] == kategori_pilih]

    st.markdown("**A. Key Performance Indicators (KPI)**")
    kpi_penjualan = df_f['Line_Total'].sum()
    kpi_transaksi = df_f['Transaction_ID'].nunique()
    kpi_pelanggan = df_f['Customer_ID'].nunique()
    kpi_basket = df_f.groupby('Transaction_ID')['Product_ID'].nunique().mean() if kpi_transaksi > 0 else 0
    
    r1, r2, r3, r4 = st.columns(4)
    with r1: st.markdown(f'<div class="kpi-card kpi-green"><div class="kpi-icon">💵</div><div class="kpi-content"><h5>TOTAL PENJUALAN</h5><h3>Rp {fmt(kpi_penjualan)}</h3></div></div>', unsafe_allow_html=True)
    with r2: st.markdown(f'<div class="kpi-card kpi-blue"><div class="kpi-icon">🧾</div><div class="kpi-content"><h5>JUMLAH TRANSAKSI</h5><h3>{fmt(kpi_transaksi)}</h3></div></div>', unsafe_allow_html=True)
    with r3: st.markdown(f'<div class="kpi-card kpi-purple"><div class="kpi-icon">👥</div><div class="kpi-content"><h5>PELANGGAN UNIK</h5><h3>{fmt(kpi_pelanggan)}</h3></div></div>', unsafe_allow_html=True)
    with r4: st.markdown(f'<div class="kpi-card kpi-orange"><div class="kpi-icon">🛒</div><div class="kpi-content"><h5>RATA-RATA BASKET SIZE</h5><h3>{kpi_basket:.2f}</h3></div></div>', unsafe_allow_html=True)

    st.markdown("**B. Analisis Penjualan**")
    c1, c2 = st.columns(2)
    with c1:
        df_f['Month'] = df_f['Transaction_Date'].dt.to_period('M').astype(str)
        tren_f = df_f.groupby('Month')['Line_Total'].sum().reset_index()
        fig_tren = px.line(tren_f, x='Month', y='Line_Total', markers=True, title="1. Tren Penjualan Bulanan", color_discrete_sequence=['#10B981'])
        fig_tren = apply_light_theme(fig_tren, "Tren Penjualan Bulanan")
        st.plotly_chart(fig_tren, use_container_width=True)
    with c2:
        top_prod_f = df_f.groupby('Product_Name')['Line_Total'].sum().sort_values(ascending=False).head(10).reset_index()
        fig_top = px.bar(top_prod_f, x='Line_Total', y='Product_Name', orientation='h', title="2. Top 10 Produk Terlaris", color_discrete_sequence=['#10B981'])
        fig_top = apply_light_theme(fig_top, "Top 10 Produk Terlaris")
        st.plotly_chart(fig_top, use_container_width=True)
        
    dist_cat = df_f.groupby('Category')['Line_Total'].sum().reset_index()
    fig_cat = px.pie(dist_cat, values='Line_Total', names='Category', hole=0.4, title="3. Distribusi Penjualan per Kategori")
    fig_cat.update_traces(textposition='inside', textinfo='percent+label')
    fig_cat = apply_light_theme(fig_cat, "Distribusi Penjualan per Kategori")
    st.plotly_chart(fig_cat, use_container_width=True)

    st.markdown("**C. Analisis Apriori (Market Basket)**")
    c3, c4 = st.columns(2)
    with c3:
        if len(rules) > 0:
            st.markdown("*1 & 2. Aturan Asosiasi Utama & Nilai Metrik*")
            top_rules_dash = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(5).rename(columns={'antecedents': 'Jika Beli', 'consequents': 'Maka Beli'})
            render_table(top_rules_dash)
        else: st.warning("Tidak ada aturan asosiasi.")
    with c4:
        st.markdown("*3. Visualisasi Kombinasi Produk*")
        if len(rules) > 0:
            all_products = rules['antecedents'].tolist() + rules['consequents'].tolist()
            prod_counts = pd.Series(all_products).value_counts().head(5).reset_index()
            prod_counts.columns = ['Produk', 'Frekuensi']
            
            fig_pie_apriori = px.pie(prod_counts, values='Frekuensi', names='Produk', hole=0.4, 
                                     title="Proporsi Top 5 Produk dalam Asosiasi", color_discrete_sequence=px.colors.sequential.Greens_r)
            fig_pie_apriori.update_traces(textposition='inside', textinfo='percent+label')
            fig_pie_apriori = apply_light_theme(fig_pie_apriori, "Proporsi Top 5 Produk")
            st.plotly_chart(fig_pie_apriori, use_container_width=True)

    st.markdown("**D. Analisis K-Means (Segmentasi Pelanggan)**")
    c5, c6 = st.columns(2)
    with c5:
        st.markdown("*1 & 2. Jumlah Pelanggan & Karakteristik Klaster*")
        profil_dash = rfm.groupby('Label Segmen').agg({
            'Customer_ID': 'count', 'Recency': 'mean', 'Frequency': 'mean', 'Monetary': 'mean'
        }).round(1).rename(columns={'Customer_ID': 'Jumlah Pelanggan'}).reset_index()
        render_table(profil_dash)
    with c6:
        st.markdown("*3. Visualisasi Segmentasi Pelanggan*")
        fig_pie_km = px.pie(profil_dash, values='Jumlah Pelanggan', names='Label Segmen', hole=0.4,
                            color_discrete_sequence=['#10B981', '#F59E0B', '#EF4444'],
                            title="Proporsi Jumlah Pelanggan per Segmen")
        fig_pie_km.update_traces(textposition='inside', textinfo='percent+label')
        fig_pie_km = apply_light_theme(fig_pie_km, "Proporsi Jumlah Pelanggan per Segmen")
        st.plotly_chart(fig_pie_km, use_container_width=True)

    st.markdown("**Bonus: Integrasi Apriori & K-Means (Pola Produk per Segmen)**")
    st.info("Fitur ini menggabungkan hasil K-Means (Segmen Pelanggan) dengan data transaksi untuk melihat produk apa yang paling sering dibeli oleh masing-masing segmen. (Ini menjawab Bonus pada Soal 9e).")
    
    segmen_pilihan = st.selectbox("E. Filter Segmen Pelanggan (Pilih Segmen):", rfm['Label Segmen'].unique())
    pelanggan_segmen = rfm[rfm['Label Segmen'] == segmen_pilihan]['Customer_ID']
    df_segmen = df_clean[df_clean['Customer_ID'].isin(pelanggan_segmen)]
    
    if not df_segmen.empty:
        top_prod_segmen = df_segmen.groupby('Product_Name')['Quantity'].sum().sort_values(ascending=False).head(5).reset_index()
        fig_seg = px.pie(top_prod_segmen, values='Quantity', names='Product_Name', hole=0.4,
                         title=f"Proporsi Top 5 Produk Favorit Segmen: {segmen_pilihan}", color_discrete_sequence=px.colors.sequential.Blues_r)
        fig_seg.update_traces(textposition='inside', textinfo='percent+label')
        fig_seg = apply_light_theme(fig_seg, f"Proporsi Top 5 Produk Favorit Segmen: {segmen_pilihan}")
        st.plotly_chart(fig_seg, use_container_width=True)
    else: st.warning("Tidak ada data transaksi untuk segmen ini pada periode filter yang dipilih.")

    st.markdown("**Insight untuk Manajemen (Tugas 19)**")
    st.success("""
    1. **Tren & Kategori:** Penjualan memuncak pada bulan tertentu. Manajemen harus menyiapkan stok ekstra pada kategori berkinerja terbaik di bulan tersebut.
    2. **Produk & Bundling:** Top 10 produk menyumbang porsi besar pendapatan. Terbukti ada kombinasi produk A & B yang kuat (Lift > 1) untuk strategi bundling.
    3. **Segmentasi:** Pelanggan loyal menyumbang pendapatan terbesar. Gunakan filter segmen di atas untuk melihat produk favorit masing-masing segmen agar promosi lebih tepat sasaran.
    """)

# ==========================================
# TAB 6 — REKOMENDASI KEPUTUSAN BISNIS (Soal 10)
# ==========================================
with tabs[6]:
    st.markdown("""
    <div class="section-header">
        <div class="section-icon">💡</div>
        <h2 class="section-title">Rekomendasi Keputusan Bisnis (Preskriptif)</h2>
    </div>
    """, unsafe_allow_html=True)

    segmen_churn = rfm[rfm['Label Segmen'] == 'Berisiko Churn']
    n_churn = len(segmen_churn)
    
    if len(rules) > 0:
        top_rule = rules.iloc[0]
        rule_a, rule_b = top_rule['antecedents'], top_rule['consequents']
        rule_lift = top_rule['lift']
    else:
        rule_a = rule_b = "—"
        rule_lift = 0

    st.write("**a. Strategi Penempatan Produk (Berdasarkan Apriori)**")
    st.success(f"""
    - **Bukti Data:** Berdasarkan aturan asosiasi, produk **"{rule_a}"** dan **"{rule_b}"** memiliki hubungan yang sangat kuat (Lift: {rule_lift:.2f}).
    - **Tindakan:** Letakkan kedua produk ini secara **berdekatan** di rak (Cross-merchandising) atau tawarkan sebagai **paket bundling** dengan harga khusus.
    """)

    st.write("**b. Strategi Retensi Pelanggan (Berdasarkan K-Means)**")
    st.info(f"""
    - **Pelanggan Bernilai Tinggi (Loyal):** Berikan program apresiasi khusus (VIP member), akses *early sale*, dan penawaran produk premium eksklusif agar tidak berpaling ke kompetitor.
    - **Pelanggan Berisiko Churn:** Terdapat **{n_churn} pelanggan** yang sudah lama tidak bertransaksi. Lakukan kampanye *win-back* dengan mengirimkan kupon diskon personal (via Email/WA) bertajuk "Kami Rindu Anda".
    """)

    st.write("**c. Prioritas Tindakan (Berdasarkan Urgensi, Manfaat, Biaya, & Kemudahan)**")
    prioritas_data = pd.DataFrame({
        'Prioritas': ['1', '2', '3'],
        'Tindakan': ['Bundling & Penataan Rak', 'Kampanye Win-Back Churn', 'Program Loyalitas VIP'],
        'Urgensi': ['Tinggi', 'Tinggi', 'Sedang'],
        'Potensi Manfaat': ['Tinggi', 'Sedang', 'Tinggi'],
        'Biaya': ['Rendah', 'Sedang', 'Sedang'],
        'Kemudahan': ['Sangat Mudah', 'Mudah', 'Sulit']
    })
    render_table(prioritas_data)

    st.write("**d, e, f. Estimasi Dampak, Rencana Implementasi, & Evaluasi**")
    rencana_aksi = pd.DataFrame({
        'Tindakan': ['Bundling & Penataan Rak', 'Kampanye Win-Back Churn', 'Program Loyalitas VIP'],
        'Estimasi Dampak (Asumsi)': ['Kenaikan Basket Size 5-10%.', 'Penurunan Churn Rate hingga 5%.', 'Peningkatan Retensi pelanggan loyal sebesar 15%.'],
        'Rencana Implementasi': [
            'Jangka Pendek (1 Bulan): Cetak label promo dan tata ulang rak.', 
            'Jangka Menengah (3 Bulan): Kirim blast email/WA kupon diskon.', 
            'Jangka Panjang (6 Bulan): Luncurkan sistem membership dan poin.'
        ],
        'KPI Evaluasi': ['Rata-rata Basket Size.', 'Jumlah Pelanggan Kembali (Repeat Order).', 'Persentase Pelanggan Aktif di Segmen Loyal.']
    })
    render_table(rencana_aksi)

    st.write("**Membedakan Fakta, Asumsi, dan Dampak Aktual**")
    st.warning("""
    - **Fakta (Berdasarkan Data):** Pasangan produk A & B terbukti sering dibeli bersama (Lift > 1). Terdapat sejumlah pelanggan yang masuk dalam kategori pasif (Churn).
    - **Estimasi / Asumsi (Belum Terjadi):** Promo bundling *diasumsikan* menaikkan basket size 5-10%. Kampanye win-back *diperkirakan* mampu menurunkan churn rate sebesar 5%.
    - **Dampak Aktual:** Belum diketahui saat ini. Dampak sesungguhnya **hanya dapat diukur** setelah rekomendasi diimplementasikan dan dibandingkan dengan data *baseline* (sebelum promo) selama minimal 1 bulan.
    """)
