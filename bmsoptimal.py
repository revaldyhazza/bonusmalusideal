import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import gamma, chi2
from scipy.special import gamma as gammascipy
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from io import BytesIO
import networkx as nx
from math import gcd
from functools import reduce

# ============================================================================
# KONFIGURASI HALAMAN & STYLING
# ============================================================================

st.set_page_config(
    layout="wide", 
    page_title="Sistem Bonus-Malus Ideal", 
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# CSS styling yang lebih modern dan user-friendly
st.markdown("""
    <style>
        /* ===== MAIN BACKGROUND ===== */
        .stApp {
            background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0f1419 100%);
            color: #e8eaed;
        }
        
        /* ===== TYPOGRAPHY ===== */
        h1, h2, h3 {
            color: #ffffff;
            font-weight: 600;
            letter-spacing: -0.02em;
        }
        
        h1 {
            font-size: 2.5rem !important;
            margin-bottom: 0.5rem !important;
            background: linear-gradient(120deg, #4facfe 0%, #00f2fe 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        h2 {
            font-size: 1.8rem !important;
            color: #4facfe !important;
            border-bottom: 2px solid #4facfe33;
            padding-bottom: 0.5rem;
            margin-top: 1.5rem !important;
        }
        
        h3 {
            font-size: 1.3rem !important;
            color: #7dd3fc !important;
        }
        
        p, li, label {
            color: #c9d1d9;
            line-height: 1.6;
        }
        
        /* ===== CARDS & CONTAINERS ===== */
        .stAlert {
            background-color: #1e293b !important;
            border-left: 4px solid #4facfe !important;
            border-radius: 8px !important;
            padding: 1rem !important;
        }
        
        div[data-testid="stExpander"] {
            background-color: #1e293b;
            border: 1px solid #334155;
            border-radius: 12px;
            margin-bottom: 1rem;
        }
        
        div[data-testid="stExpander"] summary {
            background-color: #1e293b;
            color: #ffffff;
            font-weight: 600;
            padding: 1rem;
            border-radius: 12px 12px 0 0;
        }
        
        /* ===== METRICS ===== */
        div[data-testid="stMetricValue"] {
            font-size: 2rem !important;
            font-weight: 700 !important;
            background: linear-gradient(120deg, #4facfe 0%, #00f2fe 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        div[data-testid="stMetricLabel"] {
            color: #94a3b8 !important;
            font-size: 0.9rem !important;
            font-weight: 500 !important;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        /* ===== TABLES ===== */
        table.dataframe {
            border: none !important;
            border-radius: 12px !important;
            overflow: hidden !important;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3) !important;
        }
        
        .dataframe thead tr {
            background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%) !important;
        }
        
        .dataframe th {
            background-color: transparent !important;
            color: #ffffff !important;
            font-weight: 700 !important;
            font-size: 0.9rem !important;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            padding: 1rem !important;
            border: none !important;
        }
        
        .dataframe td {
            background-color: #1e293b !important;
            color: #e8eaed !important;
            padding: 0.8rem !important;
            border: none !important;
            border-bottom: 1px solid #334155 !important;
        }
        
        .dataframe tbody tr:hover td {
            background-color: #2d3748 !important;
            transition: background-color 0.2s ease;
        }
        
        /* ===== BUTTONS ===== */
        .stButton > button {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 0.6rem 2rem;
            font-weight: 600;
            font-size: 1rem;
            transition: all 0.3s ease;
            box-shadow: 0 4px 6px -1px rgba(79, 172, 254, 0.3);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px -1px rgba(79, 172, 254, 0.5);
        }
        
        .stButton > button[kind="secondary"] {
            background: linear-gradient(135deg, #64748b 0%, #475569 100%);
        }
        
        /* ===== INPUTS ===== */
        .stNumberInput > div > div > input,
        .stTextInput > div > div > input {
            background-color: #1e293b !important;
            color: #e8eaed !important;
            border: 2px solid #334155 !important;
            border-radius: 10px !important;
            padding: 0.6rem !important;
            transition: border-color 0.3s ease;
        }
        
        .stNumberInput > div > div > input:focus,
        .stTextInput > div > div > input:focus {
            border-color: #4facfe !important;
            box-shadow: 0 0 0 3px rgba(79, 172, 254, 0.1) !important;
        }
        
        /* ===== SELECTBOX ===== */
        .stSelectbox > div > div > div {
            background-color: #1e293b !important;
            color: #e8eaed !important;
            border: 2px solid #334155 !important;
            border-radius: 10px !important;
        }
        
        .stSelectbox [data-baseweb="select"] > div {
            background-color: #1e293b !important;
            border-color: #334155 !important;
        }
        
        .stSelectbox [data-baseweb="select"] > div:hover {
            border-color: #4facfe !important;
        }
        
        /* Dropdown menu styling */
        div[role="listbox"] {
            background-color: #1e293b !important;
            border: 2px solid #334155 !important;
            border-radius: 10px !important;
        }
        
        div[role="option"] {
            background-color: #1e293b !important;
            color: #e8eaed !important;
        }
        
        div[role="option"]:hover {
            background-color: #2d3748 !important;
        }
        
        div[role="option"][aria-selected="true"] {
            background-color: #4facfe !important;
            color: white !important;
        }
        
        /* ===== SLIDER ===== */
        .stSlider > div > div > div > div {
            background-color: #4facfe !important;
        }
        
        .stSlider > div > div > div {
            background-color: #334155 !important;
        }
        
        /* ===== RADIO ===== */
        .stRadio > div {
            background-color: #1e293b;
            color: white;
            border-radius: 10px;
            padding: 1rem;
            border: 1px solid #334155;
        }
        
        /* ===== TABS ===== */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: #1e293b;
            color: black;
            padding: 0.5rem;
            border-radius: 12px;
        }
        
        .stTabs [data-baseweb="tab"] {
            background-color: transparent;
            border-radius: 8px;
            color: #94a3b8;
            font-weight: 600;
            padding: 0.8rem 1.5rem;
            border: none;
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            color: #000000 !important;
        }
        
        /* ===== SIDEBAR ===== */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
            border-right: 1px solid #334155;
        }
        
        section[data-testid="stSidebar"] .stMarkdown {
            color: #e8eaed;
        }
        
        /* ===== PROGRESS BAR ===== */
        .stProgress > div > div > div {
            background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        }
        
        /* ===== SCROLLBAR ===== */
        ::-webkit-scrollbar {
            width: 10px;
            height: 10px;
        }
        
        ::-webkit-scrollbar-track {
            background: #1e293b;
            border-radius: 5px;
        }
        
        ::-webkit-scrollbar-thumb {
            background: linear-gradient(180deg, #4facfe 0%, #00f2fe 100%);
            border-radius: 5px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(180deg, #00f2fe 0%, #4facfe 100%);
        }
        
        /* ===== FILE UPLOADER ===== */
        .stFileUploader > div {
            background-color: #1e293b;
            border: 2px dashed #4facfe;
            border-radius: 12px;
            padding: 2rem;
        }
        
        /* ===== INFO BOXES ===== */
        .info-box {
            background: linear-gradient(135deg, #1e293b 0%, #2d3748 100%);
            border-left: 4px solid #4facfe;
            border-radius: 10px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        }
        
        .success-box {
            border-left-color: #10b981;
        }
        
        .warning-box {
            border-left-color: #f59e0b;
        }
        
        .error-box {
            border-left-color: #ef4444;
        }
        
        /* ===== DOWNLOAD BUTTON ===== */
        .stDownloadButton > button {
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            color: black !important;
            border: none;
            border-radius: 10px;
            padding: 0.6rem 1.5rem;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        .stDownloadButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px -1px rgba(16, 185, 129, 0.5);
        }
        
        button[kind="primary"], 
        .stButton > button {
        color: #000000 !important;
        }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# FUNGSI-FUNGSI UTILITY
# ============================================================================

def solve_lambda_cubic(alpha, beta):
    """Mencari akar lambda dari persamaan kubik (Fourth-Degree Loss)"""
    E1 = alpha / beta
    E2 = alpha * (alpha + 1) / (beta**2)
    E3 = alpha * (alpha + 1) * (alpha + 2) / (beta**3)
    coefs = [1.0, -3.0*E1, 3.0*E2, -E3]
    r = np.roots(coefs)
    real_roots = [np.real(x) for x in r if abs(np.imag(x)) < 1e-8]
    pos_real = [x for x in real_roots if x > 0]
    if len(pos_real) == 1:
        return float(pos_real[0])
    if len(pos_real) > 1:
        cand = min(pos_real, key=lambda x: abs(x - E1))
        return float(cand)
    r_sorted = sorted(r, key=lambda z: (abs(np.imag(z)), -np.real(z)))
    return float(np.real(r_sorted[0]))

def pred_pdf(y, a, k, tau, t):
    """Fungsi predictive PDF"""
    comb = gammascipy(y + a + k) / (gammascipy(y + 1) * gammascipy(a + k))
    term1 = ((tau + t) / (tau + t + 1)) ** (a + k)
    term2 = (1 / (tau + t + 1)) ** y
    return comb * term1 * term2

def compute_baseline_probs(a, tau, k_use=0, t_use=0, max_y=5):
    """Compute baseline probabilities"""
    pred_baseline = np.array([pred_pdf(y, a, k_use, tau, t_use) for y in range(max_y + 1)])
    bsly = pred_baseline
    cumsums = np.cumsum(pred_baseline[:-1])
    bsl_ngty_full = 1 - np.append([0], cumsums)
    bsl_ngty = bsl_ngty_full[1:6]
    return tuple(np.concatenate((bsly, bsl_ngty)).tolist())

def simulate_bms(P_matrix, ncd_vec, n_classes, n_years=100, tol=1e-6, country_name="Unknown", pi_init=None, suppress_output=False):
    """Simulasi Bonus-Malus System"""
    if P_matrix.shape != (n_classes, n_classes):
        raise ValueError("P_matrix must be n_classes x n_classes.")
    if len(ncd_vec) != n_classes:
        raise ValueError("ncd_vec must have length n_classes.")

    premium_vec = (1 - np.array(ncd_vec))
    if pi_init is None:
        pi_iter = np.ones(n_classes) / n_classes
    else:
        pi_iter = np.array(pi_init) / np.sum(pi_init)

    prem_list = np.zeros(n_years)
    pi_list = [pi_iter.copy()]

    for n in range(n_years):
        pi_iter = pi_iter @ P_matrix
        pi_iter = pi_iter / np.sum(pi_iter)
        prem_list[n] = np.sum(pi_iter * premium_vec)
        pi_list.append(pi_iter.copy())

    SP = pi_list[-1]
    TV = np.array([np.sum(np.abs(pi_list[n] - SP)) for n in range(n_years)])

    thn_stabil_idx = np.where(TV < tol)[0]
    if len(thn_stabil_idx) == 0:
        if not suppress_output:
            st.warning(f"⚠️ Konvergensi (TV < tol) tidak tercapai untuk {country_name} dalam {n_years} tahun.")
        thn_stabil = np.nan
        pstasioner = np.nan
    else:
        thn_stabil = thn_stabil_idx[0]
        pstasioner = prem_list[thn_stabil - 1]
        if not suppress_output:
            st.success(f"✅ Premi stasioner untuk {country_name} = {pstasioner:.5f} pada tahun ke-{thn_stabil}")

    max_check = min(100, n_years)
    df_conv = pd.DataFrame({
        'Tahun': range(1, max_check + 1),
        'Premi': prem_list[:max_check].round(5),
        'Total_Variation': TV[:max_check].round(6)
    })

    return {
        'df_conv': df_conv,
        'SP': SP,
        'thn_stabil': thn_stabil,
        'TV': TV,
        'pstasioner': pstasioner,
        'prem_list': prem_list
    }

@st.cache_data
def load_data(uploaded_file):
    """Load dan validasi file data"""
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    if not isinstance(df, pd.DataFrame):
        st.error(f"❌ File tidak menghasilkan DataFrame yang valid. Tipe data: {type(df)}")
        return None
    
    if df.empty or len(df.columns) == 0:
        st.error("❌ DataFrame kosong atau tidak memiliki kolom. Pastikan file berisi data tabular.")
        return None
    
    return df

@st.cache_data
def compute_prior(loss_func, a, t):
    """Hitung nilai prior berdasarkan loss function"""
    if loss_func == "Squared-Error Loss":
        return a / t
    elif loss_func == "Absolute Loss Function":
        return gamma.ppf(0.5, a=a, scale=1.0/t)
    else: 
        return solve_lambda_cubic(a, t)

def check_stationary_distribution(transition_matrix):
    """Pengecekan distribusi stasioner"""
    if not isinstance(transition_matrix, np.ndarray) or transition_matrix.ndim != 2:
        st.error("❌ Matriks transisi harus berupa matriks")
        return None
    
    n = transition_matrix.shape[0]
    if transition_matrix.shape[0] != transition_matrix.shape[1]:
        st.error("❌ Matriks transisi harus persegi")
        return None
    
    row_sums = np.sum(transition_matrix, axis=1)
    if np.any(np.abs(row_sums - 1) > 1e-8):
        st.error("❌ Setiap baris matriks transisi harus berjumlah 1")
        return None
    
    if np.any(transition_matrix < 0):
        st.error("❌ Probabilitas tidak boleh negatif")
        return None
    
    result = {}
    
    # 1. Finite states
    st.markdown("### 1. Jumlah Status Berhingga")
    col1, col2 = st.columns([1, 3])
    with col1:
        st.metric("Jumlah Status", n)
    result['finite_states'] = n < np.inf
    if result['finite_states']:
        st.success("✅ Kriteria terpenuhi: Status berhingga.")
    else:
        st.error("❌ Gagal: Jumlah status tidak berhingga.")
    
    # 2. Irreducible
    st.markdown("### 2. Rantai Markov: Accessible dan Irreducible")
    G = nx.from_numpy_array(transition_matrix > 0, create_using=nx.DiGraph)
    num_scc = nx.number_strongly_connected_components(G)
    result['accessible_irreducible'] = num_scc == 1
    col_ir1, col_ir2 = st.columns(2)
    with col_ir1:
        st.metric("Komponen Kuat", num_scc)
    if result['accessible_irreducible']:
        st.success("✅ Kriteria terpenuhi: Rantai irreducible (1 komponen kuat terhubung).")
    else:
        st.error(f"❌ Gagal: Rantai tidak irreducible ({num_scc} komponen kuat).")
        sccs = list(nx.strongly_connected_components(G))
        for i, scc in enumerate(sccs):
            st.warning(f"Komponen {i+1}: Status {sorted([x+1 for x in scc])}")
    
    # 3. Positive recurrent
    st.markdown("### 3. Positive Recurrent")
    result['positive_recurrent'] = result['accessible_irreducible']
    if result['positive_recurrent']:
        st.success("✅ Kriteria terpenuhi: Semua status positive recurrent.")
    else:
        st.error("❌ Gagal: Rantai tidak positive recurrent.")
    
    # 4. Non-absorbing
    st.markdown("### 4. Non-Absorbing States")
    diag = np.diag(transition_matrix)
    off_diag = transition_matrix.copy()
    np.fill_diagonal(off_diag, 0)
    absorbing_states = np.where((diag == 1) & (np.sum(off_diag, axis=1) == 0))[0]
    result['non_absorbing'] = len(absorbing_states) == 0
    col_abs1, col_abs2 = st.columns(2)
    with col_abs1:
        st.metric("Status Absorbing", len(absorbing_states))
    if result['non_absorbing']:
        st.success("✅ Kriteria terpenuhi: Tidak ada status absorbing.")
    else:
        st.error(f"❌ Gagal: Terdapat {len(absorbing_states)} status absorbing di indeks: {[(x+1) for x in absorbing_states.tolist()]}")
    
    # 5. Aperiodic
    st.markdown("### 5. Aperiodic")
    has_self_loops = np.any(np.diag(transition_matrix) > 0)
    if has_self_loops:
        result['aperiodic'] = True
        st.success("✅ Kriteria terpenuhi: Aperiodic (terdapat self-loop).")
    else:
        cycles = []
        try:
            cycles = list(nx.simple_cycles(G))
            cycle_lengths = [len(cycle) for cycle in cycles]
            unique_cycles = sorted(set(cycle_lengths))
            if len(cycles) == 0:
                result['aperiodic'] = True
                st.success("✅ Kriteria terpenuhi: Aperiodic (tidak ada siklus).")
            else:
                def compute_gcd(numbers):
                    return reduce(gcd, numbers)
                period = compute_gcd(cycle_lengths)
                result['aperiodic'] = period == 1
                col_per1, col_per2 = st.columns(2)
                with col_per1:
                    st.metric("Periode (GCD)", period)
                with col_per2:
                    st.metric("Panjang Siklus Unik", len(unique_cycles))
                if result['aperiodic']:
                    st.success("✅ Kriteria terpenuhi: Aperiodic (GCD = 1).")
                else:
                    st.error(f"❌ Gagal: Rantai periodik dengan periode {period}.")
        except Exception as e:
            result['aperiodic'] = False
            st.error(f"❌ Gagal menentukan periodisitas: {e}")
    
    # Kesimpulan
    st.markdown("### Kesimpulan")
    result['has_stationary_distribution'] = all(result.values())
    passed_count = sum(result.values())
    col_con1, col_con2 = st.columns(2)
    with col_con1:
        st.metric("Kriteria Terpenuhi", f"{passed_count}/5")
    if result['has_stationary_distribution']:
        st.success("🎉 Rantai Markov memiliki distribusi stasioner!")
    else:
        st.error("⚠️ Rantai Markov TIDAK memiliki distribusi stasioner.")
    
    return result

# ============================================================================
# DATA NEGARA & MATRIKS
# ============================================================================

countries = {
    'Malaysia': {'ncd': [0.55, 0.45, 0.3833, 0.3, 0.25, 0], 'n_classes': 6},
    'Thailand': {'ncd': [0.4, 0.3, 0.2, 0, -0.2, -0.3, -0.4], 'n_classes': 7},
    'Denmark': {'ncd': [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0, -0.2, -0.5], 'n_classes': 10},
    'British': {'ncd': [0.77, 0.6, 0.55, 0.45, 0.35, 0.25, 0], 'n_classes': 7},
    'Kenya': {'ncd': [0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0], 'n_classes': 7},
    'Hong Kong': {'ncd': [0.6, 0.6, 0.4, 0.3, 0.2, 0], 'n_classes': 6},
    'Swedia': {'ncd': [0.75, 0.6, 0.5, 0.4, 0.3, 0.2, 0], 'n_classes': 7}
}

def P_malaysia(bsly0, bsly1, bsly2, bsly3, bsly4, bsly5, ngty1, ngty2, ngty3, ngty4, ngty5):
    data = [
        bsly0, 0, 0, 0, 0, ngty1,
        bsly0, 0, 0, 0, 0, ngty1,
        0, bsly0, 0, 0, 0, ngty1,
        0, 0, bsly0, 0, 0, ngty1,
        0, 0, 0, bsly0, 0, ngty1,
        0, 0, 0, 0, bsly0, ngty1
    ]
    return np.array(data).reshape(6, 6)

def P_thailand(bsly0, bsly1, bsly2, bsly3, bsly4, bsly5, ngty1, ngty2, ngty3, ngty4, ngty5):
    data = [
        bsly0, 0, 0, bsly1, ngty2, 0, 0,
        bsly0, 0, 0, bsly1, ngty2, 0, 0,
        0, bsly0, 0, bsly1, ngty2, 0, 0,
        0, 0, bsly0, bsly1, ngty2, 0, 0,
        0, 0, bsly0, bsly1, 0, ngty2, 0,
        0, 0, bsly0, bsly1, 0, 0, ngty2,
        0, 0, bsly0, bsly1, 0, 0, ngty2
    ]
    return np.array(data).reshape(7, 7)

def P_denmark(bsly0, bsly1, bsly2, bsly3, bsly4, bsly5, ngty1, ngty2, ngty3, ngty4, ngty5):
    data = [
        bsly0, 0, bsly1, 0, bsly2, 0, bsly3, 0, bsly4, ngty5,
        bsly0, 0, 0, bsly1, 0, bsly2, 0, bsly3, 0, ngty4,
        0, bsly0, 0, 0, bsly1, 0, bsly2, 0, bsly3, ngty4,
        0, 0, bsly0, 0, 0, bsly1, 0, bsly2, 0, ngty3,
        0, 0, 0, bsly0, 0, 0, bsly1, 0, bsly2, ngty3,
        0, 0, 0, 0, bsly0, 0, 0, bsly1, 0, ngty2,
        0, 0, 0, 0, 0, bsly0, 0, 0, bsly1, ngty2,
        0, 0, 0, 0, 0, 0, bsly0, 0, 0, ngty1,
        0, 0, 0, 0, 0, 0, 0, bsly0, 0, ngty1,
        0, 0, 0, 0, 0, 0, 0, 0, bsly0, ngty1
    ]
    return np.array(data).reshape(10, 10)

def P_british(bsly0, bsly1, bsly2, bsly3, bsly4, bsly5, ngty1, ngty2, ngty3, ngty4, ngty5):
    data = [
        bsly0, 0, 0, bsly1, 0, bsly2, ngty3,
        bsly0, 0, 0, bsly1, 0, bsly2, ngty3,
        0, bsly0, 0, 0, bsly1, 0, ngty2,
        0, 0, bsly0, 0, bsly1, 0, ngty2,
        0, 0, 0, bsly0, 0, bsly1, ngty2,
        0, 0, 0, 0, bsly0, 0, ngty1,
        0, 0, 0, 0, 0, bsly0, ngty1
    ]
    return np.array(data).reshape(7, 7)

def P_kenya(bsly0, bsly1, bsly2, bsly3, bsly4, bsly5, ngty1, ngty2, ngty3, ngty4, ngty5):
    data = [
        bsly0, 0, 0, 0, 0, 0, ngty1,
        bsly0, 0, 0, 0, 0, 0, ngty1,
        0, bsly0, 0, 0, 0, 0, ngty1,
        0, 0, bsly0, 0, 0, 0, ngty1,
        0, 0, 0, bsly0, 0, 0, ngty1,
        0, 0, 0, 0, bsly0, 0, ngty1,
        0, 0, 0, 0, 0, bsly0, ngty1
    ]
    return np.array(data).reshape(7, 7)

def P_hongkong(bsly0, bsly1, bsly2, bsly3, bsly4, bsly5, ngty1, ngty2, ngty3, ngty4, ngty5):
    data = [
        bsly0, 0, bsly1, 0, 0, ngty2,
        bsly0, 0, 0, bsly1, 0, ngty2,
        0, bsly0, 0, 0, 0, ngty1,
        0, 0, bsly0, 0, 0, ngty1,
        0, 0, 0, bsly0, 0, ngty1,
        0, 0, 0, 0, bsly0, ngty1
    ]
    return np.array(data).reshape(6, 6)

def P_swedia(bsly0, bsly1, bsly2, bsly3, bsly4, bsly5, ngty1, ngty2, ngty3, ngty4, ngty5):
    data = [
        bsly0, 0, bsly1, 0, bsly2, 0, ngty3,
        bsly0, 0, 0, bsly1, 0, bsly2, ngty3,
        0, bsly0, 0, 0, bsly1, 0, ngty2,
        0, 0, bsly0, 0, 0, bsly1, ngty2,
        0, 0, 0, bsly0, 0, 0, ngty1,
        0, 0, 0, 0, bsly0, 0, ngty1,
        0, 0, 0, 0, 0, bsly0, ngty1
    ]
    return np.array(data).reshape(7, 7)

# Assign P_builder
countries['Malaysia']['P_builder'] = P_malaysia
countries['Thailand']['P_builder'] = P_thailand
countries['Denmark']['P_builder'] = P_denmark
countries['British']['P_builder'] = P_british
countries['Kenya']['P_builder'] = P_kenya
countries['Hong Kong']['P_builder'] = P_hongkong
countries['Swedia']['P_builder'] = P_swedia

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:    
    # File uploader dengan instruksi yang jelas
    st.subheader("📂 Data Klaim")
    uploaded_file = st.file_uploader(
        "Upload file data klaim", 
        type=["csv", "xlsx"],
        help="File harus berisi kolom frekuensi klaim untuk analisis"
    )
    
    if uploaded_file:
        st.success("✅ File berhasil diunggah!")
    
    # Premi dasar dengan penjelasan
    st.subheader("💰 Premi Dasar")
    premium_value = st.number_input(
        "Nilai premi dasar (IDR):", 
        min_value=0, 
        value=1000000, 
        step=10000,
        help="Premi dasar yang akan digunakan untuk perhitungan premi optimal"
    )
    st.caption(f"📊 Premi: Rp {premium_value:,.0f}")
    
    # Panduan penggunaan
    with st.expander("📖 Panduan Penggunaan"):
        st.markdown("""
        **Langkah-langkah:**
        
        1. **Upload Data** - Unggah file CSV/Excel berisi data klaim
        2. **Estimasi Parameter** - Sistem akan menghitung parameter distribusi
        3. **Pilih Loss Function** - Tentukan metode estimasi yang sesuai
        4. **Hitung Premi** - Generate tabel premi optimal
        5. **Simulasi & Analisis** - Eksplorasi berbagai skenario
        
        💡 **Tips:** Mulai dari tab "Data & Estimasi" untuk hasil terbaik.
        """)
    
    st.caption("© 2025 Universitas Gadjah Mada")

# ============================================================================
# HEADER UTAMA
# ============================================================================

st.title("Sistem Bonus-Malus Ideal")

# Hero section dengan informasi penting
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    <div class="info-box">
        <h3>🎯 Tujuan</h3>
        <p>Merancang sistem premi yang adil berdasarkan riwayat klaim pemegang polis</p>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div class="info-box">
        <h3>📈 Metode</h3>
        <p>Pendekatan Markovian dan Bayesian untuk optimalisasi premi</p>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown("""
    <div class="info-box">
        <h3>⚡ Fitur</h3>
        <p>Analisis data, simulasi, dan sensitivitas dalam satu platform</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ============================================================================
# TABS UTAMA
# ============================================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📋 Data & Estimasi", 
    "⚖️ Loss Function", 
    "🏆 Premi Optimal", 
    "🔄 Simulasi", 
    "🏭 Premi Stasioner", 
    "📊 Sensitivitas"
])

if uploaded_file is not None:
    try:
        df = load_data(uploaded_file)
        if df is None:
            st.stop()

        # ========================================================================
        # TAB 1: DATA & ESTIMASI
        # ========================================================================
        with tab1:
            st.header("📋 Data dan Estimasi Parameter")
            
            # Preview data dengan statistik ringkas
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📊 Total Baris", f"{len(df):,}")
            with col2:
                st.metric("📝 Total Kolom", len(df.columns))
            with col3:
                st.metric("💾 Ukuran File", f"{uploaded_file.size / 1024:.1f} KB")
            with col4:
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download CSV",
                    csv,
                    "data_export.csv",
                    "text/csv",
                    use_container_width=True
                )
            
            st.markdown("---")
            
            # Data preview
            st.subheader("🔍 Preview Data")
            with st.expander("Lihat data (100 baris pertama)", expanded=True):
                st.dataframe(
                    df.head(100).style.highlight_null(props='background-color: #ef4444'),
                    use_container_width=True,
                    hide_index=True,
                    height=400
                )
            
            st.markdown("---")
            
            # Pemilihan kolom frekuensi
            st.subheader("📌 Pilih Kolom Frekuensi Klaim")
            col1, col2 = st.columns([2, 1])
            with col1:
                freq_column = st.selectbox(
                    "Kolom yang berisi jumlah klaim per polis:",
                    df.columns,
                    help="Pilih kolom yang berisi data frekuensi klaim"
                )
            with col2:
                if freq_column:
                    st.metric("Jenis Data", str(df[freq_column].dtype))
            
            if freq_column:
                freq_data = df[freq_column]
                
                if freq_data.empty:
                    st.error("❌ Kolom frekuensi kosong. Pilih kolom lain.")
                    st.stop()
                
                # Hitung parameter
                xbar = freq_data.mean()
                skuadrat = freq_data.var()
                tau = xbar / (skuadrat - xbar)
                aa = (xbar ** 2) / (skuadrat - xbar)
                
                st.markdown("---")
                
                # Parameter estimasi dengan visualisasi
                st.subheader("📊 Parameter Distribusi Binomial Negatif")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric(
                        "Rata-rata (x̄)", 
                        f"{xbar:.4f}",
                        help="Mean frekuensi klaim"
                    )
                with col2:
                    st.metric(
                        "Variansi (s²)", 
                        f"{skuadrat:.4f}",
                        help="Variance frekuensi klaim"
                    )
                with col3:
                    st.metric(
                        "Tau (τ)", 
                        f"{tau:.4f}",
                        help="Parameter rate distribusi"
                    )
                with col4:
                    st.metric(
                        "Alpha (α)", 
                        f"{aa:.4f}",
                        help="Parameter shape distribusi"
                    )
                
                # Visualisasi distribusi frekuensi
                st.markdown("---")
                st.subheader("📈 Distribusi Frekuensi Klaim")
                
                fig = go.Figure()
                
                # Histogram
                freq_counts = freq_data.value_counts().sort_index()
                fig.add_trace(go.Bar(
                    x=freq_counts.index,
                    y=freq_counts.values,
                    name='Observasi',
                    marker_color='#4facfe',
                    opacity=0.7
                ))
                
                fig.update_layout(
                    title="Distribusi Frekuensi Klaim",
                    xaxis_title="Jumlah Klaim",
                    yaxis_title="Frekuensi",
                    plot_bgcolor='#0a0e27',
                    paper_bgcolor='#0a0e27',
                    font=dict(color='white'),
                    hovermode='x unified',
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Chi-Square Test
                st.markdown("---")
                with st.expander("🔬 Uji Goodness of Fit (Chi-Square)", expanded=False):
                    st.markdown("""
                    **Hipotesis:**
                    - H₀: Data mengikuti distribusi binomial negatif
                    - H₁: Data tidak mengikuti distribusi binomial negatif
                    - Level signifikansi: α = 0.05
                    """)
                    
                    unique_categories = sorted(freq_data.unique())
                    n_categories = len(unique_categories)
                    observed = freq_data.value_counts().sort_index().reindex(unique_categories, fill_value=0).values

                    P = np.zeros(n_categories)
                    P[0] = (tau / (1 + tau)) ** aa
                    for k in range(n_categories - 1):
                        P[k + 1] = ((k + aa) / ((k + 1) * (1 + tau))) * P[k]
                    P = P / np.sum(P)

                    n = len(freq_data)
                    expected = P * n
                    chisquare = np.sum((observed - expected) ** 2 / expected)
                    df_chi = n_categories - 1 - 2
                    critical_value = chi2.ppf(1 - 0.05, df_chi)
                    p_value = 1 - chi2.cdf(chisquare, df_chi)

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("χ² Statistik", f"{chisquare:.4f}")
                    with col2:
                        st.metric("Derajat Bebas", df_chi)
                    with col3:
                        st.metric("Nilai Kritis", f"{critical_value:.4f}")
                    with col4:
                        st.metric("P-value", f"{p_value:.4f}")

                    if chisquare < critical_value:
                        st.success("✅ **Kesimpulan:** Data cocok dengan distribusi binomial negatif (Gagal tolak H₀)")
                    else:
                        st.warning("⚠️ **Kesimpulan:** Data tidak sepenuhnya cocok dengan distribusi binomial negatif (Tolak H₀)")
                    
                    # Tabel observed vs expected
                    comparison_df = pd.DataFrame({
                        'Klaim': unique_categories,
                        'Observed': observed,
                        'Expected': expected.round(2),
                        'Difference': (observed - expected).round(2)
                    })
                    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

        # ========================================================================
        # TAB 2: LOSS FUNCTION
        # ========================================================================
        with tab2:
            st.header("⚖️ Pemilihan Loss Function")
            
            st.markdown("""
            <div class="info-box">
                <p><strong>Loss Function</strong> digunakan untuk mengestimasi parameter prior/posterior mean 
                dalam pendekatan Bayesian. Pilih metode yang sesuai dengan kebutuhan analisis Anda.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Comparison table
            comparison_data = {
                'Loss Function': ['Squared-Error Loss', 'Absolute Loss Function', 'Fourth-Degree Loss'],
                'Estimator': ['Mean (μ)', 'Median', 'Custom (λ)'],
                'Karakteristik': [
                    'Optimal untuk distribusi simetris',
                    'Robust terhadap outlier',
                    'Fleksibel, meminimalkan error tingkat 4'
                ],
                'Rekomendasi': [
                    'Data normal/terdistribusi baik',
                    'Data dengan outlier/skewed',
                    'Analisis advanced/kompleks'
                ]
            }
            
            st.subheader("📊 Perbandingan Loss Function")
            comp_df = pd.DataFrame(comparison_data)
            st.dataframe(comp_df, use_container_width=True, hide_index=True)
            
            st.markdown("---")
            
            # Selection dengan visual feedback
            st.subheader("🎯 Pilih Loss Function")
            col1, col2 = st.columns([2, 1])
            
            with col1:
                loss_function = st.radio(
                    "Jenis Loss Function:",
                    ("Squared-Error Loss", "Absolute Loss Function", "Fourth-Degree Loss"),
                    help="Pilih berdasarkan karakteristik data Anda"
                )
            
            with col2:
                st.markdown("### ℹ️ Info")
                if loss_function == "Squared-Error Loss":
                    st.info("📐 **Mean-based**\n\nCocok untuk distribusi normal")
                elif loss_function == "Absolute Loss Function":
                    st.info("📊 **Median-based**\n\nRobust untuk outlier")
                else:
                    st.info("⚡ **Advanced**\n\nFleksibel & powerful")
            
            # Hitung dan tampilkan prior value
            prior_val = compute_prior(loss_function, aa, tau)
            
            st.markdown("---")
            st.subheader("🎲 Hasil Estimasi Prior")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Parameter α", f"{aa:.4f}")
            with col2:
                st.metric("Parameter τ", f"{tau:.4f}")
            with col3:
                st.metric(
                    "Prior λ₀", 
                    f"{prior_val:.4f}",
                    help="Nilai prior berdasarkan loss function yang dipilih"
                )
            
            # Formula explanation
            with st.expander("📐 Formula Matematika"):
                if loss_function == "Squared-Error Loss":
                    st.latex(r"\lambda_0 = \frac{\alpha}{\tau}")
                elif loss_function == "Absolute Loss Function":
                    st.latex(r"\lambda_0 = \text{median}(\text{Gamma}(\alpha, \tau))")
                else:
                    st.latex(r"\lambda_0^3 - 3E[X]\lambda_0^2 + 3E[X^2]\lambda_0 - E[X^3] = 0")
                
                st.markdown(f"""
                **Interpretasi:**
                - Loss function yang dipilih: **{loss_function}**
                - Nilai prior λ₀ = **{prior_val:.4f}**
                - Ini akan menjadi baseline untuk perhitungan premi posterior
                """)

        # ========================================================================
        # TAB 3: PREMI OPTIMAL
        # ========================================================================
        with tab3:
            st.header("🏆 Tabel Premi Optimal Sistem Bonus-Malus")
            
            st.markdown("""
            <div class="info-box">
                <p><strong>Premi Optimal</strong> dihitung berdasarkan kombinasi <strong>t</strong> (tahun pertanggungan) 
                dan <strong>k</strong> (jumlah klaim). Semakin tinggi k, semakin tinggi premi yang harus dibayar.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Parameter input dengan layout yang lebih baik
            st.subheader("⚙️ Konfigurasi Tabel")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                max_k_opt = st.number_input(
                    "Maksimum k (klaim):", 
                    min_value=0, 
                    max_value=15, 
                    step=1, 
                    value=4,
                    help="Batas maksimum klaim yang dilakukan pemegang polis"
                )
            
            with col2:
                max_t_opt = st.number_input(
                    "Maksimum t (tahun):", 
                    min_value=0, 
                    max_value=15, 
                    step=1, 
                    value=7,
                    help="Batas maksimum tahun pertanggungan"
                )
            
            with col3:
                st.metric(
                    "Total Kombinasi",
                    f"{(max_t_opt + 1) * (max_k_opt + 1)}",
                    help="Jumlah sel yang akan dihitung"
                )
            
            st.markdown("---")
            
            if st.button("💡 Hitung Tabel Premi", type="primary", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                t_vals = np.arange(0, max_t_opt + 1)
                k_vals = np.arange(0, max_k_opt + 1)
                result = np.full((len(t_vals), len(k_vals)), np.nan)
                
                total_iterations = len(t_vals) * len(k_vals)
                current_iteration = 0
                
                for t_idx, t in enumerate(t_vals):
                    for k_idx, k in enumerate(k_vals):
                        current_iteration += 1
                        status_text.text(f"⚙️ Menghitung... t={t}, k={k} ({current_iteration}/{total_iterations})")
                        progress_bar.progress(current_iteration / total_iterations)
                        
                        if t == 0 and k > 0:
                            continue
                        
                        alpha = aa + k
                        rate = tau + t
                        post_val = compute_prior(loss_function, alpha, rate)
                        
                        if k == 0:
                            factor = post_val / prior_val
                        else:
                            factor = 1 + post_val
                        
                        result[t_idx, k_idx] = premium_value * factor
                
                progress_bar.progress(1.0)
                status_text.text("✅ Perhitungan selesai!")
                
                # Create styled dataframe
                result_df = pd.DataFrame(
                    result, 
                    index=[f"t={t}" for t in t_vals], 
                    columns=[f"k={k}" for k in k_vals]
                ).round(2)
                
                st.markdown("---")
                st.subheader("📊 Tabel Premi Optimal")
                
                # Display with gradient styling
                styled_df = result_df.style.background_gradient(
                    cmap='RdYlGn_r',
                    axis=None
                ).format("{:,.2f}")
                
                st.dataframe(styled_df, use_container_width=True, height=400)
                
                # Statistics summary
                st.markdown("---")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Premi Minimum", f"Rp {np.nanmin(result):,.2f}")
                with col2:
                    st.metric("Premi Maksimum", f"Rp {np.nanmax(result):,.2f}")
                with col3:
                    st.metric("Premi Rata-rata", f"Rp {np.nanmean(result):,.2f}")
                with col4:
                    range_val = np.nanmax(result) - np.nanmin(result)
                    st.metric("Range", f"Rp {range_val:,.2f}")
                
                # Visualisasi heatmap
                st.markdown("---")
                st.subheader("🔥 Visualisasi Heatmap")
                
                fig = go.Figure(data=go.Heatmap(
                    z=result,
                    x=[f"k={k}" for k in k_vals],
                    y=[f"t={t}" for t in t_vals],
                    colorscale='RdYlGn_r',
                    text=np.round(result, 2),
                    texttemplate='%{text:,.0f}',
                    textfont={"size": 10},
                    colorbar=dict(title="Premi (IDR)")
                ))
                
                fig.update_layout(
                    title="Heatmap Premi Optimal",
                    xaxis_title="Jumlah Klaim (k)",
                    yaxis_title="Tahun Pertanggungan (t)",
                    plot_bgcolor='#0a0e27',
                    paper_bgcolor='#0a0e27',
                    font=dict(color='white'),
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Download options
                st.markdown("---")
                col1, col2 = st.columns(2)
                
                with col1:
                    # Excel download
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        result_df.to_excel(writer, index=True, sheet_name='Tabel_Premi')
                    xlsx = output.getvalue()
                    
                    st.download_button(
                        label="📥 Download Excel (.xlsx)",
                        data=xlsx,
                        file_name="tabel_premi_optimal.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
                
                with col2:
                    # CSV download
                    csv = result_df.to_csv(index=True).encode('utf-8')
                    st.download_button(
                        label="📥 Download CSV (.csv)",
                        data=csv,
                        file_name="tabel_premi_optimal.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

        # ========================================================================
        # TAB 4: SIMULASI
        # ========================================================================
        with tab4:
            st.header("🔄 Simulasi Premi Ideal")
            
            st.markdown("""
            <div class="info-box">
                <p><strong>Simulasi</strong> memungkinkan Anda untuk memodelkan skenario klaim tertentu 
                dan melihat bagaimana premi berkembang dari waktu ke waktu.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Mode selection
            simulation_mode = st.radio(
                "Pilih Mode Simulasi:",
                ("🖊️ Manual Input", "🌍 Berdasarkan Sistem Negara"),
                horizontal=True
            )
            
            st.markdown("---")
            
            if "Manual" in simulation_mode:
                st.subheader("🖊️ Simulasi Manual")
                st.info("💡 Masukkan jumlah klaim per tahun. Premi dihitung independen untuk setiap tahun.")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    num_years_opt = st.slider(
                        "Jumlah tahun simulasi:", 
                        min_value=1, 
                        max_value=10, 
                        value=5,
                        help="Periode simulasi yang ingin dianalisis"
                    )
                
                with col2:
                    max_k_sim_opt = st.slider(
                        "Batas maksimum klaim/tahun:", 
                        min_value=0, 
                        max_value=20, 
                        value=4,
                        help="Batasan klaim maksimal per tahun"
                    )
                
                st.markdown("---")
                
                # Input klaim per tahun dengan layout yang lebih baik
                st.subheader("📝 Input Klaim per Tahun")
                
                simulation_data = []
                
                # Create a container for inputs
                for year in range(1, num_years_opt + 1):
                    col1, col2, col3 = st.columns([2, 1, 2])
                    
                    with col1:
                        claims_this_year = st.number_input(
                            f"Tahun {year}:",
                            min_value=0,
                            max_value=max_k_sim_opt,
                            step=1,
                            key=f"claims_manual_{year}",
                            help=f"Jumlah klaim pada tahun ke-{year}"
                        )
                    
                    # Calculate premium
                    k = claims_this_year
                    t = year
                    alpha = aa + k
                    rate = tau + t
                    post_val = compute_prior(loss_function, alpha, rate)
                    factor = post_val / prior_val if k == 0 else 1 + post_val
                    premium = premium_value * factor
                    
                    with col2:
                        if k == 0:
                            st.success("✅ No Claim")
                        else:
                            st.warning(f"⚠️ {k} Klaim")
                    
                    with col3:
                        st.metric(
                            f"Premi",
                            f"Rp {premium:,.0f}",
                            delta=None
                        )
                    
                    simulation_data.append({
                        "Tahun": year,
                        "Klaim": k,
                        "Premi (IDR)": premium
                    })
                
                if simulation_data:
                    st.markdown("---")
                    st.subheader("📊 Hasil Simulasi")
                    
                    sim_df = pd.DataFrame(simulation_data)
                    
                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        total_claims = sim_df['Klaim'].sum()
                        st.metric("Total Klaim", f"{total_claims}")
                    
                    with col2:
                        avg_claims = sim_df['Klaim'].mean()
                        st.metric("Rata-rata Klaim/Tahun", f"{avg_claims:.2f}")
                    
                    with col3:
                        avg_premium = sim_df['Premi (IDR)'].mean()
                        st.metric("Rata-rata Premi", f"Rp {avg_premium:,.0f}")
                    
                    with col4:
                        total_premium = sim_df['Premi (IDR)'].sum()
                        st.metric("Total Premi", f"Rp {total_premium:,.0f}")
                    
                    # Table
                    st.dataframe(
                        sim_df.style.format({"Premi (IDR)": "Rp {:,.0f}"}),
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Chart
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=sim_df['Tahun'],
                        y=sim_df['Premi (IDR)'],
                        mode='lines+markers',
                        name='Premi',
                        line=dict(color='#4facfe', width=3),
                        marker=dict(size=10)
                    ))
                    
                    fig.update_layout(
                        title="Perkembangan Premi per Tahun",
                        xaxis_title="Tahun",
                        yaxis_title="Premi (IDR)",
                        plot_bgcolor='#0a0e27',
                        paper_bgcolor='#0a0e27',
                        font=dict(color='white'),
                        hovermode='x unified',
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Download
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        sim_df.to_excel(writer, index=False, sheet_name='Simulasi_Manual')
                    xlsx = output.getvalue()
                    
                    st.download_button(
                        label="📥 Download Hasil Simulasi (.xlsx)",
                        data=xlsx,
                        file_name="simulasi_manual.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
            
            else:  # Sistem Negara
                st.subheader("🌍 Simulasi Berdasarkan Sistem Negara")
                st.info("💡 Simulasi menggunakan matriks transisi dari sistem bonus-malus negara tertentu.")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    country_sim = st.selectbox(
                        "Pilih Sistem Negara:",
                        list(countries.keys()),
                        help="Setiap negara memiliki struktur bonus-malus yang berbeda"
                    )
                
                with col2:
                    # Info sistem
                    ncd_info = countries[country_sim]['ncd']
                    n_classes_info = countries[country_sim]['n_classes']
                    st.metric("Jumlah Kelas", n_classes_info)
                    st.caption(f"NCD: {ncd_info}")
                
                # Determine max k based on country
                k_max_dict = {
                    'Malaysia': 1, 'Thailand': 2, 'Denmark': 5,
                    'British': 3, 'Kenya': 1, 'Hong Kong': 2, 'Swedia': 3
                }
                max_k_sim_opt = k_max_dict.get(country_sim, 1)
                
                st.markdown("---")
                
                if st.button("🚀 Jalankan Simulasi", type="primary", use_container_width=True):
                    with st.spinner("⚙️ Menghitung..."):
                        # Calculate convergence
                        probs_baseline = compute_baseline_probs(aa, tau)
                        P_matrix_sim = countries[country_sim]['P_builder'](*probs_baseline)
                        ncd_vec_sim = countries[country_sim]['ncd']
                        n_classes_sim = countries[country_sim]['n_classes']
                        
                        result_conv = simulate_bms(
                            P_matrix_sim, ncd_vec_sim, n_classes_sim,
                            n_years=100, tol=1e-6, country_name=country_sim,
                            suppress_output=True
                        )
                        
                        num_years_opt = int(result_conv['thn_stabil']) if not np.isnan(result_conv['thn_stabil']) else 7
                    
                    st.success(f"✅ Sistem {country_sim}: Periode Waktu Konvergensi = {num_years_opt} tahun, Klaim Maksimum = {max_k_sim_opt}")
                    
                    st.markdown("---")
                    st.subheader("📝 Input Klaim per Tahun")
                    
                    simulation_data = []
                    cumulative_k = 0
                    
                    for year in range(1, num_years_opt + 1):
                        col1, col2, col3 = st.columns([2, 1, 2])
                        
                        with col1:
                            remaining_capacity = max_k_sim_opt - cumulative_k
                            claims_input = st.number_input(
                                f"Tahun {year} (max: {remaining_capacity}):",
                                min_value=0,
                                max_value=remaining_capacity,
                                step=1,
                                key=f"claims_country_{year}"
                            )
                        
                        cumulative_k += claims_input
                        k = cumulative_k
                        t = year
                        
                        alpha = aa + k
                        rate = tau + t
                        post_val = compute_prior(loss_function, alpha, rate)
                        factor = post_val / prior_val if k == 0 else 1 + post_val
                        premium = premium_value * factor
                        
                        with col2:
                            st.metric("Total Klaim", f"{k}")
                        
                        with col3:
                            st.metric("Premi", f"Rp {premium:,.0f}")
                        
                        simulation_data.append({
                            "Tahun": year,
                            "Klaim Tahun Ini": claims_input,
                            "Total Klaim (k)": k,
                            "Premi (IDR)": premium
                        })
                    
                    if simulation_data:
                        st.markdown("---")
                        st.subheader("📊 Hasil Simulasi")
                        
                        sim_df = pd.DataFrame(simulation_data)
                        
                        # Summary
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            total_claims = sim_df['Total Klaim (k)'].iloc[-1]
                            st.metric("Total Klaim Kumulatif", f"{total_claims}")
                        
                        with col2:
                            avg_premium = sim_df['Premi (IDR)'].mean()
                            st.metric("Rata-rata Premi", f"Rp {avg_premium:,.0f}")
                        
                        with col3:
                            final_premium = sim_df['Premi (IDR)'].iloc[-1]
                            st.metric("Premi Akhir", f"Rp {final_premium:,.0f}")
                        
                        # Table
                        st.dataframe(
                            sim_df.style.format({"Premi (IDR)": "Rp {:,.0f}"}),
                            use_container_width=True,
                            hide_index=True
                        )
                        
                        # Chart
                        fig = make_subplots(specs=[[{"secondary_y": True}]])
                        
                        fig.add_trace(
                            go.Scatter(
                                x=sim_df['Tahun'],
                                y=sim_df['Premi (IDR)'],
                                name='Premi',
                                line=dict(color='#4facfe', width=3),
                                mode='lines+markers'
                            ),
                            secondary_y=False
                        )
                        
                        fig.add_trace(
                            go.Bar(
                                x=sim_df['Tahun'],
                                y=sim_df['Klaim Tahun Ini'],
                                name='Klaim',
                                marker_color='#f59e0b',
                                opacity=0.6
                            ),
                            secondary_y=True
                        )
                        
                        fig.update_xaxes(title_text="Tahun")
                        fig.update_yaxes(title_text="Premi (IDR)", secondary_y=False)
                        fig.update_yaxes(title_text="Klaim", secondary_y=True)
                        
                        fig.update_layout(
                            title=f"Simulasi BMS {country_sim}",
                            plot_bgcolor='#0a0e27',
                            paper_bgcolor='#0a0e27',
                            font=dict(color='white'),
                            hovermode='x unified',
                            height=450
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Download
                        output = BytesIO()
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            sim_df.to_excel(writer, index=False, sheet_name=f'Simulasi_{country_sim}')
                        xlsx = output.getvalue()
                        
                        st.download_button(
                            label=f"📥 Download Simulasi {country_sim} (.xlsx)",
                            data=xlsx,
                            file_name=f"simulasi_{country_sim}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )

        # ========================================================================
        # TAB 5: PREMI STASIONER
        # ========================================================================
        with tab5:
            st.header("🏭 Perhitungan Premi Stasioner")
            
            st.markdown("""
            <div class="info-box">
                <p><strong>Premi Stasioner</strong> adalah premi jangka panjang yang dicapai sistem 
                setelah konvergensi. Ini menunjukkan keseimbangan distribusi antar kelas dalam sistem bonus-malus.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Mode selection
            matrix_mode = st.radio(
                "Pilih Sumber Matriks:",
                ("🌍 Sistem Negara", "⚙️ Matriks Custom"),
                horizontal=True
            )
            
            st.markdown("---")
            
            if "Sistem" in matrix_mode:
                # Sistem Negara
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    country = st.selectbox(
                        "Pilih Sistem Negara:",
                        list(countries.keys()),
                        help="Pilih negara untuk simulasi"
                    )
                
                with col2:
                    n_years_stat = st.slider(
                        "Tahun simulasi:",
                        min_value=50,
                        max_value=1000,
                        value=100,
                        help="Iterasi untuk mencapai konvergensi"
                    )
                
                with col3:
                    tol_stat = st.number_input(
                        "Toleransi:",
                        min_value=1e-10,
                        max_value=1e-3,
                        value=1e-6,
                        step=1e-7,
                        format="%.0e",
                        help="Threshold untuk konvergensi"
                    )
                
                st.markdown("---")
                
                # Baseline or custom parameters
                use_baseline = st.checkbox(
                    "Gunakan Parameter Baseline (k=0, t=0)",
                    value=True,
                    help="Gunakan probabilitas baseline atau sesuaikan"
                )
                
                k_stat = 0
                t_stat = 0
                
                if not use_baseline:
                    col_k, col_t = st.columns(2)
                    with col_k:
                        k_stat = st.number_input("Nilai k:", min_value=0, max_value=10, step=1, value=0)
                    with col_t:
                        t_stat = st.number_input("Nilai t:", min_value=0, max_value=10, step=1, value=0)
                
                st.markdown("---")
                
                if st.button("🚀 Hitung Premi Stasioner", type="primary", use_container_width=True):
                    with st.spinner("⚙️ Menghitung konvergensi..."):
                        progress_bar = st.progress(0)
                        
                        probs = compute_baseline_probs(aa, tau, k_stat, t_stat)
                        P_matrix = countries[country]['P_builder'](*probs)
                        ncd_vec = countries[country]['ncd']
                        n_classes = countries[country]['n_classes']
                        
                        result_stat = simulate_bms(
                            P_matrix, ncd_vec, n_classes,
                            n_years=n_years_stat, tol=tol_stat,
                            country_name=country, suppress_output=False
                        )
                        
                        progress_bar.progress(1.0)
                    
                    st.markdown("---")
                    
                    # Display results
                    if not np.isnan(result_stat['pstasioner']):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric(
                                "Premi Stasioner",
                                f"{result_stat['pstasioner']:.5f}",
                                help="Premi keseimbangan jangka panjang"
                            )
                        
                        with col2:
                            st.metric(
                                "Tahun Konvergensi",
                                f"{int(result_stat['thn_stabil'])}",
                                help="Tahun mencapai keseimbangan"
                            )
                        
                        with col3:
                            final_tv = result_stat['TV'][int(result_stat['thn_stabil'])-1]
                            st.metric(
                                "Total Variation",
                                f"{final_tv:.6g}",
                                help="Ukuran jarak dari distribusi stasioner"
                            )
                    
                    # Convergence table
                    st.subheader("📊 Tabel Konvergensi")
                    st.dataframe(
                        result_stat['df_conv'].style.format({
                            "Premi": "{:.5f}",
                            "Total_Variation": "{:.6g}"
                        }),
                        use_container_width=True,
                        hide_index=True,
                        height=400
                    )
                    
                    # Charts
                    st.markdown("---")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Premium convergence
                        fig1 = go.Figure()
                        
                        max_display = min(100, n_years_stat)
                        
                        fig1.add_trace(go.Scatter(
                            x=list(range(1, max_display + 1)),
                            y=result_stat['prem_list'][:max_display],
                            mode='lines',
                            name='Premi',
                            line=dict(color='#4facfe', width=2)
                        ))
                        
                        if not np.isnan(result_stat['pstasioner']):
                            fig1.add_hline(
                                y=result_stat['pstasioner'],
                                line_dash="dash",
                                line_color="#10b981",
                                annotation_text="Premi Stasioner"
                            )
                        
                        fig1.update_layout(
                            title="Konvergensi Premi",
                            xaxis_title="Tahun",
                            yaxis_title="Premi",
                            plot_bgcolor='#0a0e27',
                            paper_bgcolor='#0a0e27',
                            font=dict(color='white'),
                            height=400
                        )
                        
                        st.plotly_chart(fig1, use_container_width=True)
                    
                    with col2:
                        # Total variation
                        fig2 = go.Figure()
                        
                        fig2.add_trace(go.Scatter(
                            x=list(range(1, max_display + 1)),
                            y=result_stat['TV'][:max_display],
                            mode='lines',
                            name='Total Variation',
                            line=dict(color='#f59e0b', width=2)
                        ))
                        
                        fig2.add_hline(
                            y=tol_stat,
                            line_dash="dash",
                            line_color="#ef4444",
                            annotation_text="Threshold"
                        )
                        
                        fig2.update_layout(
                            title="Total Variation",
                            xaxis_title="Tahun",
                            yaxis_title="Total Variation",
                            yaxis_type="log",
                            plot_bgcolor='#0a0e27',
                            paper_bgcolor='#0a0e27',
                            font=dict(color='white'),
                            height=400
                        )
                        
                        st.plotly_chart(fig2, use_container_width=True)
                    
                    # Stationary distribution
                    st.markdown("---")
                    st.subheader("📊 Distribusi Stasioner")
                    
                    sp_df = pd.DataFrame({
                        'Kelas': [f"Kelas {i+1}" for i in range(n_classes)],
                        'Probabilitas': result_stat['SP'],
                        'NCD': ncd_vec,
                        'Premi Relatif': 1 - np.array(ncd_vec)
                    })
                    
                    st.dataframe(
                        sp_df.style.format({
                            'Probabilitas': '{:.6f}',
                            'NCD': '{:.2f}',
                            'Premi Relatif': '{:.4f}'
                        }).background_gradient(subset=['Probabilitas'], cmap='Blues'),
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Bar chart of stationary distribution
                    fig3 = go.Figure()
                    
                    fig3.add_trace(go.Bar(
                        x=sp_df['Kelas'],
                        y=sp_df['Probabilitas'],
                        marker_color='#4facfe',
                        text=sp_df['Probabilitas'].round(4),
                        textposition='auto'
                    ))
                    
                    fig3.update_layout(
                        title="Distribusi Stasioner per Kelas",
                        xaxis_title="Kelas",
                        yaxis_title="Probabilitas",
                        plot_bgcolor='#0a0e27',
                        paper_bgcolor='#0a0e27',
                        font=dict(color='white'),
                        showlegend=False,
                        height=400
                    )
                    
                    st.plotly_chart(fig3, use_container_width=True)
                    
                    # Download
                    st.markdown("---")
                    
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        result_stat['df_conv'].to_excel(writer, index=False, sheet_name='Konvergensi')
                        sp_df.to_excel(writer, index=False, sheet_name='Distribusi_Stasioner')
                    xlsx = output.getvalue()
                    
                    st.download_button(
                        label=f"📥 Download Hasil {country} (.xlsx)",
                        data=xlsx,
                        file_name=f"premi_stasioner_{country}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
            
            else:
                # Custom Matrix
                st.subheader("⚙️ Definisi Matriks Custom")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    n_classes_custom = st.number_input(
                        "Jumlah kelas:",
                        min_value=2,
                        max_value=20,
                        value=3,
                        help="Ukuran matriks transisi n×n"
                    )
                
                with col2:
                    st.info(f"📊 Matriks: {n_classes_custom}×{n_classes_custom}")
                
                st.markdown("---")
                
                # Matrix editor
                st.subheader("📝 Editor Matriks Transisi")
                st.caption("⚠️ Setiap baris harus berjumlah 1.0 (probabilitas transisi)")
                
                default_matrix = np.full((n_classes_custom, n_classes_custom), 1.0 / n_classes_custom)
                df_matrix = pd.DataFrame(
                    default_matrix,
                    columns=[f"To {j+1}" for j in range(n_classes_custom)],
                    index=[f"From {i+1}" for i in range(n_classes_custom)]
                )
                
                edited_df = st.data_editor(
                    df_matrix,
                    num_rows="fixed",
                    column_config={
                        col: st.column_config.NumberColumn(
                            col,
                            min_value=0.0,
                            max_value=1.0,
                            step=0.01,
                            format="%.4f"
                        ) for col in df_matrix.columns
                    },
                    use_container_width=True,
                    hide_index=False
                )
                
                P_matrix_custom = edited_df.values
                
                # Verify row sums
                row_sums = P_matrix_custom.sum(axis=1)
                if not np.allclose(row_sums, 1.0, atol=1e-6):
                    st.error(f"⚠️ Peringatan: Beberapa baris tidak berjumlah 1.0: {row_sums.round(4)}")
                else:
                    st.success("✅ Matriks valid: Semua baris berjumlah 1.0")
                
                st.markdown("---")
                
                # NCD vector
                st.subheader("💰 Vektor NCD (No-Claim Discount)")
                
                available_ncd = [c for c in countries if len(countries[c]['ncd']) == n_classes_custom]
                
                if available_ncd:
                    ncd_source = st.selectbox(
                        "Sumber NCD:",
                        ["Manual"] + available_ncd,
                        help="Pilih preset atau input manual"
                    )
                    
                    if ncd_source == "Manual":
                        ncd_vec_custom = []
                        cols = st.columns(min(n_classes_custom, 5))
                        for i in range(n_classes_custom):
                            with cols[i % 5]:
                                ncd_i = st.number_input(
                                    f"Kelas {i+1}:",
                                    min_value=-1.0,
                                    max_value=1.0,
                                    value=0.0,
                                    step=0.01,
                                    key=f"ncd_custom_{i}"
                                )
                                ncd_vec_custom.append(ncd_i)
                    else:
                        ncd_vec_custom = countries[ncd_source]['ncd']
                        st.info(f"✅ Menggunakan NCD dari {ncd_source}: {ncd_vec_custom}")
                else:
                    st.info("Input manual NCD untuk setiap kelas:")
                    ncd_vec_custom = []
                    cols = st.columns(min(n_classes_custom, 5))
                    for i in range(n_classes_custom):
                        with cols[i % 5]:
                            ncd_i = st.number_input(
                                f"Kelas {i+1}:",
                                min_value=-1.0,
                                max_value=1.0,
                                value=0.0,
                                step=0.01,
                                key=f"ncd_custom_only_{i}"
                            )
                            ncd_vec_custom.append(ncd_i)
                
                st.markdown("---")
                
                # Simulation parameters
                col1, col2 = st.columns(2)
                
                with col1:
                    n_years_stat_custom = st.slider(
                        "Tahun simulasi:",
                        min_value=50,
                        max_value=1000,
                        value=100
                    )
                
                with col2:
                    tol_stat_custom = st.number_input(
                        "Toleransi:",
                        min_value=1e-10,
                        max_value=1e-3,
                        value=1e-6,
                        step=1e-7,
                        format="%.0e"
                    )
                
                st.markdown("---")
                
                # Validation button
                if st.button("🔍 Validasi Matriks", type="secondary", use_container_width=True):
                    with st.expander("📊 Hasil Validasi", expanded=True):
                        check_result = check_stationary_distribution(P_matrix_custom)
                        
                        if check_result and check_result.get('has_stationary_distribution', False):
                            st.session_state.custom_matrix_valid = True
                            st.session_state.P_matrix_custom = P_matrix_custom
                            st.session_state.ncd_vec_custom = ncd_vec_custom
                            st.session_state.n_classes_custom = n_classes_custom
                        else:
                            st.session_state.custom_matrix_valid = False
                
                # Simulation button
                is_valid = getattr(st.session_state, 'custom_matrix_valid', False)
                
                if st.button(
                    "🚀 Hitung Premi Stasioner",
                    type="primary",
                    disabled=not is_valid,
                    use_container_width=True
                ):
                    if hasattr(st.session_state, 'P_matrix_custom'):
                        with st.spinner("⚙️ Menghitung..."):
                            progress_bar = st.progress(0)
                            
                            result_stat_custom = simulate_bms(
                                st.session_state.P_matrix_custom,
                                st.session_state.ncd_vec_custom,
                                st.session_state.n_classes_custom,
                                n_years=n_years_stat_custom,
                                tol=tol_stat_custom,
                                country_name="Custom",
                                suppress_output=False
                            )
                            
                            progress_bar.progress(1.0)
                        
                        # Display results (similar to country system above)
                        st.markdown("---")
                        
                        if not np.isnan(result_stat_custom['pstasioner']):
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Premi Stasioner", f"{result_stat_custom['pstasioner']:.5f}")
                            with col2:
                                st.metric("Tahun Konvergensi", f"{int(result_stat_custom['thn_stabil'])}")
                            with col3:
                                final_tv = result_stat_custom['TV'][int(result_stat_custom['thn_stabil'])-1]
                                st.metric("Total Variation", f"{final_tv:.6g}")
                        
                        # Download
                        output = BytesIO()
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            result_stat_custom['df_conv'].to_excel(writer, index=False, sheet_name='Konvergensi')
                        xlsx = output.getvalue()
                        
                        st.download_button(
                            label="📥 Download Hasil Custom (.xlsx)",
                            data=xlsx,
                            file_name="premi_stasioner_custom.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )

        # ========================================================================
        # TAB 6: SENSITIVITAS
        # ========================================================================
        with tab6:
            st.header("📊 Analisis Sensitivitas")
            
            st.markdown("""
            <div class="info-box">
                <p><strong>Analisis Sensitivitas</strong> membantu memahami bagaimana perubahan parameter 
                mempengaruhi premi stasioner dan distribusi sistem bonus-malus.</p>
            </div>
            """, unsafe_allow_html=True)
            
            sens_type = st.selectbox(
                "Pilih Jenis Analisis:",
                [
                    "📈 Sensitivitas terhadap Parameter α (Alpha)",
                    "📉 Sensitivitas terhadap Parameter τ (Tau)",
                    "🔄 Sensitivitas terhadap Kedua Parameter",
                    "🎯 Sensitivitas terhadap Loss Function",
                    "🌍 Perbandingan Antar Negara"
                ]
            )
            
            st.markdown("---")
            
            # ====================================================================
            # SENSITIVITAS ALPHA
            # ====================================================================
            if "Alpha" in sens_type:
                st.subheader("📈 Analisis Sensitivitas Parameter α (Alpha)")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    alpha_min = st.number_input(
                        "α minimum:",
                        min_value=0.1,
                        max_value=10.0,
                        value=max(0.1, aa * 0.5),
                        step=0.1
                    )
                
                with col2:
                    alpha_max = st.number_input(
                        "α maksimum:",
                        min_value=alpha_min,
                        max_value=20.0,
                        value=min(20.0, aa * 2),
                        step=0.1
                    )
                
                with col3:
                    n_points_alpha = st.slider(
                        "Jumlah titik:",
                        min_value=5,
                        max_value=50,
                        value=20
                    )
                
                country_sens = st.selectbox(
                    "Sistem Negara untuk Analisis:",
                    list(countries.keys()),
                    key="alpha_country"
                )
                
                st.markdown("---")
                
                if st.button("🚀 Jalankan Analisis Alpha", type="primary", use_container_width=True):
                    with st.spinner("⚙️ Menghitung sensitivitas..."):
                        progress_bar = st.progress(0)
                        
                        alpha_range = np.linspace(alpha_min, alpha_max, n_points_alpha)
                        results_alpha = []
                        
                        for idx, alpha_test in enumerate(alpha_range):
                            progress_bar.progress((idx + 1) / n_points_alpha)
                            
                            probs = compute_baseline_probs(alpha_test, tau)
                            P_mat = countries[country_sens]['P_builder'](*probs)
                            ncd_vec = countries[country_sens]['ncd']
                            n_classes = countries[country_sens]['n_classes']
                            
                            result = simulate_bms(
                                P_mat, ncd_vec, n_classes,
                                n_years=100, tol=1e-6,
                                country_name=country_sens,
                                suppress_output=True
                            )
                            
                            results_alpha.append({
                                'Alpha': alpha_test,
                                'Premi_Stasioner': result['pstasioner'],
                                'Tahun_Konvergensi': result['thn_stabil']
                            })
                        
                        progress_bar.progress(1.0)
                    
                    df_alpha = pd.DataFrame(results_alpha)
                    
                    st.success("✅ Analisis selesai!")
                    
                    # Metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Premi Min",
                            f"{df_alpha['Premi_Stasioner'].min():.5f}",
                            delta=f"α = {df_alpha.loc[df_alpha['Premi_Stasioner'].idxmin(), 'Alpha']:.2f}"
                        )
                    
                    with col2:
                        st.metric(
                            "Premi Maks",
                            f"{df_alpha['Premi_Stasioner'].max():.5f}",
                            delta=f"α = {df_alpha.loc[df_alpha['Premi_Stasioner'].idxmax(), 'Alpha']:.2f}"
                        )
                    
                    with col3:
                        range_val = df_alpha['Premi_Stasioner'].max() - df_alpha['Premi_Stasioner'].min()
                        st.metric("Range", f"{range_val:.5f}")
                    
                    st.markdown("---")
                    
                    # Chart
                    fig = make_subplots(specs=[[{"secondary_y": True}]])
                    
                    fig.add_trace(
                        go.Scatter(
                            x=df_alpha['Alpha'],
                            y=df_alpha['Premi_Stasioner'],
                            mode='lines+markers',
                            name='Premi Stasioner',
                            line=dict(color='#4facfe', width=3),
                            marker=dict(size=8)
                        ),
                        secondary_y=False
                    )
                    
                    fig.add_trace(
                        go.Scatter(
                            x=df_alpha['Alpha'],
                            y=df_alpha['Tahun_Konvergensi'],
                            mode='lines+markers',
                            name='Tahun Konvergensi',
                            line=dict(color='#f59e0b', width=2, dash='dash'),
                            marker=dict(size=6)
                        ),
                        secondary_y=True
                    )
                    
                    fig.update_xaxes(title_text="Parameter α (Alpha)")
                    fig.update_yaxes(title_text="Premi Stasioner", secondary_y=False)
                    fig.update_yaxes(title_text="Tahun Konvergensi", secondary_y=True)
                    
                    fig.update_layout(
                        title=f"Sensitivitas α - Sistem {country_sens}",
                        plot_bgcolor='#0a0e27',
                        paper_bgcolor='#0a0e27',
                        font=dict(color='white'),
                        hovermode='x unified',
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Table
                    st.subheader("📊 Data Sensitivitas")
                    st.dataframe(
                        df_alpha.style.format({
                            'Alpha': '{:.4f}',
                            'Premi_Stasioner': '{:.5f}',
                            'Tahun_Konvergensi': '{:.0f}'
                        }).background_gradient(subset=['Premi_Stasioner'], cmap='RdYlGn_r'),
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Download
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        df_alpha.to_excel(writer, index=False, sheet_name='Sensitivitas_Alpha')
                    xlsx = output.getvalue()
                    
                    st.download_button(
                        label="📥 Download Hasil (.xlsx)",
                        data=xlsx,
                        file_name=f"sensitivitas_alpha_{country_sens}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
            
            # ====================================================================
            # SENSITIVITAS TAU
            # ====================================================================
            elif "Tau" in sens_type:
                st.subheader("📉 Analisis Sensitivitas Parameter τ (Tau)")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    tau_min = st.number_input(
                        "τ minimum:",
                        min_value=0.1,
                        max_value=10.0,
                        value=max(0.1, tau * 0.5),
                        step=0.1
                    )
                
                with col2:
                    tau_max = st.number_input(
                        "τ maksimum:",
                        min_value=tau_min,
                        max_value=20.0,
                        value=min(20.0, tau * 2),
                        step=0.1
                    )
                
                with col3:
                    n_points_tau = st.slider(
                        "Jumlah titik:",
                        min_value=5,
                        max_value=50,
                        value=20
                    )
                
                country_sens_tau = st.selectbox(
                    "Sistem Negara untuk Analisis:",
                    list(countries.keys()),
                    key="tau_country"
                )
                
                st.markdown("---")
                
                if st.button("🚀 Jalankan Analisis Tau", type="primary", use_container_width=True):
                    with st.spinner("⚙️ Menghitung sensitivitas..."):
                        progress_bar = st.progress(0)
                        
                        tau_range = np.linspace(tau_min, tau_max, n_points_tau)
                        results_tau = []
                        
                        for idx, tau_test in enumerate(tau_range):
                            progress_bar.progress((idx + 1) / n_points_tau)
                            
                            probs = compute_baseline_probs(aa, tau_test)
                            P_mat = countries[country_sens_tau]['P_builder'](*probs)
                            ncd_vec = countries[country_sens_tau]['ncd']
                            n_classes = countries[country_sens_tau]['n_classes']
                            
                            result = simulate_bms(
                                P_mat, ncd_vec, n_classes,
                                n_years=100, tol=1e-6,
                                country_name=country_sens_tau,
                                suppress_output=True
                            )
                            
                            results_tau.append({
                                'Tau': tau_test,
                                'Premi_Stasioner': result['pstasioner'],
                                'Tahun_Konvergensi': result['thn_stabil']
                            })
                        
                        progress_bar.progress(1.0)
                    
                    df_tau = pd.DataFrame(results_tau)
                    
                    st.success("✅ Analisis selesai!")
                    
                    # Metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Premi Min",
                            f"{df_tau['Premi_Stasioner'].min():.5f}",
                            delta=f"τ = {df_tau.loc[df_tau['Premi_Stasioner'].idxmin(), 'Tau']:.2f}"
                        )
                    
                    with col2:
                        st.metric(
                            "Premi Maks",
                            f"{df_tau['Premi_Stasioner'].max():.5f}",
                            delta=f"τ = {df_tau.loc[df_tau['Premi_Stasioner'].idxmax(), 'Tau']:.2f}"
                        )
                    
                    with col3:
                        range_val = df_tau['Premi_Stasioner'].max() - df_tau['Premi_Stasioner'].min()
                        st.metric("Range", f"{range_val:.5f}")
                    
                    st.markdown("---")
                    
                    # Chart
                    fig = make_subplots(specs=[[{"secondary_y": True}]])
                    
                    fig.add_trace(
                        go.Scatter(
                            x=df_tau['Tau'],
                            y=df_tau['Premi_Stasioner'],
                            mode='lines+markers',
                            name='Premi Stasioner',
                            line=dict(color='#4facfe', width=3),
                            marker=dict(size=8)
                        ),
                        secondary_y=False
                    )
                    
                    fig.add_trace(
                        go.Scatter(
                            x=df_tau['Tau'],
                            y=df_tau['Tahun_Konvergensi'],
                            mode='lines+markers',
                            name='Tahun Konvergensi',
                            line=dict(color='#f59e0b', width=2, dash='dash'),
                            marker=dict(size=6)
                        ),
                        secondary_y=True
                    )
                    
                    fig.update_xaxes(title_text="Parameter τ (Tau)")
                    fig.update_yaxes(title_text="Premi Stasioner", secondary_y=False)
                    fig.update_yaxes(title_text="Tahun Konvergensi", secondary_y=True)
                    
                    fig.update_layout(
                        title=f"Sensitivitas τ - Sistem {country_sens_tau}",
                        plot_bgcolor='#0a0e27',
                        paper_bgcolor='#0a0e27',
                        font=dict(color='white'),
                        hovermode='x unified',
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Table
                    st.subheader("📊 Data Sensitivitas")
                    st.dataframe(
                        df_tau.style.format({
                            'Tau': '{:.4f}',
                            'Premi_Stasioner': '{:.5f}',
                            'Tahun_Konvergensi': '{:.0f}'
                        }).background_gradient(subset=['Premi_Stasioner'], cmap='RdYlGn_r'),
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Download
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        df_tau.to_excel(writer, index=False, sheet_name='Sensitivitas_Tau')
                    xlsx = output.getvalue()
                    
                    st.download_button(
                        label="📥 Download Hasil (.xlsx)",
                        data=xlsx,
                        file_name=f"sensitivitas_tau_{country_sens_tau}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
            
            # ====================================================================
            # SENSITIVITAS KEDUA PARAMETER
            # ====================================================================
            elif "Kedua" in sens_type:
                st.subheader("🔄 Analisis Sensitivitas α dan τ (3D Surface)")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Parameter α (Alpha)**")
                    alpha_min_2d = st.number_input(
                        "α minimum:",
                        min_value=0.1,
                        max_value=10.0,
                        value=max(0.1, aa * 0.7),
                        step=0.1,
                        key="alpha_min_2d"
                    )
                    alpha_max_2d = st.number_input(
                        "α maksimum:",
                        min_value=alpha_min_2d,
                        max_value=20.0,
                        value=min(20.0, aa * 1.5),
                        step=0.1,
                        key="alpha_max_2d"
                    )
                    n_alpha_2d = st.slider(
                        "Jumlah titik α:",
                        min_value=5,
                        max_value=20,
                        value=10,
                        key="n_alpha_2d"
                    )
                
                with col2:
                    st.markdown("**Parameter τ (Tau)**")
                    tau_min_2d = st.number_input(
                        "τ minimum:",
                        min_value=0.1,
                        max_value=15.0,
                        value=max(0.1, tau * 0.7),
                        step=0.1,
                        key="tau_min_2d"
                    )
                    tau_max_2d = st.number_input(
                        "τ maksimum:",
                        min_value=tau_min_2d,
                        max_value=100.0,
                        value=min(20.0, tau * 1.5),
                        step=0.1,
                        key="tau_max_2d"
                    )
                    n_tau_2d = st.slider(
                        "Jumlah titik τ:",
                        min_value=5,
                        max_value=20,
                        value=10,
                        key="n_tau_2d"
                    )
                
                country_sens_2d = st.selectbox(
                    "Sistem Negara untuk Analisis:",
                    list(countries.keys()),
                    key="2d_country"
                )
                
                total_iterations = n_alpha_2d * n_tau_2d
                st.info(f"⚙️ Total kombinasi: {total_iterations} iterasi")
                
                st.markdown("---")
                
                if st.button("🚀 Jalankan Analisis 2D", type="primary", use_container_width=True):
                    with st.spinner("⚙️ Menghitung surface plot..."):
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        alpha_range_2d = np.linspace(alpha_min_2d, alpha_max_2d, n_alpha_2d)
                        tau_range_2d = np.linspace(tau_min_2d, tau_max_2d, n_tau_2d)
                        
                        results_2d = []
                        current_iter = 0
                        
                        for alpha_val in alpha_range_2d:
                            for tau_val in tau_range_2d:
                                current_iter += 1
                                status_text.text(f"⚙️ Progress: {current_iter}/{total_iterations} (α={alpha_val:.2f}, τ={tau_val:.2f})")
                                progress_bar.progress(current_iter / total_iterations)
                                
                                probs = compute_baseline_probs(alpha_val, tau_val)
                                P_mat = countries[country_sens_2d]['P_builder'](*probs)
                                ncd_vec = countries[country_sens_2d]['ncd']
                                n_classes = countries[country_sens_2d]['n_classes']
                                
                                result = simulate_bms(
                                    P_mat, ncd_vec, n_classes,
                                    n_years=100, tol=1e-6,
                                    country_name=country_sens_2d,
                                    suppress_output=True
                                )
                                
                                results_2d.append({
                                    'Alpha': alpha_val,
                                    'Tau': tau_val,
                                    'Premi_Stasioner': result['pstasioner'],
                                    'Tahun_Konvergensi': result['thn_stabil']
                                })
                        
                        progress_bar.progress(1.0)
                        status_text.text("✅ Perhitungan selesai!")
                    
                    df_2d = pd.DataFrame(results_2d)
                    
                    # Create pivot table for surface plot
                    pivot_premi = df_2d.pivot(index='Alpha', columns='Tau', values='Premi_Stasioner')
                    
                    # 3D Surface Plot
                    fig_3d = go.Figure(data=[go.Surface(
                        x=pivot_premi.columns,
                        y=pivot_premi.index,
                        z=pivot_premi.values,
                        colorscale='Viridis',
                        colorbar=dict(title="Premi Stasioner")
                    )])
                    
                    fig_3d.update_layout(
                        title=f"Surface Plot Sensitivitas - {country_sens_2d}",
                        scene=dict(
                            xaxis_title='τ (Tau)',
                            yaxis_title='α (Alpha)',
                            zaxis_title='Premi Stasioner',
                            bgcolor='#0a0e27'
                        ),
                        paper_bgcolor='#0a0e27',
                        font=dict(color='white'),
                        height=600
                    )
                    
                    st.plotly_chart(fig_3d, use_container_width=True)
                    
                    st.markdown("---")
                    
                    # Contour plot
                    fig_contour = go.Figure(data=go.Contour(
                        x=pivot_premi.columns,
                        y=pivot_premi.index,
                        z=pivot_premi.values,
                        colorscale='Viridis',
                        colorbar=dict(title="Premi Stasioner"),
                        contours=dict(
                            showlabels=True,
                            labelfont=dict(size=10, color='white')
                        )
                    ))
                    
                    fig_contour.update_layout(
                        title="Contour Plot Sensitivitas",
                        xaxis_title='τ (Tau)',
                        yaxis_title='α (Alpha)',
                        plot_bgcolor='#0a0e27',
                        paper_bgcolor='#0a0e27',
                        font=dict(color='white'),
                        height=500
                    )
                    
                    st.plotly_chart(fig_contour, use_container_width=True)
                    
                    # Statistics
                    st.markdown("---")
                    st.subheader("📊 Statistik Hasil")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Premi Min", f"{df_2d['Premi_Stasioner'].min():.5f}")
                    with col2:
                        st.metric("Premi Maks", f"{df_2d['Premi_Stasioner'].max():.5f}")
                    with col3:
                        st.metric("Premi Rata-rata", f"{df_2d['Premi_Stasioner'].mean():.5f}")
                    with col4:
                        st.metric("Std Dev", f"{df_2d['Premi_Stasioner'].std():.5f}")
                    
                    # Table
                    st.subheader("📋 Data Lengkap")
                    st.dataframe(
                        df_2d.style.format({
                            'Alpha': '{:.4f}',
                            'Tau': '{:.4f}',
                            'Premi_Stasioner': '{:.5f}',
                            'Tahun_Konvergensi': '{:.0f}'
                        }),
                        use_container_width=True,
                        hide_index=True,
                        height=400
                    )
                    
                    # Download
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        df_2d.to_excel(writer, index=False, sheet_name='Sensitivitas_2D')
                        pivot_premi.to_excel(writer, sheet_name='Pivot_Premi')
                    xlsx = output.getvalue()
                    
                    st.download_button(
                        label="📥 Download Hasil 2D (.xlsx)",
                        data=xlsx,
                        file_name=f"sensitivitas_2d_{country_sens_2d}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
            
            # ====================================================================
            # SENSITIVITAS LOSS FUNCTION
            # ====================================================================
            elif "Loss Function" in sens_type:
                st.subheader("🎯 Perbandingan Loss Functions")
                
                country_loss = st.selectbox(
                    "Sistem Negara untuk Analisis:",
                    list(countries.keys()),
                    key="loss_country"
                )
                
                st.markdown("---")
                
                if st.button("🚀 Bandingkan Loss Functions", type="primary", use_container_width=True):
                    with st.spinner("⚙️ Menghitung untuk ketiga loss functions..."):
                        progress_bar = st.progress(0)
                        
                        loss_functions = ["Squared-Error Loss", "Absolute Loss Function", "Fourth-Degree Loss"]
                        results_loss = []
                        
                        for idx, lf in enumerate(loss_functions):
                            progress_bar.progress((idx + 1) / len(loss_functions))
                            
                            prior_lf = compute_prior(lf, aa, tau)
                            probs = compute_baseline_probs(aa, tau)
                            P_mat = countries[country_loss]['P_builder'](*probs)
                            ncd_vec = countries[country_loss]['ncd']
                            n_classes = countries[country_loss]['n_classes']
                            
                            result = simulate_bms(
                                P_mat, ncd_vec, n_classes,
                                n_years=100, tol=1e-6,
                                country_name=country_loss,
                                suppress_output=True
                            )
                            
                            results_loss.append({
                                'Loss_Function': lf,
                                'Prior_λ0': prior_lf,
                                'Premi_Stasioner': result['pstasioner'],
                                'Tahun_Konvergensi': result['thn_stabil']
                            })
                        
                        progress_bar.progress(1.0)
                    
                    df_loss = pd.DataFrame(results_loss)
                    
                    st.success("✅ Analisis selesai!")
                    
                    # Table comparison
                    st.subheader("📊 Hasil Perbandingan")
                    st.dataframe(
                        df_loss.style.format({
                            'Prior_λ0': '{:.5f}',
                            'Premi_Stasioner': '{:.5f}',
                            'Tahun_Konvergensi': '{:.0f}'
                        }).background_gradient(subset=['Premi_Stasioner'], cmap='RdYlGn_r'),
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Bar chart comparison
                    st.markdown("---")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig1 = go.Figure()
                        fig1.add_trace(go.Bar(
                            x=df_loss['Loss_Function'],
                            y=df_loss['Prior_λ0'],
                            marker_color='#4facfe',
                            text=df_loss['Prior_λ0'].round(5),
                            textposition='auto'
                        ))
                        
                        fig1.update_layout(
                            title="Prior λ₀ per Loss Function",
                            xaxis_title="Loss Function",
                            yaxis_title="Prior λ₀",
                            plot_bgcolor='#0a0e27',
                            paper_bgcolor='#0a0e27',
                            font=dict(color='white'),
                            showlegend=False,
                            height=400
                        )
                        
                        st.plotly_chart(fig1, use_container_width=True)
                    
                    with col2:
                        fig2 = go.Figure()
                        fig2.add_trace(go.Bar(
                            x=df_loss['Loss_Function'],
                            y=df_loss['Premi_Stasioner'],
                            marker_color='#10b981',
                            text=df_loss['Premi_Stasioner'].round(5),
                            textposition='auto'
                        ))
                        
                        fig2.update_layout(
                            title="Premi Stasioner per Loss Function",
                            xaxis_title="Loss Function",
                            yaxis_title="Premi Stasioner",
                            plot_bgcolor='#0a0e27',
                            paper_bgcolor='#0a0e27',
                            font=dict(color='white'),
                            showlegend=False,
                            height=400
                        )
                        
                        st.plotly_chart(fig2, use_container_width=True)
                    
                    # Insights
                    st.markdown("---")
                    st.subheader("💡 Insights")
                    
                    best_loss = df_loss.loc[df_loss['Premi_Stasioner'].idxmin()]
                    worst_loss = df_loss.loc[df_loss['Premi_Stasioner'].idxmax()]
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"""
                        <div class="success-box">
                            <h4>🏆 Premi Terendah</h4>
                            <p><strong>{best_loss['Loss_Function']}</strong></p>
                            <p>Premi: {best_loss['Premi_Stasioner']:.5f}</p>
                            <p>Prior λ₀: {best_loss['Prior_λ0']:.5f}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class="warning-box">
                            <h4>📈 Premi Tertinggi</h4>
                            <p><strong>{worst_loss['Loss_Function']}</strong></p>
                            <p>Premi: {worst_loss['Premi_Stasioner']:.5f}</p>
                            <p>Prior λ₀: {worst_loss['Prior_λ0']:.5f}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    difference = worst_loss['Premi_Stasioner'] - best_loss['Premi_Stasioner']
                    pct_diff = (difference / best_loss['Premi_Stasioner']) * 100
                    
                    st.info(f"📊 Perbedaan premi: {difference:.5f} ({pct_diff:.2f}%)")
                    
                    # Download
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        df_loss.to_excel(writer, index=False, sheet_name='Perbandingan_Loss')
                    xlsx = output.getvalue()
                    
                    st.download_button(
                        label="📥 Download Perbandingan (.xlsx)",
                        data=xlsx,
                        file_name=f"perbandingan_loss_{country_loss}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
            
            # ====================================================================
            # PERBANDINGAN ANTAR NEGARA
            # ====================================================================
            else:  # Perbandingan Antar Negara
                st.subheader("🌍 Perbandingan Sistem Bonus-Malus Antar Negara")
                
                selected_countries = st.multiselect(
                    "Pilih negara untuk dibandingkan:",
                    list(countries.keys()),
                    default=list(countries.keys())[:3]
                )
                
                if len(selected_countries) == 0:
                    st.warning("⚠️ Pilih minimal 1 negara untuk analisis")
                    st.stop()
                
                st.markdown("---")
                
                if st.button("🚀 Bandingkan Negara", type="primary", use_container_width=True):
                    with st.spinner("⚙️ Menghitung untuk setiap negara..."):
                        progress_bar = st.progress(0)
                        
                        results_countries = []
                        
                        for idx, country in enumerate(selected_countries):
                            progress_bar.progress((idx + 1) / len(selected_countries))
                            
                            probs = compute_baseline_probs(aa, tau)
                            P_mat = countries[country]['P_builder'](*probs)
                            ncd_vec = countries[country]['ncd']
                            n_classes = countries[country]['n_classes']
                            
                            result = simulate_bms(
                                P_mat, ncd_vec, n_classes,
                                n_years=100, tol=1e-6,
                                country_name=country,
                                suppress_output=True
                            )
                            
                            results_countries.append({
                                'Negara': country,
                                'Jumlah_Kelas': n_classes,
                                'NCD_Min': min(ncd_vec),
                                'NCD_Maks': max(ncd_vec),
                                'Premi_Stasioner': result['pstasioner'],
                                'Tahun_Konvergensi': result['thn_stabil']
                            })
                        
                        progress_bar.progress(1.0)
                    
                    df_countries = pd.DataFrame(results_countries)
                    
                    st.success("✅ Analisis selesai!")
                    
                    # Summary table
                    st.subheader("📊 Ringkasan Perbandingan")
                    st.dataframe(
                        df_countries.style.format({
                            'NCD_Min': '{:.2f}',
                            'NCD_Maks': '{:.2f}',
                            'Premi_Stasioner': '{:.5f}',
                            'Tahun_Konvergensi': '{:.0f}'
                        }).background_gradient(subset=['Premi_Stasioner'], cmap='RdYlGn_r'),
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    st.markdown("---")
                    
                    # Visualizations
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Bar chart - Premi Stasioner
                        fig1 = go.Figure()
                        fig1.add_trace(go.Bar(
                            x=df_countries['Negara'],
                            y=df_countries['Premi_Stasioner'],
                            marker_color='#4facfe',
                            text=df_countries['Premi_Stasioner'].round(5),
                            textposition='auto'
                        ))
                        
                        fig1.update_layout(
                            title="Premi Stasioner per Negara",
                            xaxis_title="Negara",
                            yaxis_title="Premi Stasioner",
                            plot_bgcolor='#0a0e27',
                            paper_bgcolor='#0a0e27',
                            font=dict(color='white'),
                            showlegend=False,
                            height=400
                        )
                        
                        st.plotly_chart(fig1, use_container_width=True)
                    
                    with col2:
                        # Bar chart - Tahun Konvergensi
                        fig2 = go.Figure()
                        fig2.add_trace(go.Bar(
                            x=df_countries['Negara'],
                            y=df_countries['Tahun_Konvergensi'],
                            marker_color='#f59e0b',
                            text=df_countries['Tahun_Konvergensi'].round(0),
                            textposition='auto'
                        ))
                        
                        fig2.update_layout(
                            title="Tahun Konvergensi per Negara",
                            xaxis_title="Negara",
                            yaxis_title="Tahun",
                            plot_bgcolor='#0a0e27',
                            paper_bgcolor='#0a0e27',
                            font=dict(color='white'),
                            showlegend=False,
                            height=400
                        )
                        
                        st.plotly_chart(fig2, use_container_width=True)
                    
                    # Scatter plot - Jumlah Kelas vs Premi
                    st.markdown("---")
                    
                    fig3 = go.Figure()
                    fig3.add_trace(go.Scatter(
                        x=df_countries['Jumlah_Kelas'],
                        y=df_countries['Premi_Stasioner'],
                        mode='markers+text',
                        marker=dict(size=15, color='#10b981'),
                        text=df_countries['Negara'],
                        textposition='top center'
                    ))
                    
                    fig3.update_layout(
                        title="Hubungan Jumlah Kelas vs Premi Stasioner",
                        xaxis_title="Jumlah Kelas",
                        yaxis_title="Premi Stasioner",
                        plot_bgcolor='#0a0e27',
                        paper_bgcolor='#0a0e27',
                        font=dict(color='white'),
                        showlegend=False,
                        height=500
                    )
                    
                    st.plotly_chart(fig3, use_container_width=True)
                    
                    # Rankings
                    st.markdown("---")
                    st.subheader("🏆 Peringkat Sistem")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Berdasarkan Premi Stasioner (Terendah)**")
                        df_ranked_premi = df_countries.sort_values('Premi_Stasioner')[['Negara', 'Premi_Stasioner']].reset_index(drop=True)
                        df_ranked_premi.index += 1
                        st.dataframe(
                            df_ranked_premi.style.format({'Premi_Stasioner': '{:.5f}'}),
                            use_container_width=True
                        )
                    
                    with col2:
                        st.markdown("**Berdasarkan Kecepatan Konvergensi**")
                        df_ranked_conv = df_countries.sort_values('Tahun_Konvergensi')[['Negara', 'Tahun_Konvergensi']].reset_index(drop=True)
                        df_ranked_conv.index += 1
                        st.dataframe(
                            df_ranked_conv.style.format({'Tahun_Konvergensi': '{:.0f}'}),
                            use_container_width=True
                        )
                    
                    # Download
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        df_countries.to_excel(writer, index=False, sheet_name='Perbandingan_Negara')
                        df_ranked_premi.to_excel(writer, sheet_name='Ranking_Premi')
                        df_ranked_conv.to_excel(writer, sheet_name='Ranking_Konvergensi')
                    xlsx = output.getvalue()
                    
                    st.download_button(
                        label="📥 Download Perbandingan Negara (.xlsx)",
                        data=xlsx,
                        file_name="perbandingan_antar_negara.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.exception(e)

else:
    st.info("""
    👋 **Selamat Datang di Sistem Bonus-Malus Ideal!**
    
    Untuk memulai, silakan upload file data klaim Anda melalui sidebar di sebelah kiri.
    
    **Format file yang didukung:**
    - 📄 CSV (.csv)
    - 📊 Excel (.xlsx)
    
    Sistem akan secara otomatis menghitung parameter distribusi dan memberikan berbagai analisis bonus-malus.
    """)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 2rem 0; color: #94a3b8;'>
    <p><strong>Sistem Bonus-Malus Ideal</strong></p>
    <p style='font-size: 0.9rem;'>© 2025 Universitas Gadjah Mada | Ilmu Aktuaria | Revaldy Hazza Daniswara</p>
    <p style='font-size: 0.8rem; margin-top: 1rem;'>
        💡 Powered by Streamlit | 🐍 Python
    </p>
</div>
""", unsafe_allow_html=True)