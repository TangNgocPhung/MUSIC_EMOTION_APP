"""
================================================================================
HE THONG PHAN TICH DIEN BIEN CAM XUC TRONG AM NHAC
Streamlit Demo App — Phien ban Master Thesis
================================================================================
Tac gia : Phung
Model   : CNN + BiLSTM + Multi-head Attention (Combined Loss + Class Weighting)
Dataset : DEAM + PMEmo (cross-cultural)
================================================================================
"""

import os
import json
import io
import tempfile
import traceback
from pathlib import Path
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd
import librosa
import librosa.display
import torch
import torch.nn as nn
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# =============================================================================
# 1. CONFIGURATION
# =============================================================================
SR                 = 22050
CLIP_START_SEC     = 15.0
CLIP_END_SEC       = 45.0
WINDOW_SEC         = 0.5
N_STEPS            = int((CLIP_END_SEC - CLIP_START_SEC) / WINDOW_SEC)
SAMPLES_PER_WINDOW = int(SR * WINDOW_SEC)
N_FFT, HOP_LENGTH, N_MELS = 1024, 256, 64
FMIN, FMAX = 20, SR // 2
CNN_OUT_DIM, LSTM_HIDDEN, LSTM_LAYERS, LSTM_BIDIR, DROPOUT = 256, 128, 2, True, 0.3
LABEL_MIN, LABEL_MAX, LABEL_MEAN = -1.0, 1.0, 0.0

CKPT_DIR = Path(".")
AVAILABLE_CHECKPOINTS = {
    "Attention + Balanced (BEST)": "best_attention_balanced.pt",
    "Attention (CCC Loss)":         "best_attention.pt",
    "CCC Loss":                      "best_ccc.pt",
    "Baseline (SmoothL1)":           "best.pt",
}

QUADRANTS = {(+1,+1):"happy/excited", (-1,+1):"tense/angry",
              (-1,-1):"sad",          (+1,-1):"calm/relaxed"}
MOOD_COLORS = {"happy/excited":"#f39c12", "tense/angry":"#e74c3c",
                "sad":"#3498db",          "calm/relaxed":"#2ecc71"}
MOOD_EMOJIS = {"happy/excited":"😊", "tense/angry":"😠",
                "sad":"😢",          "calm/relaxed":"😌"}
MOOD_VI = {"happy/excited":"Vui ve / Hung phan",
            "tense/angry":  "Cang thang / Tuc gian",
            "sad":           "Buon ba",
            "calm/relaxed":  "Thu thai / Binh yen"}


# =============================================================================
# 2. MODEL ARCHITECTURE
# =============================================================================
class CNNEncoder(nn.Module):
    def __init__(self, out_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),  nn.BatchNorm2d(32),  nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),nn.BatchNorm2d(128), nn.ReLU(True),
            nn.AdaptiveAvgPool2d((4, 4)))
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(128*4*4, out_dim),
                                 nn.ReLU(True), nn.Dropout(DROPOUT))
    def forward(self, x): return self.fc(self.conv(x))


class CNNLSTMEmotion(nn.Module):
    """Baseline: CNN + BiLSTM (cho best.pt, best_ccc.pt)"""
    def __init__(self):
        super().__init__()
        self.cnn = CNNEncoder(CNN_OUT_DIM)
        self.lstm = nn.LSTM(CNN_OUT_DIM, LSTM_HIDDEN, LSTM_LAYERS, batch_first=True,
                             bidirectional=LSTM_BIDIR, dropout=DROPOUT if LSTM_LAYERS>1 else 0.0)
        h = LSTM_HIDDEN * (2 if LSTM_BIDIR else 1)
        self.head = nn.Sequential(nn.Linear(h, 64), nn.ReLU(True),
                                   nn.Dropout(DROPOUT), nn.Linear(64, 2), nn.Tanh())
    def forward(self, x):
        B, T, C, M, Fr = x.shape
        f = self.cnn(x.view(B*T, C, M, Fr)).view(B, T, -1)
        s, _ = self.lstm(f)
        return self.head(s)


class CNNLSTMAttentionEmotion(nn.Module):
    """Full: CNN + BiLSTM + Multi-head Attention (cho best_attention*.pt)"""
    def __init__(self):
        super().__init__()
        self.cnn  = CNNEncoder(CNN_OUT_DIM)
        self.lstm = nn.LSTM(CNN_OUT_DIM, LSTM_HIDDEN, LSTM_LAYERS,
                             batch_first=True, bidirectional=LSTM_BIDIR,
                             dropout=DROPOUT if LSTM_LAYERS>1 else 0.0)
        h = LSTM_HIDDEN * (2 if LSTM_BIDIR else 1)
        self.attn = nn.MultiheadAttention(embed_dim=h, num_heads=4,
                                           dropout=0.1, batch_first=True)
        self.norm = nn.LayerNorm(h)
        self.head = nn.Sequential(nn.Linear(h, 64), nn.ReLU(True),
                                   nn.Dropout(DROPOUT), nn.Linear(64, 2), nn.Tanh())
    def forward(self, x):
        B, T, C, M, Fr = x.shape
        f = self.cnn(x.view(B*T, C, M, Fr)).view(B, T, -1)
        s, _ = self.lstm(f)
        a, _ = self.attn(s, s, s)
        s = self.norm(s + a)
        return self.head(s)


# =============================================================================
# 3. AUDIO + INFERENCE FUNCTIONS
# =============================================================================
def audio_to_mel_sequence(y):
    s, e = int(CLIP_START_SEC*SR), int(CLIP_END_SEC*SR)
    if len(y) < e:
        y = np.pad(y, (0, e - len(y)))
    y = y[s:e]
    mels = []
    for i in range(N_STEPS):
        chunk = y[i*SAMPLES_PER_WINDOW:(i+1)*SAMPLES_PER_WINDOW]
        m = librosa.feature.melspectrogram(y=chunk, sr=SR, n_fft=N_FFT,
            hop_length=HOP_LENGTH, n_mels=N_MELS, fmin=FMIN, fmax=FMAX, power=2.0)
        mels.append(librosa.power_to_db(m, ref=np.max))
    return np.stack(mels, 0)[:, None, :, :].astype(np.float32)


def quadrant(v, a):
    return QUADRANTS[(+1 if v >= LABEL_MEAN else -1, +1 if a >= LABEL_MEAN else -1)]


def group_timeline(times, moods, min_len=2.0):
    if len(moods) == 0:
        return []
    segs = []; cm = moods[0]; cs = times[0]
    for i in range(1, len(moods)):
        if moods[i] != cm:
            e = times[i]
            if e - cs >= min_len or not segs:
                segs.append({"start": float(cs), "end": float(e), "mood": cm})
            else:
                segs[-1]["end"] = float(e)
            cm = moods[i]; cs = times[i]
    segs.append({"start": float(cs), "end": float(times[-1] + WINDOW_SEC), "mood": cm})
    return segs


def smooth_predictions(arr, window=5):
    if window <= 1:
        return arr
    kernel = np.ones(window) / window
    smoothed = np.zeros_like(arr)
    for i in range(arr.shape[1]):
        smoothed[:, i] = np.convolve(arr[:, i], kernel, mode='same')
    return smoothed


@st.cache_resource
def load_model(ckpt_name):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if "attention" in ckpt_name.lower():
        m = CNNLSTMAttentionEmotion().to(dev)
    else:
        m = CNNLSTMEmotion().to(dev)
    ckpt = torch.load(ckpt_name, map_location=dev, weights_only=False)
    m.load_state_dict(ckpt["model"])
    m.eval()
    return m, dev


@torch.no_grad()
def predict(audio_path, model, dev, smoothing=5):
    y, _ = librosa.load(audio_path, sr=SR, mono=True)
    total = len(y) / SR

    if total < CLIP_END_SEC:
        mel = audio_to_mel_sequence(y)
        pr = model(torch.from_numpy(mel[None]).to(dev)).cpu().numpy()[0]
        times = np.arange(N_STEPS) * WINDOW_SEC + CLIP_START_SEC
    else:
        nwin = int(total / WINDOW_SEC)
        acc = np.zeros((nwin, 2), np.float32); cnt = np.zeros(nwin, np.int32)
        L = CLIP_END_SEC - CLIP_START_SEC; stride = L / 2; start = 0.0
        while start + L <= total:
            yc = y[int(start*SR):int((start+L)*SR)]
            yp = np.concatenate([np.zeros(int(CLIP_START_SEC*SR)), yc])
            mel = audio_to_mel_sequence(yp)
            pr_ = model(torch.from_numpy(mel[None]).to(dev)).cpu().numpy()[0]
            idx0 = int(start / WINDOW_SEC)
            for i in range(N_STEPS):
                k = idx0 + i
                if k < nwin:
                    acc[k] += pr_[i]; cnt[k] += 1
            start += stride
        valid_mask = cnt > 0
        pr = acc[valid_mask] / cnt[valid_mask, None]
        times = np.arange(len(pr)) * WINDOW_SEC

    pr = smooth_predictions(pr, smoothing)
    return y, total, times, pr


# =============================================================================
# 4. STREAMLIT UI — CUSTOM CSS + PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="Music Emotion Analyzer",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* ============ LIGHT THEME — Pastel gradient ============ */
    .stApp {
        background: linear-gradient(135deg, #fdfbfb 0%, #ebedee 50%, #f5e6f7 100%);
    }

    /* Header — gradient màu vẫn giữ nổi bật */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        padding: 2.5rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 15px 40px rgba(118, 75, 162, 0.25);
    }
    .main-header h1 {
        color: white;
        text-align: center;
        font-size: 2.8rem;
        margin: 0;
        text-shadow: 2px 3px 8px rgba(0,0,0,0.2);
        font-weight: 700;
    }
    .main-header p {
        color: rgba(255,255,255,0.95);
        text-align: center;
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
    }

    /* Metric cards — trắng với shadow nhẹ */
    .metric-card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid rgba(118, 75, 162, 0.08);
        box-shadow: 0 4px 15px rgba(118, 75, 162, 0.08);
        text-align: center;
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 30px rgba(118, 75, 162, 0.2);
        border-color: rgba(118, 75, 162, 0.2);
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0.5rem 0;
    }
    .metric-label {
        color: #5a6c7d;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        font-weight: 600;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: white;
        padding: 10px;
        border-radius: 14px;
        box-shadow: 0 4px 15px rgba(118, 75, 162, 0.08);
    }
    .stTabs [data-baseweb="tab"] {
        background: #f7f7fb;
        border-radius: 10px;
        color: #5a6c7d;
        padding: 12px 22px;
        font-weight: 600;
        border: none;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: #eef0ff;
        color: #667eea;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: white !important;
        box-shadow: 0 4px 15px rgba(118, 75, 162, 0.3);
    }

    /* Sidebar — light với accent màu */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f5f0ff 100%);
        border-right: 1px solid rgba(118, 75, 162, 0.1);
    }
    [data-testid="stSidebar"] * {
        color: #2c3e50;
    }
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4 {
        color: #667eea !important;
    }

    /* Buttons */
    .stButton button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.6rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(118, 75, 162, 0.25);
    }
    .stButton button:hover {
        transform: scale(1.05);
        box-shadow: 0 8px 25px rgba(118, 75, 162, 0.4);
    }
    .stDownloadButton button {
        background: linear-gradient(135deg, #43e97b, #38f9d7);
        color: white;
        border-radius: 25px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(67, 233, 123, 0.3);
    }
    .stDownloadButton button:hover {
        transform: scale(1.03);
        box-shadow: 0 8px 20px rgba(67, 233, 123, 0.45);
    }

    /* Insight box — trắng với accent gradient */
    .insight-box {
        background: white;
        border-left: 5px solid;
        border-image: linear-gradient(180deg, #667eea, #f093fb) 1;
        padding: 1.5rem 1.8rem;
        border-radius: 12px;
        margin: 1rem 0;
        color: #2c3e50;
        font-size: 1.05rem;
        line-height: 1.7;
        box-shadow: 0 4px 15px rgba(118, 75, 162, 0.08);
    }

    /* File uploader */
    [data-testid="stFileUploaderDropzone"] {
        background: white;
        border: 2px dashed #667eea;
        border-radius: 14px;
        padding: 1.5rem;
    }

    /* Headers main area */
    h1, h2, h3, h4, h5 {
        color: #2c3e50 !important;
    }

    /* Slider */
    .stSlider [data-baseweb="slider"] [role="slider"] {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
    }

    /* Selectbox */
    .stSelectbox [data-baseweb="select"] {
        background: white;
        border-radius: 10px;
    }

    /* Caption text */
    .stCaption {
        color: #6c7a89 !important;
    }

    /* Dataframe */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(118, 75, 162, 0.08);
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #6c7a89;
        padding: 2rem;
        margin-top: 3rem;
        border-top: 1px solid rgba(118, 75, 162, 0.1);
        font-size: 0.95rem;
    }

    /* Audio player */
    audio {
        width: 100%;
        border-radius: 10px;
    }

    /* Info / success / error boxes */
    .stAlert {
        border-radius: 12px;
        border: none;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
    }

    /* Spinner */
    .stSpinner > div {
        border-top-color: #667eea !important;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# 5. SESSION STATE INIT
# =============================================================================
if "history" not in st.session_state:
    st.session_state.history = []
if "current_result" not in st.session_state:
    st.session_state.current_result = None


# =============================================================================
# 6. HEADER
# =============================================================================
st.markdown("""
<div class="main-header">
    <h1>🎵 Music Emotion Analyzer</h1>
    <p>Phan tich dien bien cam xuc theo thoi gian | CNN + BiLSTM + Attention | DEAM + PMEmo</p>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# 7. SIDEBAR — CONTROLS
# =============================================================================
with st.sidebar:
    st.markdown("### ⚙️ Cau hinh phan tich")

    available = {k: v for k, v in AVAILABLE_CHECKPOINTS.items()
                 if (CKPT_DIR / v).exists()}
    if not available:
        st.error("Khong tim thay checkpoint nao!")
        st.stop()

    selected_model = st.selectbox(
        "🤖 Chon model",
        list(available.keys()),
        help="Model tot nhat: 'Attention + Balanced'"
    )
    ckpt_file = available[selected_model]

    st.markdown("---")
    st.markdown("### 🎛️ Tham so")

    smoothing = st.slider("Do muot du doan", 1, 15, 5,
                          help="Cua so moving average — tang de giam nhieu")
    min_seg_len = st.slider("Do dai toi thieu doan (s)", 1.0, 10.0, 3.0, 0.5,
                            help="Doan ngan hon se duoc gop vao doan truoc")

    st.markdown("---")
    st.markdown("### 📊 Thong tin model")

    model_info = {
        "Attention + Balanced (BEST)": ("CNN + BiLSTM + Attention", "Combined Loss + Class Weighting"),
        "Attention (CCC Loss)": ("CNN + BiLSTM + Attention", "CCC Loss"),
        "CCC Loss": ("CNN + BiLSTM", "CCC Loss"),
        "Baseline (SmoothL1)": ("CNN + BiLSTM", "SmoothL1 Loss"),
    }
    arch, loss = model_info.get(selected_model, ("?", "?"))
    st.markdown(f"""
    - **Kien truc**: `{arch}`
    - **Loss function**: `{loss}`
    - **Thong so**: ~1.7M params
    - **Dataset**: DEAM + PMEmo
    """)

    st.markdown("---")
    if st.button("🗑️ Xoa lich su"):
        st.session_state.history = []
        st.session_state.current_result = None
        st.rerun()

    st.markdown("---")
    st.markdown("""
    <div style='text-align:center; color:#9aa5b1; font-size:0.85rem; padding-top:1rem;'>
        🎓 Master Thesis Demo<br>
        Built with Streamlit + PyTorch
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# 8. MAIN TABS
# =============================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎼 Phan tich bai nhac",
    "⚖️ So sanh 2 bai",
    "🔍 Mel-Spectrogram Explorer",
    "📚 Lich su phan tich",
    "ℹ️ Gioi thieu",
])


# =============================================================================
# TAB 1 — SINGLE SONG ANALYSIS
# =============================================================================
with tab1:
    st.markdown("### 📤 Upload bai nhac")

    col_u1, col_u2 = st.columns([3, 1])
    with col_u1:
        up = st.file_uploader("Chon file MP3 / WAV", type=["mp3", "wav"],
                              key="single_upload", label_visibility="collapsed")
    with col_u2:
        analyze_btn = st.button("🚀 Phan tich", use_container_width=True, type="primary")

    if up is not None and analyze_btn:
        tmp = Path(tempfile.gettempdir()) / up.name
        tmp.write_bytes(up.read())

        try:
            with st.spinner("🎵 Dang trich xuat dac trung..."):
                model, dev = load_model(ckpt_file)
                y, total, times, pr = predict(tmp, model, dev, smoothing)

            moods = [quadrant(v, a) for v, a in pr]
            segs = group_timeline(times, moods, min_seg_len)
            mv, ma = float(pr[:, 0].mean()), float(pr[:, 1].mean())
            dom_mood = quadrant(mv, ma)

            result = {
                "filename": up.name,
                "duration": total,
                "times": times.tolist(),
                "valence": pr[:, 0].tolist(),
                "arousal": pr[:, 1].tolist(),
                "moods": moods,
                "segments": segs,
                "avg_v": mv, "avg_a": ma,
                "dominant_mood": dom_mood,
                "model": selected_model,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "audio_path": str(tmp),
            }
            st.session_state.current_result = result
            st.session_state.history.append(result)

        except Exception as e:
            st.error(f"❌ Loi: {str(e)}")
            with st.expander("Chi tiet loi"):
                st.code(traceback.format_exc())

    if st.session_state.current_result:
        r = st.session_state.current_result
        st.audio(r["audio_path"])

        st.markdown("### 📈 Tong quan")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Thoi luong</div>
                <div class="metric-value">{r['duration']:.1f}s</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Valence trung binh</div>
                <div class="metric-value">{r['avg_v']:+.2f}</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Arousal trung binh</div>
                <div class="metric-value">{r['avg_a']:+.2f}</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            color = MOOD_COLORS[r['dominant_mood']]
            emoji = MOOD_EMOJIS[r['dominant_mood']]
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Cam xuc chu dao</div>
                <div style="font-size:2.2rem; margin:0.3rem 0;">{emoji}</div>
                <div style="color:{color}; font-weight:700; font-size:1.05rem;">
                    {MOOD_VI[r['dominant_mood']]}
                </div>
            </div>""", unsafe_allow_html=True)

        unique_moods = list(set(r['moods']))
        if len(unique_moods) > 1:
            mood_seq = " → ".join([MOOD_VI[s['mood']] for s in r['segments']])
            insight_text = (
                f"<b>Bai nhac trai qua {len(r['segments'])} doan cam xuc:</b> {mood_seq}.<br>"
                f"<b>Cam xuc trung binh</b>: {MOOD_VI[r['dominant_mood']]} (V={r['avg_v']:+.2f}, A={r['avg_a']:+.2f})."
            )
        else:
            insight_text = (
                f"<b>Cam xuc nhat quan xuyen suot</b>: {MOOD_VI[r['dominant_mood']]} "
                f"(V={r['avg_v']:+.2f}, A={r['avg_a']:+.2f})."
            )
        st.markdown(f'<div class="insight-box">💡 {insight_text}</div>', unsafe_allow_html=True)

        st.markdown("### 📉 Dien bien V-A theo thoi gian")
        times_arr = np.array(r['times'])
        v_arr = np.array(r['valence'])
        a_arr = np.array(r['arousal'])

        fig_timeline = go.Figure()
        fig_timeline.add_trace(go.Scatter(x=times_arr, y=v_arr, name="Valence",
                                           line=dict(color="#4facfe", width=3),
                                           fill='tozeroy', fillcolor='rgba(79,172,254,0.1)'))
        fig_timeline.add_trace(go.Scatter(x=times_arr, y=a_arr, name="Arousal",
                                           line=dict(color="#f5576c", width=3),
                                           fill='tozeroy', fillcolor='rgba(245,87,108,0.1)'))
        fig_timeline.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig_timeline.update_layout(
            template="plotly_white",
            height=400,
            xaxis_title="Thoi gian (giay)",
            yaxis_title="V / A (-1 to 1)",
            yaxis_range=[-1.1, 1.1],
            hovermode='x unified',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0.2)',
        )
        st.plotly_chart(fig_timeline, use_container_width=True)

        col_a, col_b = st.columns([2, 1])

        with col_a:
            st.markdown("#### 🎨 Ban do mood theo thoi gian")
            fig_mood = go.Figure()
            for s in r['segments']:
                color = MOOD_COLORS.get(s['mood'], "#95a5a6")
                fig_mood.add_trace(go.Scatter(
                    x=[s['start'], s['end'], s['end'], s['start'], s['start']],
                    y=[0, 0, 1, 1, 0],
                    fill='toself', fillcolor=color, line=dict(color=color),
                    name=MOOD_VI[s['mood']],
                    text=f"{MOOD_VI[s['mood']]}<br>{s['start']:.1f}s – {s['end']:.1f}s",
                    hoverinfo='text', opacity=0.85,
                    showlegend=False,
                ))
                fig_mood.add_annotation(
                    x=(s['start'] + s['end']) / 2, y=0.5,
                    text=f"<b>{MOOD_EMOJIS[s['mood']]}</b>",
                    showarrow=False, font=dict(size=24),
                )
            fig_mood.update_layout(
                template="plotly_white", height=200,
                xaxis_title="Thoi gian (giay)", yaxis=dict(visible=False),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(t=20, b=40),
            )
            st.plotly_chart(fig_mood, use_container_width=True)

        with col_b:
            st.markdown("#### 🎯 Quy dao V-A")
            fig_va = go.Figure()
            fig_va.add_shape(type="rect", x0=0, y0=0, x1=1, y1=1,
                             fillcolor="rgba(243,156,18,0.1)", line_width=0)
            fig_va.add_shape(type="rect", x0=-1, y0=0, x1=0, y1=1,
                             fillcolor="rgba(231,76,60,0.1)", line_width=0)
            fig_va.add_shape(type="rect", x0=-1, y0=-1, x1=0, y1=0,
                             fillcolor="rgba(52,152,219,0.1)", line_width=0)
            fig_va.add_shape(type="rect", x0=0, y0=-1, x1=1, y1=0,
                             fillcolor="rgba(46,204,113,0.1)", line_width=0)
            fig_va.add_trace(go.Scatter(
                x=v_arr, y=a_arr, mode='lines+markers',
                line=dict(color="#8e44ad", width=2),
                marker=dict(size=5, color=times_arr, colorscale='Viridis',
                            showscale=True, colorbar=dict(title="Time(s)")),
                name="Quy dao",
            ))
            for (vs, as_), name in QUADRANTS.items():
                fig_va.add_annotation(x=vs*0.7, y=as_*0.85,
                                       text=MOOD_EMOJIS[name],
                                       showarrow=False, font=dict(size=24))
            fig_va.add_hline(y=0, line_color="#2c3e50", opacity=0.4)
            fig_va.add_vline(x=0, line_color="#2c3e50", opacity=0.4)
            fig_va.update_layout(
                template="plotly_white", height=400,
                xaxis_title="Valence", yaxis_title="Arousal",
                xaxis_range=[-1, 1], yaxis_range=[-1, 1],
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                showlegend=False,
            )
            st.plotly_chart(fig_va, use_container_width=True)

        col_c, col_d = st.columns([1, 1])

        with col_c:
            st.markdown("#### 🥧 Phan bo mood")
            mood_counts = Counter(r['moods'])
            fig_pie = go.Figure(data=[go.Pie(
                labels=[MOOD_VI[m] for m in mood_counts.keys()],
                values=list(mood_counts.values()),
                marker=dict(colors=[MOOD_COLORS[m] for m in mood_counts.keys()],
                            line=dict(color='white', width=2)),
                hole=0.45,
                textinfo='label+percent',
                textfont=dict(color='white', size=12, family='Arial Black'),
            )])
            fig_pie.update_layout(
                template="plotly_white", height=350,
                paper_bgcolor='rgba(0,0,0,0)',
                showlegend=False,
                margin=dict(t=20, b=20),
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_d:
            st.markdown("#### 📋 Bang doan cam xuc")
            seg_df = pd.DataFrame([
                {
                    "#": i+1,
                    "Tu": f"{s['start']:.1f}s",
                    "Den": f"{s['end']:.1f}s",
                    "Thoi luong": f"{s['end']-s['start']:.1f}s",
                    "Mood": f"{MOOD_EMOJIS[s['mood']]} {MOOD_VI[s['mood']]}",
                }
                for i, s in enumerate(r['segments'])
            ])
            st.dataframe(seg_df, use_container_width=True, hide_index=True, height=350)

        st.markdown("### 💾 Xuat ket qua")
        ec1, ec2, ec3 = st.columns(3)

        with ec1:
            csv_data = pd.DataFrame({
                "time_sec": r['times'],
                "valence":  r['valence'],
                "arousal":  r['arousal'],
                "mood":     r['moods'],
            }).to_csv(index=False)
            st.download_button("📄 Tai CSV (chi tiet)", csv_data,
                                file_name=f"{r['filename']}_emotion.csv",
                                mime="text/csv", use_container_width=True)

        with ec2:
            json_data = json.dumps({
                "filename": r['filename'],
                "model": r['model'],
                "timestamp": r['timestamp'],
                "duration_sec": r['duration'],
                "avg_valence": r['avg_v'],
                "avg_arousal": r['avg_a'],
                "dominant_mood": r['dominant_mood'],
                "segments": r['segments'],
            }, indent=2, ensure_ascii=False)
            st.download_button("📊 Tai JSON (tong hop)", json_data,
                                file_name=f"{r['filename']}_summary.json",
                                mime="application/json", use_container_width=True)

        with ec3:
            report = f"""BAO CAO PHAN TICH CAM XUC AM NHAC
==========================================
File:     {r['filename']}
Model:    {r['model']}
Time:     {r['timestamp']}
Duration: {r['duration']:.1f}s

KET QUA:
- Valence trung binh: {r['avg_v']:+.3f}
- Arousal trung binh: {r['avg_a']:+.3f}
- Mood chu dao:       {MOOD_VI[r['dominant_mood']]}
- So doan cam xuc:    {len(r['segments'])}

CHI TIET TIMELINE:
"""
            for i, s in enumerate(r['segments'], 1):
                report += f"  {i}. {s['start']:6.1f}s -> {s['end']:6.1f}s : {MOOD_VI[s['mood']]}\n"
            st.download_button("📝 Tai bao cao TXT", report,
                                file_name=f"{r['filename']}_report.txt",
                                mime="text/plain", use_container_width=True)


# =============================================================================
# TAB 2 — COMPARE 2 SONGS
# =============================================================================
with tab2:
    st.markdown("### ⚖️ So sanh 2 bai nhac")
    st.caption("Upload 2 file de so sanh dien bien cam xuc song song.")

    cc1, cc2 = st.columns(2)
    with cc1:
        f1 = st.file_uploader("🎵 Bai 1", type=["mp3", "wav"], key="cmp1")
    with cc2:
        f2 = st.file_uploader("🎵 Bai 2", type=["mp3", "wav"], key="cmp2")

    if f1 and f2 and st.button("🔍 So sanh", type="primary"):
        try:
            with st.spinner("Dang phan tich 2 bai..."):
                model, dev = load_model(ckpt_file)
                results = []
                for f in [f1, f2]:
                    tmp = Path(tempfile.gettempdir()) / f.name
                    tmp.write_bytes(f.read())
                    y, total, times, pr = predict(tmp, model, dev, smoothing)
                    results.append({
                        "name": f.name, "audio": str(tmp),
                        "times": times, "v": pr[:, 0], "a": pr[:, 1],
                        "avg_v": float(pr[:, 0].mean()),
                        "avg_a": float(pr[:, 1].mean()),
                    })

            for i, r in enumerate(results):
                st.markdown(f"#### 🎵 {r['name']}")
                st.audio(r['audio'])

            fig_cmp = make_subplots(
                rows=2, cols=1,
                subplot_titles=("Valence comparison", "Arousal comparison"),
                vertical_spacing=0.15,
            )
            colors = ["#4facfe", "#f5576c"]
            for i, r in enumerate(results):
                fig_cmp.add_trace(go.Scatter(x=r['times'], y=r['v'],
                                              name=f"V — {r['name'][:20]}",
                                              line=dict(color=colors[i], width=3)),
                                  row=1, col=1)
                fig_cmp.add_trace(go.Scatter(x=r['times'], y=r['a'],
                                              name=f"A — {r['name'][:20]}",
                                              line=dict(color=colors[i], width=3, dash='dash')),
                                  row=2, col=1)
            fig_cmp.update_layout(template="plotly_white", height=600,
                                   paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_cmp, use_container_width=True)

            cmp_df = pd.DataFrame({
                "Bai": [r['name'] for r in results],
                "Valence TB": [f"{r['avg_v']:+.3f}" for r in results],
                "Arousal TB": [f"{r['avg_a']:+.3f}" for r in results],
                "Mood chu dao": [f"{MOOD_EMOJIS[quadrant(r['avg_v'], r['avg_a'])]} "
                                  f"{MOOD_VI[quadrant(r['avg_v'], r['avg_a'])]}"
                                  for r in results],
            })
            st.dataframe(cmp_df, use_container_width=True, hide_index=True)

            min_len = min(len(results[0]['v']), len(results[1]['v']))
            corr_v = np.corrcoef(results[0]['v'][:min_len], results[1]['v'][:min_len])[0, 1]
            corr_a = np.corrcoef(results[0]['a'][:min_len], results[1]['a'][:min_len])[0, 1]
            sim = (corr_v + corr_a) / 2
            st.markdown(f"""
            <div class="insight-box">
                🤝 <b>Do tuong dong cam xuc</b>: {sim*100:.1f}%
                (Valence corr: {corr_v:.3f}, Arousal corr: {corr_a:.3f})
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Loi: {e}")


# =============================================================================
# TAB 3 — MEL-SPECTROGRAM EXPLORER
# =============================================================================
with tab3:
    st.markdown("### 🔍 Kham pha Mel-Spectrogram")
    st.caption("Xem dac trung audio ma model dung de du doan cam xuc.")

    mel_file = st.file_uploader("Upload file de xem mel-spectrogram",
                                  type=["mp3", "wav"], key="mel_upload")

    if mel_file is not None:
        tmp = Path(tempfile.gettempdir()) / mel_file.name
        tmp.write_bytes(mel_file.read())
        st.audio(str(tmp))

        with st.spinner("Trich xuat mel..."):
            y, sr = librosa.load(tmp, sr=SR, mono=True)
            duration = len(y) / sr

            mel_full = librosa.feature.melspectrogram(
                y=y, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH,
                n_mels=N_MELS, fmin=FMIN, fmax=FMAX, power=2.0)
            mel_db = librosa.power_to_db(mel_full, ref=np.max)

        st.markdown("#### 〰️ Waveform")
        t_wave = np.arange(len(y)) / sr
        step = max(1, len(y) // 5000)
        fig_wave = go.Figure(go.Scatter(x=t_wave[::step], y=y[::step],
                                          line=dict(color="#4facfe", width=1)))
        fig_wave.update_layout(template="plotly_white", height=200,
                                xaxis_title="Time (s)", yaxis_title="Amplitude",
                                paper_bgcolor='rgba(0,0,0,0)',
                                margin=dict(t=20, b=40))
        st.plotly_chart(fig_wave, use_container_width=True)

        st.markdown("#### 🌈 Mel-Spectrogram (Log scale)")
        fig_mel = go.Figure(data=go.Heatmap(
            z=mel_db,
            x=np.linspace(0, duration, mel_db.shape[1]),
            y=np.linspace(FMIN, FMAX, N_MELS),
            colorscale='Magma',
            colorbar=dict(title="dB"),
        ))
        fig_mel.update_layout(template="plotly_white", height=400,
                               xaxis_title="Time (s)", yaxis_title="Mel frequency (Hz)",
                               paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_mel, use_container_width=True)

        st.markdown("#### 📊 Thong ke audio")
        spec_centroid = librosa.feature.spectral_centroid(y=y, sr=SR)[0].mean()
        zero_crossing = librosa.feature.zero_crossing_rate(y)[0].mean()
        rms_energy = librosa.feature.rms(y=y)[0].mean()
        tempo, _ = librosa.beat.beat_track(y=y, sr=SR)
        tempo_val = float(tempo) if np.isscalar(tempo) else float(tempo[0])

        sc1, sc2, sc3, sc4 = st.columns(4)
        with sc1:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Tempo</div>
                <div class="metric-value">{tempo_val:.0f}</div>
                <div style="color:rgba(255,255,255,0.5)">BPM</div>
            </div>""", unsafe_allow_html=True)
        with sc2:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Spectral Centroid</div>
                <div class="metric-value">{spec_centroid:.0f}</div>
                <div style="color:rgba(255,255,255,0.5)">Hz</div>
            </div>""", unsafe_allow_html=True)
        with sc3:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Zero Crossing Rate</div>
                <div class="metric-value">{zero_crossing:.3f}</div>
            </div>""", unsafe_allow_html=True)
        with sc4:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">RMS Energy</div>
                <div class="metric-value">{rms_energy:.3f}</div>
            </div>""", unsafe_allow_html=True)


# =============================================================================
# TAB 4 — HISTORY
# =============================================================================
with tab4:
    st.markdown("### 📚 Lich su phan tich")

    if not st.session_state.history:
        st.info("Chua co phan tich nao. Quay lai tab 'Phan tich bai nhac' de bat dau!")
    else:
        st.caption(f"Da phan tich {len(st.session_state.history)} bai nhac trong phien nay.")

        all_v = [r['avg_v'] for r in st.session_state.history]
        all_a = [r['avg_a'] for r in st.session_state.history]
        all_moods = [r['dominant_mood'] for r in st.session_state.history]

        sh1, sh2, sh3 = st.columns(3)
        with sh1:
            st.metric("Tong so bai", len(st.session_state.history))
        with sh2:
            st.metric("V trung binh tat ca", f"{np.mean(all_v):+.3f}")
        with sh3:
            most_common = Counter(all_moods).most_common(1)[0][0]
            st.metric("Mood pho bien nhat", MOOD_EMOJIS[most_common] + " " + MOOD_VI[most_common])

        st.markdown("#### 📜 Danh sach")
        hist_df = pd.DataFrame([
            {
                "STT": i+1,
                "File": r['filename'],
                "Model": r['model'],
                "Mood": f"{MOOD_EMOJIS[r['dominant_mood']]} {MOOD_VI[r['dominant_mood']]}",
                "V_avg": f"{r['avg_v']:+.2f}",
                "A_avg": f"{r['avg_a']:+.2f}",
                "Time": r['timestamp'],
            }
            for i, r in enumerate(st.session_state.history)
        ])
        st.dataframe(hist_df, use_container_width=True, hide_index=True)

        st.markdown("#### 📊 Phan bo cam xuc cua tat ca cac bai")
        all_mood_counts = Counter(all_moods)
        fig_hist = go.Figure(data=[go.Bar(
            x=[MOOD_VI[m] for m in all_mood_counts.keys()],
            y=list(all_mood_counts.values()),
            marker=dict(color=[MOOD_COLORS[m] for m in all_mood_counts.keys()]),
            text=list(all_mood_counts.values()),
            textposition='auto',
        )])
        fig_hist.update_layout(template="plotly_white", height=300,
                                paper_bgcolor='rgba(0,0,0,0)',
                                yaxis_title="So bai")
        st.plotly_chart(fig_hist, use_container_width=True)

        st.markdown("#### 🎯 Tat ca bai tren ban do V-A")
        fig_scatter = go.Figure()
        for (vs, as_), name in QUADRANTS.items():
            fig_scatter.add_shape(
                type="rect",
                x0=min(0, vs), y0=min(0, as_), x1=max(0, vs), y1=max(0, as_),
                fillcolor=MOOD_COLORS[name], opacity=0.1, line_width=0
            )
        fig_scatter.add_trace(go.Scatter(
            x=all_v, y=all_a, mode='markers+text',
            marker=dict(size=14, color=[MOOD_COLORS[m] for m in all_moods],
                         line=dict(color='#2c3e50', width=1.5)),
            text=[r['filename'][:15] for r in st.session_state.history],
            textposition='top center', textfont=dict(color='#2c3e50', size=10),
        ))
        fig_scatter.add_hline(y=0, line_color="#2c3e50", opacity=0.3)
        fig_scatter.add_vline(x=0, line_color="#2c3e50", opacity=0.3)
        fig_scatter.update_layout(template="plotly_white", height=500,
                                   xaxis_title="Valence", yaxis_title="Arousal",
                                   xaxis_range=[-1, 1], yaxis_range=[-1, 1],
                                   paper_bgcolor='rgba(0,0,0,0)',
                                   showlegend=False)
        st.plotly_chart(fig_scatter, use_container_width=True)


# =============================================================================
# TAB 5 — ABOUT
# =============================================================================
with tab5:
    st.markdown("""
    ### ℹ️ Gioi thieu he thong

    #### 🎯 Muc tieu
    Phan tich **dien bien cam xuc theo thoi gian** trong am nhac, du doan **Valence** (tich cuc/tieu cuc)
    va **Arousal** (nang luong) cho moi cua so 0.5 giay.

    #### 🏗️ Kien truc
    - **CNN encoder**: trich dac trung khong gian tu Mel-spectrogram (3 lop conv, 1.4M params)
    - **BiLSTM**: mo hinh moi quan he thoi gian giua cac cua so 0.5s
    - **Multi-head Attention**: bat dependency dai han giua cac timestep
    - **CCC Loss + Class Weighting**: toi uu truc tiep metric danh gia + xu ly imbalance

    #### 📊 Dataset
    - **DEAM** (1802 bai nhac phuong Tay, da the loai) — train chinh
    - **PMEmo** (~767 bai Chinese pop) — cross-dataset evaluation

    #### 📈 Ket qua
    | Metric | Gia tri | So voi SOTA Aljanaki 2017 |
    |---|---|---|
    | CCC_V (DEAM) | 0.65 | +116% |
    | CCC_A (DEAM) | 0.77 | +24% |
    | CCC_V (PMEmo, transfer) | 0.69 | best in class |

    #### 🎨 Bon vung cam xuc V-A (Russell 1980)
    """)

    cols = st.columns(4)
    quadrants_info = [
        ("happy/excited", "Vui ve / Hung phan", "V > 0, A > 0", "Pop, Dance, Disco"),
        ("tense/angry", "Cang thang / Tuc gian", "V < 0, A > 0", "Heavy Metal, Punk"),
        ("sad", "Buon ba", "V < 0, A < 0", "Blues, Slow ballad"),
        ("calm/relaxed", "Thu thai / Binh yen", "V > 0, A < 0", "Jazz, Classical, Lo-fi"),
    ]
    for col, (mood, name, va, genres) in zip(cols, quadrants_info):
        color = MOOD_COLORS[mood]
        emoji = MOOD_EMOJIS[mood]
        with col:
            st.markdown(f"""
            <div style="background:white; padding:1.5rem; border-radius:14px;
                         border-top:5px solid {color}; height:210px;
                         box-shadow:0 4px 15px rgba(118, 75, 162, 0.08);">
                <div style="font-size:2.8rem; text-align:center;">{emoji}</div>
                <div style="text-align:center; font-weight:bold; color:{color}; margin:0.5rem 0;">
                    {name}
                </div>
                <div style="text-align:center; color:#5a6c7d; font-size:0.85rem;">
                    {va}<br><i>{genres}</i>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    ---
    #### 🛠️ Cong nghe su dung
    - **Backend**: PyTorch + Librosa + NumPy
    - **Frontend**: Streamlit + Plotly
    - **Audio**: Mel-spectrogram (64 mel-bands, 22050 Hz, hop 256)

    #### 📚 Tai lieu tham khao
    1. Aljanaki et al. (2017). *Developing a benchmark for emotional analysis of music*. PLoS ONE.
    2. Zhang et al. (2018). *The PMEmo Dataset for Music Emotion Recognition*. ICMR.
    3. Vaswani et al. (2017). *Attention is All You Need*. NeurIPS.

    #### 👨‍🎓 Tac gia
    Phung — Master Thesis — Khoa hoc May tinh Ung dung
    """)


# =============================================================================
# FOOTER
# =============================================================================
st.markdown("""
<div class="footer">
    🎵 Music Emotion Analyzer | Master Thesis Demo |
    Built with Streamlit, PyTorch & ❤️
</div>
""", unsafe_allow_html=True)
