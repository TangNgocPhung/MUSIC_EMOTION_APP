"""
================================================================================
HỆ THỐNG PHÂN TÍCH DIỄN BIẾN CẢM XÚC TRONG ÂM NHẠC
Streamlit Demo App — Phiên bản Master Thesis
================================================================================
Tác giả : Phụng
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
# 1. CẤU HÌNH
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

# === HẰNG SỐ DEFAULT (thay cho slider trong UI) ===
DEFAULT_SMOOTHING   = 5      # Cửa sổ moving average tối ưu (qua thực nghiệm)
DEFAULT_MIN_SEG_LEN = 3.0    # Đoạn ngắn hơn 3 giây sẽ được gộp

CKPT_DIR = Path(".")

# === 4 MODELS CỦA ĐỀ TÀI ===
# Tên ngắn gọn để combobox hiển thị đầy đủ, không bị cắt thành "..."
AVAILABLE_CHECKPOINTS = {
    "⭐ Đề xuất chính (Best)":      "best_attention_balanced.pt",
    "Attention + CCC Loss":          "best_attention.pt",
    "Baseline (SmoothL1)":           "best.pt",
    "Transfer Learning":             "best_pmemo_ft_head.pt",
}

QUADRANTS = {(+1,+1):"happy/excited", (-1,+1):"tense/angry",
              (-1,-1):"sad",          (+1,-1):"calm/relaxed"}
MOOD_COLORS = {"happy/excited":"#f39c12", "tense/angry":"#e74c3c",
                "sad":"#3498db",          "calm/relaxed":"#2ecc71"}
MOOD_EMOJIS = {"happy/excited":"😊", "tense/angry":"😠",
                "sad":"😢",          "calm/relaxed":"😌"}
MOOD_VI = {"happy/excited":"Vui vẻ / Hưng phấn",
            "tense/angry":  "Căng thẳng / Tức giận",
            "sad":           "Buồn bã",
            "calm/relaxed":  "Thư thái / Bình yên"}


# =============================================================================
# 2. KIẾN TRÚC MODEL
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
    """Baseline: CNN + BiLSTM (cho best.pt, best_pmemo_ft_head.pt)"""
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
# 3. HÀM XỬ LÝ AUDIO + DỰ ĐOÁN
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


# =============================================================================
# 3.1. HÀM SINH GIẢI THÍCH DÀNH CHO NGƯỜI KHÔNG CHUYÊN
# =============================================================================
def describe_valence(v):
    """Mô tả mức Valence bằng ngôn ngữ thường."""
    if v >= 0.5:    return "RẤT TÍCH CỰC", "Bài nhạc nghe rất vui, hạnh phúc"
    if v >= 0.15:   return "TÍCH CỰC", "Bài nhạc nghe vui, dễ chịu"
    if v >= -0.15:  return "TRUNG TÍNH", "Bài nhạc cân bằng, không quá vui cũng không quá buồn"
    if v >= -0.5:   return "TIÊU CỰC NHẸ", "Bài nhạc hơi buồn, u sầu nhẹ"
    return "RẤT TIÊU CỰC", "Bài nhạc rất buồn, sầu thảm"


def describe_arousal(a):
    """Mô tả mức Arousal bằng ngôn ngữ thường."""
    if a >= 0.5:    return "RẤT CAO", "Năng lượng mãnh liệt, sôi động (tiếng trống mạnh, tempo nhanh)"
    if a >= 0.15:   return "CAO", "Có nhịp điệu rõ ràng, tương đối sôi động"
    if a >= -0.15:  return "TRUNG BÌNH", "Năng lượng cân bằng, không quá nhanh cũng không quá chậm"
    if a >= -0.5:   return "THẤP", "Bài nhạc chậm rãi, êm dịu"
    return "RẤT THẤP", "Bài nhạc rất yên tĩnh, gần như tĩnh lặng"


def explain_emotion_decision(v, a, mood):
    """Giải thích vì sao model phân loại thành cảm xúc này."""
    v_label, v_desc = describe_valence(v)
    a_label, a_desc = describe_arousal(a)

    # Phân tích theo dấu V và A
    v_sign = "DƯƠNG (+)" if v >= 0 else "ÂM (−)"
    a_sign = "DƯƠNG (+)" if a >= 0 else "ÂM (−)"

    explanation = f"""
**Bước 1: Đo 2 chỉ số chính**
- 🎯 **Valence = {v:+.2f}** → mức {v_label}: _{v_desc}_
- ⚡ **Arousal = {a:+.2f}** → mức {a_label}: _{a_desc}_

**Bước 2: Kết hợp dấu của V và A**
- Valence {v_sign} và Arousal {a_sign}
- → Kết quả: **{MOOD_EMOJIS[mood]} {MOOD_VI[mood]}**

**Bước 3: Cảnh báo độ tin cậy**
"""
    confidence = min(abs(v), abs(a))
    if confidence < 0.1:
        explanation += "⚠️ Cả V và A đều gần 0 → kết quả **ranh giới**, có thể model dao động giữa 2 cảm xúc."
    elif confidence < 0.25:
        explanation += "🟡 V hoặc A gần 0 → kết quả **trung bình**, model khá chắc chắn nhưng có thể nhầm."
    else:
        explanation += "✅ V và A đều cách xa 0 → kết quả **CHẮC CHẮN**, model phân loại rõ ràng."
    return explanation


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
def predict(audio_path, model, dev, smoothing=DEFAULT_SMOOTHING):
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
# 4. STREAMLIT UI — CSS + CẤU HÌNH TRANG
# =============================================================================
st.set_page_config(
    page_title="Phân tích Cảm xúc Âm nhạc",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* ============ THEME SÁNG — Pastel gradient ============ */
    .stApp {
        background: linear-gradient(135deg, #fdfbfb 0%, #ebedee 50%, #f5e6f7 100%);
    }

    /* Header */
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

    /* Metric cards */
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

    /* Sidebar */
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

    /* Insight box */
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
# 5. KHỞI TẠO SESSION STATE
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
    <h1>🎵 Phân tích Cảm xúc Âm nhạc</h1>
    <p>Phân tích diễn biến cảm xúc theo thời gian | CNN + BiLSTM + Attention | DEAM + PMEmo</p>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# 7. SIDEBAR — CHỌN MODEL
# =============================================================================
with st.sidebar:
    st.markdown("### 🤖 Chọn mô hình")

    available = {k: v for k, v in AVAILABLE_CHECKPOINTS.items()
                 if (CKPT_DIR / v).exists()}
    if not available:
        st.error("⚠️ Không tìm thấy checkpoint nào trong thư mục!")
        st.info("Cần các file: best_attention_balanced.pt, best_attention.pt, best.pt, best_pmemo_ft_head.pt")
        st.stop()

    selected_model = st.selectbox(
        "Mô hình sử dụng để phân tích",
        list(available.keys()),
        help="⭐ là mô hình đề xuất chính (tốt nhất)"
    )
    ckpt_file = available[selected_model]

    st.markdown("---")
    st.markdown("### 📊 Thông tin mô hình")

    # Mô tả chi tiết cho 4 models (key ngắn gọn khớp với AVAILABLE_CHECKPOINTS)
    model_info = {
        "⭐ Đề xuất chính (Best)":
            ("CNN + BiLSTM + Multi-head Attention",
             "Combined Loss + Class Weighting",
             "DEAM (1802 bài)",
             "Mô hình cuối cùng, xử lý class imbalance"),
        "Attention + CCC Loss":
            ("CNN + BiLSTM + Multi-head Attention",
             "CCC Loss",
             "DEAM (1802 bài)",
             "Chứng minh đóng góp của Attention"),
        "Baseline (SmoothL1)":
            ("CNN + BiLSTM",
             "SmoothL1 Loss",
             "DEAM (1802 bài)",
             "Mô hình tham chiếu cơ bản"),
        "Transfer Learning":
            ("CNN + BiLSTM (Freeze CNN)",
             "SmoothL1 Loss",
             "Pretrain DEAM + Fine-tune PMEmo",
             "Cross-cultural generalization"),
    }
    arch, loss, dataset, role = model_info.get(selected_model,
                                                ("?", "?", "?", "?"))
    st.markdown(f"""
- **Kiến trúc**: `{arch}`
- **Hàm mất mát**: `{loss}`
- **Dữ liệu huấn luyện**: {dataset}
- **Vai trò**: _{role}_
""")

    st.markdown("---")
    st.markdown("### 📋 4 mô hình của đề tài")
    st.markdown("""
1. ⭐ **Attention + Balanced** — Đề xuất chính
2. **Attention thuần** — Ablation
3. **Baseline** — Tham chiếu
4. **Transfer Learning** — Cross-dataset
""")

    st.markdown("---")
    if st.button("🗑️ Xóa lịch sử"):
        st.session_state.history = []
        st.session_state.current_result = None
        st.rerun()

    st.markdown("---")
    st.markdown("""
    <div style='text-align:center; color:#9aa5b1; font-size:0.85rem; padding-top:1rem;'>
        🎓 Advanced Machine Learning course<br>
        Built with Streamlit + PyTorch
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# 8. MAIN TABS
# =============================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎼 Phân tích bài nhạc",
    "⚖️ So sánh giữa các bài",
    "🔍 Mel-Spectrogram",
    "📚 Lịch sử phân tích",
    "ℹ️ Giới thiệu",
])


# =============================================================================
# TAB 1 — PHÂN TÍCH 1 BÀI NHẠC
# =============================================================================
with tab1:
    st.markdown("### 📤 Tải lên bài nhạc")

    col_u1, col_u2 = st.columns([3, 1])
    with col_u1:
        up = st.file_uploader("Chọn file MP3 / WAV", type=["mp3", "wav"],
                              key="single_upload", label_visibility="collapsed")
    with col_u2:
        analyze_btn = st.button("🚀 Phân tích", use_container_width=True, type="primary")

    if up is not None and analyze_btn:
        tmp = Path(tempfile.gettempdir()) / up.name
        tmp.write_bytes(up.read())

        try:
            with st.spinner("🎵 Đang trích xuất đặc trưng âm thanh..."):
                model, dev = load_model(ckpt_file)
                y, total, times, pr = predict(tmp, model, dev, DEFAULT_SMOOTHING)

            moods = [quadrant(v, a) for v, a in pr]
            segs = group_timeline(times, moods, DEFAULT_MIN_SEG_LEN)
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
            st.error(f"❌ Lỗi: {str(e)}")
            with st.expander("Chi tiết lỗi"):
                st.code(traceback.format_exc())

    if st.session_state.current_result:
        r = st.session_state.current_result
        st.audio(r["audio_path"])

        st.markdown("### 📈 Tổng quan")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Thời lượng</div>
                <div class="metric-value">{r['duration']:.1f}s</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Valence trung bình</div>
                <div class="metric-value">{r['avg_v']:+.2f}</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Arousal trung bình</div>
                <div class="metric-value">{r['avg_a']:+.2f}</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            color = MOOD_COLORS[r['dominant_mood']]
            emoji = MOOD_EMOJIS[r['dominant_mood']]
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Cảm xúc chủ đạo</div>
                <div style="font-size:2.2rem; margin:0.3rem 0;">{emoji}</div>
                <div style="color:{color}; font-weight:700; font-size:1.05rem;">
                    {MOOD_VI[r['dominant_mood']]}
                </div>
            </div>""", unsafe_allow_html=True)

        unique_moods = list(set(r['moods']))
        if len(unique_moods) > 1:
            mood_seq = " → ".join([MOOD_VI[s['mood']] for s in r['segments']])
            insight_text = (
                f"<b>Bài nhạc trải qua {len(r['segments'])} đoạn cảm xúc:</b> {mood_seq}.<br>"
                f"<b>Cảm xúc trung bình</b>: {MOOD_VI[r['dominant_mood']]} (V={r['avg_v']:+.2f}, A={r['avg_a']:+.2f})."
            )
        else:
            insight_text = (
                f"<b>Cảm xúc nhất quán xuyên suốt bài nhạc</b>: {MOOD_VI[r['dominant_mood']]} "
                f"(V={r['avg_v']:+.2f}, A={r['avg_a']:+.2f})."
            )
        st.markdown(f'<div class="insight-box">💡 {insight_text}</div>', unsafe_allow_html=True)

        # === GIẢI THÍCH CHI TIẾT CÁCH MODEL QUYẾT ĐỊNH CẢM XÚC ===
        with st.expander("🎓 **Giải thích chi tiết: Vì sao bài này được phân loại như vậy?**", expanded=True):
            st.markdown(explain_emotion_decision(r['avg_v'], r['avg_a'], r['dominant_mood']))

            st.markdown("---")
            st.markdown("""
            #### 📚 Hiểu về 2 chỉ số Valence và Arousal

            Trong khoa học âm nhạc, mọi cảm xúc của bài nhạc đều có thể đo bằng **2 con số**:

            **🎯 Valence (Tích cực/Tiêu cực)** — _Thang từ -1 đến +1_
            - **+1.0** = Cực kỳ vui (như "Happy" của Pharrell Williams)
            - **+0.5** = Vui vẻ rõ rệt (Pop ballad happy)
            - **0.0** = Trung tính (nhạc nền không cảm xúc)
            - **-0.5** = Buồn rõ rệt (Slow ballad)
            - **-1.0** = Cực kỳ buồn (Funeral march)

            **⚡ Arousal (Năng lượng)** — _Thang từ -1 đến +1_
            - **+1.0** = Cực kỳ sôi động (EDM, Heavy Metal)
            - **+0.5** = Tempo nhanh, có nhịp (Pop, Rock)
            - **0.0** = Năng lượng vừa phải
            - **-0.5** = Chậm rãi (Acoustic, Slow Jazz)
            - **-1.0** = Gần như tĩnh lặng (Ambient, Drone)

            #### 🗺️ 4 vùng cảm xúc khi kết hợp V và A

            | Valence | Arousal | Cảm xúc | Ví dụ thể loại |
            |---|---|---|---|
            | + (vui) | + (cao) | 😊 **Vui vẻ / Hưng phấn** | Pop, Dance, Disco |
            | − (buồn) | + (cao) | 😠 **Căng thẳng / Tức giận** | Heavy Metal, Punk |
            | − (buồn) | − (thấp) | 😢 **Buồn bã** | Blues, Slow Ballad |
            | + (vui) | − (thấp) | 😌 **Thư thái / Bình yên** | Lo-fi, Jazz, Classical |

            **Mô hình của Russell (1980)** — đây là cách tâm lý học chuẩn để biểu diễn cảm xúc.
            """)

        st.markdown("### 📉 Diễn biến V-A theo thời gian")
        st.caption("📊 Biểu đồ thể hiện cảm xúc thay đổi từng giây trong bài nhạc")
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
            xaxis_title="Thời gian (giây)",
            yaxis_title="V / A (-1 đến 1)",
            yaxis_range=[-1.1, 1.1],
            hovermode='x unified',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0.2)',
        )
        st.plotly_chart(fig_timeline, use_container_width=True)

        # === GIẢI THÍCH CÁCH ĐỌC BIỂU ĐỒ TIMELINE ===
        with st.expander("📖 **Cách đọc biểu đồ này**", expanded=True):
            # Phân tích xu hướng thực tế của bài nhạc
            v_start = float(np.mean(v_arr[:5])); v_end = float(np.mean(v_arr[-5:]))
            a_start = float(np.mean(a_arr[:5])); a_end = float(np.mean(a_arr[-5:]))
            v_trend = "TĂNG (vui dần lên)" if v_end - v_start > 0.1 else \
                      "GIẢM (buồn dần)" if v_start - v_end > 0.1 else "ỔN ĐỊNH"
            a_trend = "TĂNG (sôi động dần)" if a_end - a_start > 0.1 else \
                      "GIẢM (lắng dần)" if a_start - a_end > 0.1 else "ỔN ĐỊNH"

            st.markdown(f"""
            #### 🔵 Đường XANH DƯƠNG = Valence (Mức độ tích cực)
            - **Lên cao (>0)** = Bài đang VUI hơn 😊
            - **Xuống thấp (<0)** = Bài đang BUỒN hơn 😢
            - **Cắt qua đường 0** = Cảm xúc CHUYỂN HƯỚNG (vui → buồn hoặc ngược lại)

            #### 🔴 Đường ĐỎ = Arousal (Mức độ năng lượng)
            - **Lên cao (>0)** = Bài đang SÔI ĐỘNG hơn ⚡
            - **Xuống thấp (<0)** = Bài đang YÊN TĨNH hơn 🌙

            ---

            #### 📊 Phân tích bài nhạc của bạn:

            - **Valence**: bắt đầu **{v_start:+.2f}**, kết thúc **{v_end:+.2f}** → xu hướng **{v_trend}**
            - **Arousal**: bắt đầu **{a_start:+.2f}**, kết thúc **{a_end:+.2f}** → xu hướng **{a_trend}**

            #### 💡 Ý nghĩa:
            - **2 đường cùng đi LÊN** → Bài đang "BUNG NỞ" (ví dụ: chorus, cao trào)
            - **2 đường cùng đi XUỐNG** → Bài đang "LẮNG XUỐNG" (ví dụ: outro, đoạn lặng)
            - **Đường xanh ↑ + đường đỏ ↓** → Đang chuyển sang ÊM DỊU, vui nhẹ
            - **Đường xanh ↓ + đường đỏ ↑** → Đang chuyển sang CĂNG THẲNG
            - **Hai đường gần nhau** → Cảm xúc nhất quán, không "nổi loạn"
            """)

        col_a, col_b = st.columns([2, 1])

        with col_a:
            st.markdown("#### 🎨 Bản đồ cảm xúc theo thời gian")
            st.caption("Mỗi vùng màu = 1 cảm xúc đang chiếm ưu thế trong khoảng thời gian đó")
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
                xaxis_title="Thời gian (giây)", yaxis=dict(visible=False),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(t=20, b=40),
            )
            st.plotly_chart(fig_mood, use_container_width=True)

            # === GIẢI THÍCH BẢN ĐỒ MOOD ===
            with st.expander("ℹ️ Giải thích biểu đồ này", expanded=True):
                st.markdown("""
                **Cách đọc:**
                - 🟠 **Cam** = Vui vẻ / Hưng phấn (V+, A+)
                - 🔴 **Đỏ** = Căng thẳng / Tức giận (V−, A+)
                - 🔵 **Xanh dương** = Buồn bã (V−, A−)
                - 🟢 **Xanh lá** = Thư thái / Bình yên (V+, A−)

                **Đoạn càng DÀI** = Cảm xúc đó duy trì càng lâu trong bài.
                **Nhiều đoạn liên tiếp khác màu** = Bài có nhiều biến chuyển cảm xúc (kể chuyện).
                """)

        with col_b:
            st.markdown("#### 🎯 Quỹ đạo V-A")
            st.caption("Đường đi cảm xúc trên bản đồ 2 chiều — đậm = đầu bài, sáng = cuối bài")
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
                            showscale=True, colorbar=dict(title="Giây")),
                name="Quỹ đạo",
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

            # === GIẢI THÍCH QUỸ ĐẠO V-A ===
            with st.expander("ℹ️ Giải thích quỹ đạo", expanded=True):
                st.markdown("""
                **Cách đọc:**
                - **Trục NGANG** = Valence (càng phải = càng vui)
                - **Trục DỌC** = Arousal (càng lên = càng sôi động)
                - **Màu điểm** = Thời gian (đậm = đầu, sáng = cuối)

                **4 góc tương ứng 4 cảm xúc** (xem 4 emoji ở 4 góc).

                **Quỹ đạo nói gì?**
                - Ngắn, tập trung 1 góc → Cảm xúc nhất quán
                - Dài, trải rộng → Cảm xúc biến chuyển nhiều
                - Đi từ góc dưới-phải lên trên-phải → Bình yên → Hưng phấn (cao trào)
                - Vòng tròn quanh tâm (0,0) → Cảm xúc trung tính, không rõ ràng
                """)

        col_c, col_d = st.columns([1, 1])

        with col_c:
            st.markdown("#### 🥧 Phân bố cảm xúc")
            st.caption("Tỷ lệ thời gian mỗi cảm xúc xuất hiện trong toàn bài")
            mood_counts = Counter(r['moods'])
            max_count = max(mood_counts.values())
            fig_pie = go.Figure(data=[go.Pie(
                labels=[MOOD_VI[m] for m in mood_counts.keys()],
                values=list(mood_counts.values()),
                marker=dict(colors=[MOOD_COLORS[m] for m in mood_counts.keys()],
                            line=dict(color='white', width=2)),
                hole=0.45,
                # Đặt label + % BÊN NGOÀI pie để đọc rõ tiếng Việt có dấu
                textinfo='label+percent',
                textposition='outside',
                # Font mặc định hỗ trợ tiếng Việt, màu đen đậm để đọc rõ trên nền trắng
                textfont=dict(color='#2c3e50', size=13),
                insidetextorientation='radial',
                # Hover chi tiết khi rê chuột
                hovertemplate='<b>%{label}</b><br>Số timestep: %{value}<br>Tỷ lệ: %{percent}<extra></extra>',
                # Tách nhẹ slice lớn nhất để làm nổi bật
                pull=[0.04 if v == max_count else 0 for v in mood_counts.values()],
                sort=False,
            )])
            fig_pie.update_layout(
                template="plotly_white", height=400,
                paper_bgcolor='rgba(0,0,0,0)',
                showlegend=True,
                legend=dict(
                    orientation="v",
                    yanchor="middle", y=0.5,
                    xanchor="left", x=1.05,
                    font=dict(size=11, color='#2c3e50'),
                ),
                margin=dict(t=40, b=40, l=40, r=120),
            )
            st.plotly_chart(fig_pie, use_container_width=True)

            # === GIẢI THÍCH PIE CHART ===
            with st.expander("ℹ️ Cách đọc biểu đồ tròn", expanded=True):
                # Tự động tìm cảm xúc chiếm nhiều nhất
                most_mood = max(mood_counts, key=mood_counts.get)
                most_pct = 100 * mood_counts[most_mood] / sum(mood_counts.values())
                st.markdown(f"""
                **Bài nhạc của bạn:**
                - Cảm xúc **{MOOD_VI[most_mood]}** chiếm **{most_pct:.1f}%** thời lượng
                - Có {len(mood_counts)} loại cảm xúc khác nhau xuất hiện

                **Hiểu đơn giản:**
                - Pie 1 màu lớn → Bài có cảm xúc CHỦ ĐẠO RÕ RỆT
                - Pie chia đều → Bài có nhiều cảm xúc đan xen
                - Pie nhiều miếng nhỏ → Bài "phức tạp" về cảm xúc
                """)

        with col_d:
            st.markdown("#### 📋 Bảng các đoạn cảm xúc")
            st.caption("Liệt kê chi tiết từng đoạn cảm xúc trong bài (theo thứ tự thời gian)")
            seg_df = pd.DataFrame([
                {
                    "#": i+1,
                    "Từ": f"{s['start']:.1f}s",
                    "Đến": f"{s['end']:.1f}s",
                    "Thời lượng": f"{s['end']-s['start']:.1f}s",
                    "Cảm xúc": f"{MOOD_EMOJIS[s['mood']]} {MOOD_VI[s['mood']]}",
                }
                for i, s in enumerate(r['segments'])
            ])
            st.dataframe(seg_df, use_container_width=True, hide_index=True, height=350)

        st.markdown("### 💾 Xuất kết quả")
        ec1, ec2, ec3 = st.columns(3)

        with ec1:
            csv_data = pd.DataFrame({
                "thoi_gian_giay": r['times'],
                "valence":         r['valence'],
                "arousal":         r['arousal'],
                "cam_xuc":         r['moods'],
            }).to_csv(index=False)
            st.download_button("📄 Tải CSV (chi tiết)", csv_data,
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
            st.download_button("📊 Tải JSON (tổng hợp)", json_data,
                                file_name=f"{r['filename']}_summary.json",
                                mime="application/json", use_container_width=True)

        with ec3:
            report = f"""BÁO CÁO PHÂN TÍCH CẢM XÚC ÂM NHẠC
==========================================
File:     {r['filename']}
Model:    {r['model']}
Thời gian: {r['timestamp']}
Thời lượng: {r['duration']:.1f}s

KẾT QUẢ:
- Valence trung bình: {r['avg_v']:+.3f}
- Arousal trung bình: {r['avg_a']:+.3f}
- Cảm xúc chủ đạo:    {MOOD_VI[r['dominant_mood']]}
- Số đoạn cảm xúc:    {len(r['segments'])}

CHI TIẾT TIMELINE:
"""
            for i, s in enumerate(r['segments'], 1):
                report += f"  {i}. {s['start']:6.1f}s -> {s['end']:6.1f}s : {MOOD_VI[s['mood']]}\n"
            st.download_button("📝 Tải báo cáo TXT", report,
                                file_name=f"{r['filename']}_report.txt",
                                mime="text/plain", use_container_width=True)


# =============================================================================
# TAB 2 — SO SÁNH NHIỀU BÀI NHẠC (2-10 BÀI)
# =============================================================================
with tab2:
    st.markdown("### ⚖️ So sánh nhiều bài nhạc")
    st.caption("Chọn số lượng bài, tải lên các file và bấm So sánh để xem diễn biến cảm xúc song song.")

    # === BƯỚC 1: NHẬP SỐ LƯỢNG BÀI ===
    n_songs = st.number_input(
        "📊 **Số lượng bài muốn so sánh**",
        min_value=2, max_value=10, value=2, step=1,
        help="Chọn từ 2 đến 10 bài nhạc để so sánh"
    )

    st.markdown(f"#### 📤 Tải lên {n_songs} bài nhạc:")

    # === BƯỚC 2: HIỂN THỊ N FILE UPLOADER ĐỘNG ===
    files = []
    cols_per_row = min(n_songs, 3)  # Tối đa 3 cột mỗi hàng
    for row_start in range(0, n_songs, cols_per_row):
        cols = st.columns(cols_per_row)
        for i in range(row_start, min(row_start + cols_per_row, n_songs)):
            with cols[i - row_start]:
                f = st.file_uploader(
                    f"🎵 Bài {i+1}",
                    type=["mp3", "wav"],
                    key=f"cmp_{i}"
                )
                files.append(f)

    all_uploaded = all(f is not None for f in files)

    if not all_uploaded:
        st.info(f"⏳ Vui lòng tải lên đủ {n_songs} bài để bắt đầu so sánh.")

    if all_uploaded and st.button("🔍 So sánh", type="primary", use_container_width=True):
        try:
            with st.spinner(f"⏳ Đang phân tích {n_songs} bài nhạc..."):
                model, dev = load_model(ckpt_file)
                results = []
                progress_bar = st.progress(0)
                for i, f in enumerate(files):
                    tmp = Path(tempfile.gettempdir()) / f.name
                    tmp.write_bytes(f.read())
                    y, total, times, pr = predict(tmp, model, dev, DEFAULT_SMOOTHING)
                    moods = [quadrant(v, a) for v, a in pr]
                    segs = group_timeline(times, moods, DEFAULT_MIN_SEG_LEN)
                    results.append({
                        "name": f.name, "audio": str(tmp),
                        "times": times, "v": pr[:, 0], "a": pr[:, 1],
                        "avg_v": float(pr[:, 0].mean()),
                        "avg_a": float(pr[:, 1].mean()),
                        "duration": total,
                        "segments": segs,
                        "moods": moods,
                    })
                    progress_bar.progress((i + 1) / n_songs)
                progress_bar.empty()

            st.success(f"✅ Đã phân tích xong {n_songs} bài!")

            # === HIỂN THỊ AUDIO PLAYERS ===
            st.markdown("### 🎵 Nghe lại các bài đã upload")
            audio_cols_per_row = min(n_songs, 3)
            for row_start in range(0, n_songs, audio_cols_per_row):
                acols = st.columns(audio_cols_per_row)
                for i in range(row_start, min(row_start + audio_cols_per_row, n_songs)):
                    with acols[i - row_start]:
                        short_name = results[i]['name'][:25] + "..." if len(results[i]['name']) > 25 else results[i]['name']
                        st.markdown(f"**🎵 Bài {i+1}**: {short_name}")
                        st.audio(results[i]['audio'])

            # === BIỂU ĐỒ SO SÁNH ===
            st.markdown("### 📈 Biểu đồ so sánh diễn biến cảm xúc")

            # Bảng màu cho tối đa 10 bài
            colors = ["#4facfe", "#f5576c", "#43e97b", "#f093fb",
                      "#feca57", "#ff6b6b", "#48dbfb", "#a55eea",
                      "#fd79a8", "#fdcb6e"]

            fig_cmp = make_subplots(
                rows=2, cols=1,
                subplot_titles=("📊 So sánh Valence (Tích cực/Tiêu cực) theo thời gian",
                                "📊 So sánh Arousal (Năng lượng) theo thời gian"),
                vertical_spacing=0.18,
            )
            for i, r in enumerate(results):
                color = colors[i % len(colors)]
                fig_cmp.add_trace(go.Scatter(
                    x=r['times'], y=r['v'],
                    name=f"Bài {i+1}",
                    legendgroup=f"song{i}",
                    line=dict(color=color, width=2.5),
                    hovertemplate=f'<b>Bài {i+1}</b><br>Thời gian: %{{x:.1f}}s<br>Valence: %{{y:.3f}}<extra></extra>'
                ), row=1, col=1)
                fig_cmp.add_trace(go.Scatter(
                    x=r['times'], y=r['a'],
                    name=f"Bài {i+1}",
                    legendgroup=f"song{i}",
                    showlegend=False,
                    line=dict(color=color, width=2.5, dash='dash'),
                    hovertemplate=f'<b>Bài {i+1}</b><br>Thời gian: %{{x:.1f}}s<br>Arousal: %{{y:.3f}}<extra></extra>'
                ), row=2, col=1)
            fig_cmp.update_xaxes(title_text="Thời gian (giây)", row=2, col=1)
            fig_cmp.update_yaxes(title_text="Valence", row=1, col=1, range=[-1.1, 1.1])
            fig_cmp.update_yaxes(title_text="Arousal", row=2, col=1, range=[-1.1, 1.1])
            fig_cmp.add_hline(y=0, line_dash="dot", line_color="gray", row=1, col=1)
            fig_cmp.add_hline(y=0, line_dash="dot", line_color="gray", row=2, col=1)
            fig_cmp.update_layout(template="plotly_white", height=650,
                                   paper_bgcolor='rgba(0,0,0,0)', hovermode='x unified')
            st.plotly_chart(fig_cmp, use_container_width=True)

            # === GIẢI THÍCH BIỂU ĐỒ ===
            with st.expander("📖 **Cách đọc biểu đồ này**", expanded=True):
                st.markdown(f"""
                Biểu đồ chia thành **2 phần** — phần TRÊN là Valence, phần DƯỚI là Arousal.

                **🔵 Phần TRÊN — Valence (đường liền nét)**
                - Thể hiện mức độ **tích cực/tiêu cực** của bài nhạc theo thời gian
                - Đường **lên cao** (>0) = bài đang VUI hơn 😊
                - Đường **xuống thấp** (<0) = bài đang BUỒN hơn 😢
                - Đường ngang **0** = ranh giới giữa vui và buồn

                **🔴 Phần DƯỚI — Arousal (đường nét đứt)**
                - Thể hiện mức độ **năng lượng** của bài nhạc theo thời gian
                - Đường **lên cao** (>0) = bài đang SÔI ĐỘNG hơn ⚡
                - Đường **xuống thấp** (<0) = bài đang YÊN TĨNH hơn 🌙

                **🎨 Mỗi màu = 1 bài nhạc**
                - {n_songs} bài được phân biệt bằng {n_songs} màu khác nhau
                - Cùng màu xuất hiện ở cả phần trên và dưới (Valence + Arousal của cùng 1 bài)

                **💡 Cách so sánh giữa các bài:**
                - **Các đường đi cùng xu hướng** → Các bài có cảm xúc TƯƠNG TỰ
                - **Các đường ngược nhau** → Các bài có cảm xúc TRÁI NGƯỢC
                - **Khoảng cách giữa các đường** = mức độ KHÁC BIỆT giữa các bài
                """)

            # === BẢNG SO SÁNH TỔNG HỢP ===
            st.markdown("### 📋 Bảng tổng hợp")
            cmp_df = pd.DataFrame({
                "STT": [f"Bài {i+1}" for i in range(len(results))],
                "Tên file": [r['name'][:40] + "..." if len(r['name']) > 40 else r['name'] for r in results],
                "Thời lượng": [f"{r['duration']:.1f}s" for r in results],
                "Valence TB": [f"{r['avg_v']:+.3f}" for r in results],
                "Arousal TB": [f"{r['avg_a']:+.3f}" for r in results],
                "Cảm xúc chủ đạo": [f"{MOOD_EMOJIS[quadrant(r['avg_v'], r['avg_a'])]} "
                                     f"{MOOD_VI[quadrant(r['avg_v'], r['avg_a'])]}"
                                     for r in results],
            })
            st.dataframe(cmp_df, use_container_width=True, hide_index=True)

            # === MA TRẬN ĐỘ TƯƠNG ĐỒNG ===
            st.markdown("### 🤝 Ma trận độ tương đồng cảm xúc")

            # Tính ma trận tương đồng từng cặp
            n = len(results)
            sim_matrix    = np.zeros((n, n))
            corr_v_matrix = np.zeros((n, n))
            corr_a_matrix = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    if i == j:
                        sim_matrix[i, j]    = 1.0
                        corr_v_matrix[i, j] = 1.0
                        corr_a_matrix[i, j] = 1.0
                    else:
                        min_len = min(len(results[i]['v']), len(results[j]['v']))
                        cv = np.corrcoef(results[i]['v'][:min_len], results[j]['v'][:min_len])[0, 1]
                        ca = np.corrcoef(results[i]['a'][:min_len], results[j]['a'][:min_len])[0, 1]
                        corr_v_matrix[i, j] = cv if not np.isnan(cv) else 0
                        corr_a_matrix[i, j] = ca if not np.isnan(ca) else 0
                        sim_matrix[i, j]    = (corr_v_matrix[i, j] + corr_a_matrix[i, j]) / 2

            # Hiển thị heatmap
            labels_short = [f"Bài {i+1}" for i in range(n)]
            fig_sim = go.Figure(data=go.Heatmap(
                z=sim_matrix * 100,
                x=labels_short, y=labels_short,
                colorscale='RdYlGn',
                zmin=-100, zmax=100,
                text=[[f"{v*100:.0f}%" for v in row] for row in sim_matrix],
                texttemplate="%{text}",
                textfont={"size": 14},
                colorbar=dict(title="Tương đồng (%)"),
                hovertemplate='<b>%{x} ↔ %{y}</b><br>Tương đồng: %{z:.1f}%<extra></extra>'
            ))
            fig_sim.update_layout(
                template="plotly_white", height=400,
                title="Ma trận độ tương đồng cảm xúc giữa các bài (%)",
                paper_bgcolor='rgba(0,0,0,0)',
            )
            st.plotly_chart(fig_sim, use_container_width=True)

            # === GIẢI THÍCH ĐỘ TƯƠNG ĐỒNG ===
            with st.expander("🧮 **Vì sao có độ tương đồng X%?** (Giải thích cách tính)", expanded=True):
                if n == 2:
                    # Trường hợp đặc biệt 2 bài
                    cv = corr_v_matrix[0, 1]
                    ca = corr_a_matrix[0, 1]
                    sim = sim_matrix[0, 1]

                    st.markdown(f"""
                    #### 📐 Công thức tính độ tương đồng

                    **Độ tương đồng** = trung bình cộng của 2 hệ số tương quan Pearson:

                    1. **Tương quan Valence** = `{cv:.3f}` (giữa 2 đường Valence của Bài 1 và Bài 2)
                    2. **Tương quan Arousal** = `{ca:.3f}` (giữa 2 đường Arousal của Bài 1 và Bài 2)

                    **→ Độ tương đồng = ({cv:.3f} + {ca:.3f}) / 2 = {sim:.3f} = {sim*100:.1f}%**

                    #### 🎯 Ý nghĩa hệ số Pearson Correlation

                    | Hệ số | Ý nghĩa | Ví dụ |
                    |---|---|---|
                    | **+1.0 (+100%)** | HOÀN TOÀN GIỐNG | Cả 2 bài cùng vui lên cùng buồn xuống |
                    | **+0.7 (+70%)** | Rất giống | Xu hướng tương tự, vài chỗ lệch |
                    | **+0.5 (+50%)** | Khá giống | Có nét chung nhưng cũng khác biệt |
                    | **0.0 (0%)** | Không liên quan | Hoàn toàn khác nhau |
                    | **-0.5 (-50%)** | Hơi NGƯỢC | Khi bài 1 vui thì bài 2 buồn |
                    | **-1.0 (-100%)** | HOÀN TOÀN NGƯỢC | 1 lên thì 1 xuống y hệt |

                    #### 💡 Kết luận cho 2 bài này:
                    """)

                    if sim > 0.7:
                        st.success(f"✅ **Cảm xúc RẤT GIỐNG NHAU ({sim*100:.1f}%)** — 2 bài có cùng kiểu thay đổi cảm xúc theo thời gian. Có thể cùng thể loại, cùng tâm trạng.")
                    elif sim > 0.4:
                        st.info(f"🟡 **Cảm xúc KHÁ GIỐNG ({sim*100:.1f}%)** — Có nhiều điểm chung nhưng cũng có khác biệt rõ rệt.")
                    elif sim > 0.0:
                        st.warning(f"🟠 **Cảm xúc CÓ CHÚT GIỐNG ({sim*100:.1f}%)** — Hơi tương đồng nhưng phần lớn khác.")
                    elif sim > -0.4:
                        st.warning(f"⚠️ **Cảm xúc KHÁC NHAU ({sim*100:.1f}%)** — 2 bài có cảm xúc khác biệt rõ rệt.")
                    else:
                        st.error(f"🔴 **Cảm xúc HOÀN TOÀN NGƯỢC ({sim*100:.1f}%)** — Khi 1 bài vui thì bài kia buồn.")
                else:
                    # Trường hợp N > 2: ma trận
                    st.markdown(f"""
                    #### 📐 Cách tính độ tương đồng giữa 2 bài bất kỳ

                    Với **mỗi cặp bài (i, j)**, độ tương đồng được tính bằng 3 bước:

                    1. **Tính tương quan Pearson** của Valence theo thời gian → giá trị từ -1 đến +1
                    2. **Tính tương quan Pearson** của Arousal theo thời gian → giá trị từ -1 đến +1
                    3. **Độ tương đồng = (tương quan V + tương quan A) / 2** → đổi ra phần trăm

                    #### 📊 Cách đọc ma trận heatmap ở trên

                    - **Mỗi ô (i, j)** = % tương đồng giữa Bài i và Bài j
                    - **Đường chéo** = 100% (mỗi bài giống chính nó)
                    - 🟢 **Màu XANH ĐẬM** = Rất giống (>70%)
                    - 🟡 **Màu VÀNG** = Trung bình (30-70%)
                    - 🔴 **Màu ĐỎ** = Khác nhau hoặc ngược nhau (<0%)

                    #### 🎯 Phân tích các cặp:
                    """)

                    # Tìm cặp giống nhất và khác nhất
                    triu = np.triu(sim_matrix, k=1)  # Upper triangular
                    if triu.max() > -2:
                        most_sim_idx = np.unravel_index(triu.argmax(), triu.shape)
                    else:
                        most_sim_idx = (0, 1)

                    # Tìm min trong upper triangular (set diagonal/lower về giá trị lớn)
                    triu_for_min = np.copy(sim_matrix)
                    for i in range(n):
                        for j in range(i + 1):
                            triu_for_min[i, j] = 999
                    most_diff_idx = np.unravel_index(triu_for_min.argmin(), triu_for_min.shape)

                    st.markdown(f"""
                    - 🏆 **Cặp giống nhau NHẤT**: Bài {most_sim_idx[0]+1} ↔ Bài {most_sim_idx[1]+1} (**{sim_matrix[most_sim_idx]*100:.1f}%**)
                    - 🔀 **Cặp khác nhau NHẤT**: Bài {most_diff_idx[0]+1} ↔ Bài {most_diff_idx[1]+1} (**{sim_matrix[most_diff_idx]*100:.1f}%**)
                    """)

                    # Bảng tương quan chi tiết
                    st.markdown("##### 📑 Bảng tương quan chi tiết:")
                    pair_rows = []
                    for i in range(n):
                        for j in range(i + 1, n):
                            pair_rows.append({
                                "Cặp": f"Bài {i+1} ↔ Bài {j+1}",
                                "Tương quan V": f"{corr_v_matrix[i,j]:.3f}",
                                "Tương quan A": f"{corr_a_matrix[i,j]:.3f}",
                                "Tương đồng": f"{sim_matrix[i,j]*100:.1f}%",
                            })
                    st.dataframe(pd.DataFrame(pair_rows), use_container_width=True, hide_index=True)

            # === BUTTONS DOWNLOAD ===
            st.markdown("### 💾 Tải báo cáo so sánh")
            dc1, dc2, dc3 = st.columns(3)

            # CSV: time series chi tiết của tất cả bài
            with dc1:
                rows = []
                for i, r in enumerate(results):
                    for t_idx, t in enumerate(r['times']):
                        rows.append({
                            "bai":            f"Bài {i+1}",
                            "ten_file":       r['name'],
                            "thoi_gian_giay": float(t),
                            "valence":        float(r['v'][t_idx]),
                            "arousal":        float(r['a'][t_idx]),
                            "cam_xuc":        r['moods'][t_idx],
                        })
                csv_data = pd.DataFrame(rows).to_csv(index=False)
                st.download_button(
                    "📄 Tải CSV chi tiết",
                    csv_data,
                    file_name=f"so_sanh_{n_songs}_bai_chi_tiet.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            # JSON: tổng hợp + ma trận tương đồng
            with dc2:
                json_data = json.dumps({
                    "so_bai":              n_songs,
                    "thoi_gian_phan_tich": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "model":               selected_model,
                    "ket_qua_tung_bai": [{
                        "stt":             i + 1,
                        "ten_file":        r['name'],
                        "thoi_luong_giay": r['duration'],
                        "valence_tb":      r['avg_v'],
                        "arousal_tb":      r['avg_a'],
                        "cam_xuc_chu_dao": MOOD_VI[quadrant(r['avg_v'], r['avg_a'])],
                    } for i, r in enumerate(results)],
                    "ma_tran_tuong_dong_phan_tram": [
                        [round(float(sim_matrix[i, j] * 100), 1) for j in range(n)]
                        for i in range(n)
                    ],
                    "ma_tran_tuong_quan_valence": [
                        [round(float(corr_v_matrix[i, j]), 3) for j in range(n)]
                        for i in range(n)
                    ],
                    "ma_tran_tuong_quan_arousal": [
                        [round(float(corr_a_matrix[i, j]), 3) for j in range(n)]
                        for i in range(n)
                    ],
                }, indent=2, ensure_ascii=False)
                st.download_button(
                    "📊 Tải JSON tổng hợp",
                    json_data,
                    file_name=f"so_sanh_{n_songs}_bai_tong_hop.json",
                    mime="application/json",
                    use_container_width=True
                )

            # TXT: báo cáo dạng văn bản
            with dc3:
                report = f"""BÁO CÁO SO SÁNH {n_songs} BÀI NHẠC
====================================================
Mô hình:   {selected_model}
Thời gian: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Số bài:    {n_songs}

THÔNG TIN CHI TIẾT TỪNG BÀI:
"""
                for i, r in enumerate(results):
                    mood = quadrant(r['avg_v'], r['avg_a'])
                    report += f"""
─── Bài {i+1} ────────────────────────────
File:            {r['name']}
Thời lượng:      {r['duration']:.1f}s
Valence TB:      {r['avg_v']:+.3f}
Arousal TB:      {r['avg_a']:+.3f}
Cảm xúc chủ đạo: {MOOD_VI[mood]}
"""
                report += "\n\nMA TRẬN ĐỘ TƯƠNG ĐỒNG (%):\n"
                report += "          " + "  ".join([f"Bài {i+1:2d}" for i in range(n)]) + "\n"
                for i in range(n):
                    report += f"Bài {i+1:2d}:    " + "  ".join([f"{sim_matrix[i,j]*100:5.1f}" for j in range(n)]) + "\n"

                report += f"""

GIẢI THÍCH CÔNG THỨC TÍNH ĐỘ TƯƠNG ĐỒNG:
─────────────────────────────────────────
Với mỗi cặp bài (i, j):
  1. Tính tương quan Pearson của Valence theo thời gian → r_V
  2. Tính tương quan Pearson của Arousal theo thời gian → r_A
  3. Độ tương đồng = (r_V + r_A) / 2

Ý NGHĨA:
  +100% : 2 bài có cảm xúc HOÀN TOÀN GIỐNG NHAU
  +70%  : Rất giống
  +50%  : Khá giống
   0%   : Không liên quan
  -50%  : Hơi NGƯỢC nhau
  -100% : HOÀN TOÀN NGƯỢC nhau (1 vui thì 1 buồn)
"""
                st.download_button(
                    "📝 Tải báo cáo TXT",
                    report,
                    file_name=f"bao_cao_so_sanh_{n_songs}_bai.txt",
                    mime="text/plain",
                    use_container_width=True
                )

        except Exception as e:
            st.error(f"❌ Lỗi: {e}")
            with st.expander("Chi tiết lỗi"):
                st.code(traceback.format_exc())


# =============================================================================
# TAB 3 — KHÁM PHÁ MEL-SPECTROGRAM
# =============================================================================
with tab3:
    st.markdown("### 🔍 Khám phá Mel-Spectrogram")
    st.caption("Xem đặc trưng âm thanh mà mô hình AI dùng để dự đoán cảm xúc.")

    # === GIẢI THÍCH MỤC ĐÍCH CỦA TAB NÀY ===
    with st.expander("❓ **Tab này dùng để làm gì?**", expanded=True):
        st.markdown("""
        ### 🎯 Mục đích

        Tab này giúp bạn **"nhìn thấy"** âm thanh — vốn là thứ chỉ có thể nghe.
        Bạn sẽ thấy được **AI thực sự xử lý gì** khi nghe bài nhạc của bạn.

        ### 💡 4 chức năng chính

        | # | Chức năng | Tác dụng |
        |---|---|---|
        | 1 | **Dạng sóng (Waveform)** | Xem biên độ âm thanh — chỗ nào to, chỗ nào nhỏ |
        | 2 | **Mel-Spectrogram** | Xem "ảnh chụp" tần số âm thanh — đây là **input thực tế của AI** |
        | 3 | **Thống kê đặc trưng** | Đo các chỉ số khoa học: nhịp độ, độ sáng, độ ồn... |
        | 4 | **Hiểu cách AI nhìn nhạc** | Tham khảo trước khi dùng tab "Phân tích bài nhạc" |

        ### 📌 Khác biệt với Tab "Phân tích bài nhạc"?

        - **Tab Phân tích**: AI **dự đoán cảm xúc** (Vui/Buồn/Sôi động/Yên tĩnh)
        - **Tab này**: Hiển thị **đặc trưng kỹ thuật** mà AI dùng để đưa ra dự đoán đó
        - 💡 Có thể coi đây là **"phía sau hậu trường"** của AI
        """)

    mel_file = st.file_uploader("📤 Tải lên file để khám phá",
                                  type=["mp3", "wav"], key="mel_upload")

    if mel_file is not None:
        tmp = Path(tempfile.gettempdir()) / mel_file.name
        tmp.write_bytes(mel_file.read())
        st.audio(str(tmp))

        with st.spinner("Đang trích xuất mel-spectrogram..."):
            y, sr = librosa.load(tmp, sr=SR, mono=True)
            duration = len(y) / sr

            mel_full = librosa.feature.melspectrogram(
                y=y, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH,
                n_mels=N_MELS, fmin=FMIN, fmax=FMAX, power=2.0)
            mel_db = librosa.power_to_db(mel_full, ref=np.max)

        # ===========================================================
        # 1. DẠNG SÓNG (WAVEFORM)
        # ===========================================================
        st.markdown("#### 〰️ 1. Dạng sóng (Waveform)")
        st.caption("Biên độ âm thanh theo thời gian — chỗ nào sóng cao = nhạc to/mạnh, sóng thấp = nhạc nhỏ/lặng")
        t_wave = np.arange(len(y)) / sr
        step = max(1, len(y) // 5000)
        fig_wave = go.Figure(go.Scatter(x=t_wave[::step], y=y[::step],
                                          line=dict(color="#4facfe", width=1)))
        fig_wave.update_layout(template="plotly_white", height=200,
                                xaxis_title="Thời gian (s)", yaxis_title="Biên độ",
                                paper_bgcolor='rgba(0,0,0,0)',
                                margin=dict(t=20, b=40))
        st.plotly_chart(fig_wave, use_container_width=True)

        with st.expander("ℹ️ Hiểu về Dạng sóng (Waveform)", expanded=True):
            st.markdown("""
            **Waveform là gì?**
            - Là biểu đồ thể hiện **độ to/nhỏ** của âm thanh theo từng khoảnh khắc
            - Trục NGANG = thời gian (giây)
            - Trục DỌC = biên độ (sóng âm thanh)

            **Cách đọc đơn giản:**
            - 🔊 Sóng dày, cao → đoạn nhạc TO, MẠNH (chorus, drop)
            - 🔉 Sóng mỏng, thấp → đoạn nhạc NHỎ, LẶNG (intro, outro)
            - 🔇 Đường thẳng tại 0 → IM LẶNG hoàn toàn

            **Hạn chế của Waveform:**
            - Chỉ thấy ĐỘ TO, không thấy được TẦN SỐ (cao/trầm)
            - → Cần Mel-Spectrogram (bên dưới) để xem chi tiết hơn
            """)

        # ===========================================================
        # 2. MEL-SPECTROGRAM (THANG LOG)
        # ===========================================================
        st.markdown("#### 🌈 2. Mel-Spectrogram (thang Log)")
        st.caption("\"Ảnh chụp\" của âm thanh — đây chính là input mà AI nhận để dự đoán cảm xúc")
        fig_mel = go.Figure(data=go.Heatmap(
            z=mel_db,
            x=np.linspace(0, duration, mel_db.shape[1]),
            y=np.linspace(FMIN, FMAX, N_MELS),
            colorscale='Magma',
            colorbar=dict(title="dB"),
        ))
        fig_mel.update_layout(template="plotly_white", height=400,
                               xaxis_title="Thời gian (s)", yaxis_title="Tần số Mel (Hz)",
                               paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_mel, use_container_width=True)

        with st.expander("ℹ️ **Mel-Spectrogram là gì?** (Quan trọng — đọc kỹ)", expanded=True):
            st.markdown("""
            ### 📖 Định nghĩa đơn giản

            Mel-Spectrogram là cách **biến âm thanh thành một bức tranh 2D** mà máy tính có thể "nhìn".

            > 💡 Tưởng tượng: Bạn không thấy được giọng người, nhưng nếu chuyển giọng nói thành sóng âm
            > rồi vẽ ra → bạn có thể "đọc" giọng người qua hình ảnh. Đó chính là Mel-Spectrogram.

            ### 🎨 Cách đọc

            - **Trục NGANG** = Thời gian (giây) — giống như video tua từ trái sang phải
            - **Trục DỌC** = Tần số (Hz) — thấp ở dưới (bass, trống), cao ở trên (chũm, hi-hat)
            - **Màu sắc** = Cường độ tại tần số đó
              - 🟡 **Vàng/cam (sáng)** = Tần số đó MẠNH (rõ, to)
              - 🟣 **Tím/đen (tối)** = Tần số đó YẾU (im, không có)

            ### 🎵 Áp dụng cho âm nhạc

            | Vùng tần số | Loại âm thanh điển hình |
            |---|---|
            | **0-200 Hz** (dưới cùng) | Trống bass, kick drum |
            | **200-2000 Hz** (giữa) | Giọng hát, guitar, piano |
            | **2000-5000 Hz** (trên giữa) | Snare, vocals chi tiết |
            | **5000-11000 Hz** (trên cùng) | Hi-hat, cymbal, tiếng s/sh |

            ### 🎯 Vì sao dùng "thang Log" và "thang Mel"?

            - **Log scale** (decibel - dB): Tai người không cảm nhận to/nhỏ tuyến tính.
              Tai chuyển 10x to → ta cảm 2x → cần thang log để đúng cách tai nghe.
            - **Mel scale**: Tai người cảm nhận tần số không tuyến tính.
              Khoảng 100Hz → 200Hz nghe khác hẳn, nhưng 7000Hz → 7100Hz gần như giống.
              Thang Mel mô phỏng cách tai người cảm nhận.

            ### 🤖 Vai trò của Mel-Spectrogram trong AI

            Đề tài này dùng **CNN + BiLSTM + Attention** xử lý Mel-Spectrogram như một bức ảnh:
            1. **CNN** quét bức ảnh, tìm các "pattern" (ví dụ: vạch sáng = nốt nhạc)
            2. **BiLSTM** ghép các pattern theo thời gian thành "câu chuyện cảm xúc"
            3. **Attention** chọn đoạn nào quan trọng nhất để dự đoán

            → Mel-Spectrogram là **mắt** của AI, không có nó AI không thể "nhìn" nhạc!

            ### 💡 Mẹo "đọc" bài nhạc qua Mel-Spectrogram

            - **Vạch ngang sáng dài** → có nốt giữ lâu (string, pad synth)
            - **Vạch dọc sáng** → có cú đánh nhanh (drum, percussion)
            - **Cả ảnh sáng đều** → bài đầy đủ, dày dặn (full mix)
            - **Ảnh tối, thưa** → bài đơn giản, ít nhạc cụ (acoustic, piano solo)
            - **Vùng dưới (bass) sáng** → bài có tempo mạnh, drum bass
            - **Vùng trên (treble) sáng** → bài có nhiều chi tiết âm cao
            """)

        # ===========================================================
        # 3. THỐNG KÊ ĐẶC TRƯNG ÂM THANH
        # ===========================================================
        st.markdown("#### 📊 3. Thống kê đặc trưng âm thanh")
        st.caption("4 chỉ số khoa học mô tả đặc tính của bài nhạc")
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
                <div style="color:#5a6c7d">BPM</div>
            </div>""", unsafe_allow_html=True)
        with sc2:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Trọng tâm phổ</div>
                <div class="metric-value">{spec_centroid:.0f}</div>
                <div style="color:#5a6c7d">Hz</div>
            </div>""", unsafe_allow_html=True)
        with sc3:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Tỷ lệ qua 0</div>
                <div class="metric-value">{zero_crossing:.3f}</div>
            </div>""", unsafe_allow_html=True)
        with sc4:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Năng lượng RMS</div>
                <div class="metric-value">{rms_energy:.3f}</div>
            </div>""", unsafe_allow_html=True)

        # === GIẢI THÍCH 4 CHỈ SỐ + PHÂN TÍCH BÀI HIỆN TẠI ===
        with st.expander("ℹ️ **Hiểu 4 chỉ số này** (cho người không chuyên)", expanded=True):
            # Phân loại tempo
            if tempo_val < 60:        tempo_desc = "**RẤT CHẬM** (Largo) — Như nhạc thiền, ballad chậm"
            elif tempo_val < 80:      tempo_desc = "**CHẬM** (Adagio) — Như slow ballad, lo-fi"
            elif tempo_val < 100:     tempo_desc = "**VỪA** (Andante) — Như pop ballad"
            elif tempo_val < 120:     tempo_desc = "**TRUNG BÌNH** (Moderato) — Như pop trung bình"
            elif tempo_val < 140:     tempo_desc = "**NHANH** (Allegro) — Như pop sôi động, rock"
            elif tempo_val < 160:     tempo_desc = "**KHÁ NHANH** (Vivace) — Như EDM, dance"
            else:                      tempo_desc = "**RẤT NHANH** (Presto) — Như drum & bass, hardcore"

            # Phân loại spectral centroid
            if spec_centroid < 1500:   sc_desc = "**TRẦM** — Bài nhiều bass, nốt thấp (jazz, blues, hip-hop)"
            elif spec_centroid < 2500: sc_desc = "**CÂN BẰNG** — Pop bình thường, có cả bass và treble"
            elif spec_centroid < 4000: sc_desc = "**SÁNG** — Có nhiều âm cao (synth, treble rõ)"
            else:                       sc_desc = "**RẤT SÁNG** — Tập trung âm cao (cymbal, hi-hat, vocal cao)"

            # Phân loại ZCR
            if zero_crossing < 0.05:   zcr_desc = "**MƯỢT, TONAL** — Nhạc cụ giữ note (string, vocal)"
            elif zero_crossing < 0.1:  zcr_desc = "**TRUNG BÌNH** — Mix vocal + nhạc cụ"
            else:                       zcr_desc = "**NHIỄU CAO** — Có percussion, hi-hat, hoặc tiếng s/sh"

            # Phân loại RMS
            if rms_energy < 0.05:     rms_desc = "**RẤT NHỎ** — Đoạn yên tĩnh, intro nhẹ"
            elif rms_energy < 0.1:    rms_desc = "**NHỎ** — Acoustic, ballad chậm"
            elif rms_energy < 0.2:    rms_desc = "**TRUNG BÌNH** — Pop bình thường"
            else:                      rms_desc = "**LỚN** — Rock, EDM, có drop mạnh"

            st.markdown(f"""
            ### 🎵 1. Tempo (Nhịp độ): **{tempo_val:.0f} BPM**

            **Định nghĩa:** Số nhịp trên phút (Beats Per Minute) — đo "tốc độ" bài nhạc.

            **Bài của bạn:** {tempo_desc}

            **Tham chiếu:**
            - 60-70 BPM: Nhạc ru, nhạc thiền
            - 90-110 BPM: Pop ballad, R&B
            - 120-130 BPM: Pop, House music
            - 130-150 BPM: Rock, Dance
            - 150+ BPM: EDM, Drum & Bass

            **Liên quan đến cảm xúc:** Tempo cao thường = Arousal cao (sôi động).

            ---

            ### 🎵 2. Trọng tâm phổ (Spectral Centroid): **{spec_centroid:.0f} Hz**

            **Định nghĩa:** "Trọng tâm" của âm thanh trên trục tần số — bài nhạc thiên về âm trầm hay âm cao.

            **Bài của bạn:** {sc_desc}

            **Tham chiếu:**
            - **< 1500 Hz** = Âm TRẦM (bass, nam giọng trầm)
            - **1500-3000 Hz** = Cân bằng (pop trung bình)
            - **> 3000 Hz** = Âm CAO (treble, nữ giọng cao)

            **Liên quan đến cảm xúc:** Bài "sáng" thường vui hơn (Valence cao), "trầm" thường buồn hơn.

            ---

            ### 🎵 3. Tỷ lệ qua 0 (Zero Crossing Rate): **{zero_crossing:.3f}**

            **Định nghĩa:** Tốc độ sóng âm cắt qua đường 0 — đo "độ nhiễu" hay tonal của âm thanh.

            **Bài của bạn:** {zcr_desc}

            **Tham chiếu:**
            - **< 0.05** = MƯỢT (giọng hát giữ note, violin, piano)
            - **0.05-0.1** = TRUNG BÌNH (mix có nhạc cụ + vocal)
            - **> 0.1** = NHIỄU (drum, percussion, tiếng s/sh trong giọng hát)

            **Liên quan đến cảm xúc:** ZCR cao thường liên quan đến Arousal cao (có drum mạnh).

            ---

            ### 🎵 4. Năng lượng RMS (RMS Energy): **{rms_energy:.3f}**

            **Định nghĩa:** Mức năng lượng trung bình (cảm nhận như "to" của bài) — RMS = Root Mean Square.

            **Bài của bạn:** {rms_desc}

            **Tham chiếu:**
            - **< 0.05** = Rất nhỏ (ambient, intro yên tĩnh)
            - **0.05-0.1** = Nhỏ (acoustic, ballad)
            - **0.1-0.2** = Trung bình (pop chuẩn)
            - **> 0.2** = Lớn (rock, EDM, có "wall of sound")

            **Liên quan đến cảm xúc:** RMS cao = nhiều năng lượng = Arousal cao.
            """)

        # === KẾT LUẬN TỔNG HỢP CHO BÀI ===
        with st.expander("🎯 **Phân tích tổng hợp bài này** (kết hợp 4 chỉ số)", expanded=True):
            # Tổng hợp dự đoán cảm xúc dựa trên 4 chỉ số
            arousal_score = 0
            arousal_score += 1 if tempo_val > 110 else (-1 if tempo_val < 80 else 0)
            arousal_score += 1 if rms_energy > 0.15 else (-1 if rms_energy < 0.07 else 0)
            arousal_score += 1 if zero_crossing > 0.08 else (-1 if zero_crossing < 0.04 else 0)

            valence_score = 0
            valence_score += 1 if spec_centroid > 2500 else (-1 if spec_centroid < 1500 else 0)

            if arousal_score >= 2:    arousal_label = "CAO (sôi động)"
            elif arousal_score <= -2: arousal_label = "THẤP (yên tĩnh)"
            else:                      arousal_label = "TRUNG BÌNH"

            if valence_score >= 1:    valence_label = "TÍCH CỰC (vui, sáng)"
            elif valence_score <= -1: valence_label = "TIÊU CỰC (buồn, trầm)"
            else:                      valence_label = "TRUNG TÍNH"

            st.markdown(f"""
            #### 📋 Tóm tắt đặc trưng kỹ thuật

            Dựa trên **4 chỉ số khoa học**, bài nhạc của bạn có đặc trưng:

            - 🎵 **Nhịp độ**: {tempo_val:.0f} BPM
            - 🌈 **Tính chất phổ**: {sc_desc}
            - 🎚️ **Mức độ ồn**: {rms_desc}
            - 🎼 **Độ mượt sóng**: {zcr_desc}

            #### 🔮 Dự đoán cảm xúc sơ bộ (chỉ dựa trên 4 chỉ số trên)

            - **Năng lượng (Arousal)**: {arousal_label}
            - **Tính tích cực (Valence)**: {valence_label}

            > ⚠️ **Lưu ý**: Đây chỉ là phân tích sơ bộ dựa trên các chỉ số "thủ công".
            > Để có kết quả chính xác hơn, hãy dùng tab **"🎼 Phân tích bài nhạc"** —
            > nơi mô hình AI deep learning sẽ phân tích chi tiết theo từng giây.
            """)


# =============================================================================
# TAB 4 — LỊCH SỬ
# =============================================================================
with tab4:
    st.markdown("### 📚 Lịch sử phân tích")

    if not st.session_state.history:
        st.info("Chưa có phân tích nào. Quay lại tab 'Phân tích bài nhạc' để bắt đầu!")
    else:
        st.caption(f"Đã phân tích {len(st.session_state.history)} bài nhạc trong phiên này.")

        all_v = [r['avg_v'] for r in st.session_state.history]
        all_a = [r['avg_a'] for r in st.session_state.history]
        all_moods = [r['dominant_mood'] for r in st.session_state.history]

        sh1, sh2, sh3 = st.columns(3)
        with sh1:
            st.metric("Tổng số bài", len(st.session_state.history))
        with sh2:
            st.metric("V trung bình tất cả", f"{np.mean(all_v):+.3f}")
        with sh3:
            most_common = Counter(all_moods).most_common(1)[0][0]
            st.metric("Cảm xúc phổ biến nhất",
                      MOOD_EMOJIS[most_common] + " " + MOOD_VI[most_common])

        st.markdown("#### 📜 Danh sách")
        hist_df = pd.DataFrame([
            {
                "STT": i+1,
                "File": r['filename'],
                "Mô hình": r['model'],
                "Cảm xúc": f"{MOOD_EMOJIS[r['dominant_mood']]} {MOOD_VI[r['dominant_mood']]}",
                "V_TB": f"{r['avg_v']:+.2f}",
                "A_TB": f"{r['avg_a']:+.2f}",
                "Thời gian": r['timestamp'],
            }
            for i, r in enumerate(st.session_state.history)
        ])
        st.dataframe(hist_df, use_container_width=True, hide_index=True)

        st.markdown("#### 📊 Phân bố cảm xúc của tất cả các bài")
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
                                yaxis_title="Số bài")
        st.plotly_chart(fig_hist, use_container_width=True)

        st.markdown("#### 🎯 Tất cả bài trên bản đồ V-A")
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
# TAB 5 — GIỚI THIỆU
# =============================================================================
with tab5:
    st.markdown("""
    ### ℹ️ Giới thiệu hệ thống

    #### 🎯 Mục tiêu
    Phân tích **diễn biến cảm xúc theo thời gian** trong âm nhạc, dự đoán **Valence** (tích cực/tiêu cực)
    và **Arousal** (năng lượng) cho mỗi cửa sổ 0.5 giây.

    #### 🏗️ Kiến trúc
    - **CNN encoder**: trích đặc trưng không gian từ Mel-spectrogram (3 lớp conv)
    - **BiLSTM**: mô hình mối quan hệ thời gian giữa các cửa sổ 0.5s
    - **Multi-head Attention**: bắt dependency dài hạn giữa các timestep
    - **Combined Loss + Class Weighting**: tối ưu trực tiếp metric đánh giá + xử lý imbalance

    #### 📊 Dữ liệu
    - **DEAM** (1802 bài nhạc phương Tây, đa thể loại) — train chính
    - **PMEmo** (~767 bài Chinese pop) — cross-dataset evaluation

    #### 📈 Kết quả
    | Metric | Giá trị | So với SOTA Aljanaki 2017 |
    |---|---|---|
    | CCC_V (DEAM) | 0.65 | +116% |
    | CCC_A (DEAM) | 0.77 | +24% |
    | CCC_V (PMEmo, transfer) | 0.69 | best in class |

    #### 🎨 Bốn vùng cảm xúc V-A (Russell 1980)
    """)

    cols = st.columns(4)
    quadrants_info = [
        ("happy/excited", "Vui vẻ / Hưng phấn", "V > 0, A > 0", "Pop, Dance, Disco"),
        ("tense/angry", "Căng thẳng / Tức giận", "V < 0, A > 0", "Heavy Metal, Punk"),
        ("sad", "Buồn bã", "V < 0, A < 0", "Blues, Slow ballad"),
        ("calm/relaxed", "Thư thái / Bình yên", "V > 0, A < 0", "Jazz, Classical, Lo-fi"),
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
    #### 🛠️ Công nghệ sử dụng
    - **Backend**: PyTorch + Librosa + NumPy
    - **Frontend**: Streamlit + Plotly
    - **Audio**: Mel-spectrogram (64 mel-bands, 22050 Hz, hop 256)

    #### 📚 Tài liệu tham khảo
    1. Aljanaki et al. (2017). *Developing a benchmark for emotional analysis of music*. PLoS ONE.
    2. Zhang et al. (2018). *The PMEmo Dataset for Music Emotion Recognition*. ICMR.
    3. Vaswani et al. (2017). *Attention is All You Need*. NeurIPS.

    #### 👨‍🎓 Tác giả
    Tăng Ngọc Phụng — KHMT836027 — Khoa học Máy tính Ứng dụng 
    """)


# =============================================================================
# FOOTER
# =============================================================================
st.markdown("""
<div class="footer">
    🎵 Music Emotion Recognition (MER) | Advanced Machine Learning |  Academic Supervisor: Dr. Ngo Quoc Viet
</div>
""", unsafe_allow_html=True)
