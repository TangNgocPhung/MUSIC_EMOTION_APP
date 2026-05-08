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
AVAILABLE_CHECKPOINTS = {
    "⭐ Mô hình đề xuất chính (Attention + Combined Loss + Balanced)": "best_attention_balanced.pt",
    "Mô hình Attention thuần (CCC Loss)":                              "best_attention.pt",
    "Mô hình Baseline (CNN + BiLSTM)":                                  "best.pt",
    "Mô hình Transfer Learning (DEAM → PMEmo)":                         "best_pmemo_ft_head.pt",
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

    # Mô tả chi tiết cho 4 models
    model_info = {
        "⭐ Mô hình đề xuất chính (Attention + Combined Loss + Balanced)":
            ("CNN + BiLSTM + Multi-head Attention",
             "Combined Loss + Class Weighting",
             "DEAM (1802 bài)",
             "Mô hình cuối cùng, xử lý class imbalance"),
        "Mô hình Attention thuần (CCC Loss)":
            ("CNN + BiLSTM + Multi-head Attention",
             "CCC Loss",
             "DEAM (1802 bài)",
             "Chứng minh đóng góp của Attention"),
        "Mô hình Baseline (CNN + BiLSTM)":
            ("CNN + BiLSTM",
             "SmoothL1 Loss",
             "DEAM (1802 bài)",
             "Mô hình tham chiếu cơ bản"),
        "Mô hình Transfer Learning (DEAM → PMEmo)":
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
        🎓 Master Thesis Demo<br>
        Built with Streamlit + PyTorch
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# 8. MAIN TABS
# =============================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎼 Phân tích bài nhạc",
    "⚖️ So sánh 2 bài",
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

        st.markdown("### 📉 Diễn biến V-A theo thời gian")
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

        col_a, col_b = st.columns([2, 1])

        with col_a:
            st.markdown("#### 🎨 Bản đồ cảm xúc theo thời gian")
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

        with col_b:
            st.markdown("#### 🎯 Quỹ đạo V-A")
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

        col_c, col_d = st.columns([1, 1])

        with col_c:
            st.markdown("#### 🥧 Phân bố cảm xúc")
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
            st.markdown("#### 📋 Bảng các đoạn cảm xúc")
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
# TAB 2 — SO SÁNH 2 BÀI NHẠC
# =============================================================================
with tab2:
    st.markdown("### ⚖️ So sánh 2 bài nhạc")
    st.caption("Tải lên 2 file để so sánh diễn biến cảm xúc song song.")

    cc1, cc2 = st.columns(2)
    with cc1:
        f1 = st.file_uploader("🎵 Bài 1", type=["mp3", "wav"], key="cmp1")
    with cc2:
        f2 = st.file_uploader("🎵 Bài 2", type=["mp3", "wav"], key="cmp2")

    if f1 and f2 and st.button("🔍 So sánh", type="primary"):
        try:
            with st.spinner("Đang phân tích 2 bài..."):
                model, dev = load_model(ckpt_file)
                results = []
                for f in [f1, f2]:
                    tmp = Path(tempfile.gettempdir()) / f.name
                    tmp.write_bytes(f.read())
                    y, total, times, pr = predict(tmp, model, dev, DEFAULT_SMOOTHING)
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
                subplot_titles=("So sánh Valence", "So sánh Arousal"),
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
                "Bài": [r['name'] for r in results],
                "Valence TB": [f"{r['avg_v']:+.3f}" for r in results],
                "Arousal TB": [f"{r['avg_a']:+.3f}" for r in results],
                "Cảm xúc chủ đạo": [f"{MOOD_EMOJIS[quadrant(r['avg_v'], r['avg_a'])]} "
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
                🤝 <b>Độ tương đồng cảm xúc</b>: {sim*100:.1f}%
                (Tương quan Valence: {corr_v:.3f}, Tương quan Arousal: {corr_a:.3f})
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Lỗi: {e}")


# =============================================================================
# TAB 3 — KHÁM PHÁ MEL-SPECTROGRAM
# =============================================================================
with tab3:
    st.markdown("### 🔍 Khám phá Mel-Spectrogram")
    st.caption("Xem đặc trưng âm thanh mà mô hình dùng để dự đoán cảm xúc.")

    mel_file = st.file_uploader("Tải lên file để xem mel-spectrogram",
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

        st.markdown("#### 〰️ Dạng sóng (Waveform)")
        t_wave = np.arange(len(y)) / sr
        step = max(1, len(y) // 5000)
        fig_wave = go.Figure(go.Scatter(x=t_wave[::step], y=y[::step],
                                          line=dict(color="#4facfe", width=1)))
        fig_wave.update_layout(template="plotly_white", height=200,
                                xaxis_title="Thời gian (s)", yaxis_title="Biên độ",
                                paper_bgcolor='rgba(0,0,0,0)',
                                margin=dict(t=20, b=40))
        st.plotly_chart(fig_wave, use_container_width=True)

        st.markdown("#### 🌈 Mel-Spectrogram (thang Log)")
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

        st.markdown("#### 📊 Thống kê đặc trưng âm thanh")
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
    Phụng — Master Thesis — Khoa học Máy tính Ứng dụng
    """)


# =============================================================================
# FOOTER
# =============================================================================
st.markdown("""
<div class="footer">
    🎵 Phân tích Cảm xúc Âm nhạc | Master Thesis Demo |
    Built with Streamlit, PyTorch & ❤️
</div>
""", unsafe_allow_html=True)
