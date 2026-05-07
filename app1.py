
import os
from pathlib import Path
import numpy as np, librosa, torch, torch.nn as nn
import streamlit as st
import matplotlib.pyplot as plt
import tempfile
SR = 22050
CLIP_START_SEC, CLIP_END_SEC, WINDOW_SEC = 15.0, 45.0, 0.5
N_STEPS = int((CLIP_END_SEC - CLIP_START_SEC) / WINDOW_SEC)
SAMPLES_PER_WINDOW = int(SR * WINDOW_SEC)
N_FFT, HOP_LENGTH, N_MELS = 1024, 256, 64
FMIN, FMAX = 20, SR // 2
CNN_OUT_DIM, LSTM_HIDDEN, LSTM_LAYERS, LSTM_BIDIR, DROPOUT = 256, 128, 2, True, 0.3
LABEL_MIN, LABEL_MAX, LABEL_MEAN = -1.0, 1.0, 0.0
CKPT_PATH = Path("best.pt")

def denormalize_label(y): return y * ((LABEL_MAX - LABEL_MIN) / 2.0) + LABEL_MEAN

class CNNEncoder(nn.Module):
    def __init__(self, out_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(True), nn.MaxPool2d(2,2),
            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(True), nn.MaxPool2d(2,2),
            nn.Conv2d(64,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.AdaptiveAvgPool2d((4,4)))
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(128*4*4, out_dim),
                                 nn.ReLU(True), nn.Dropout(DROPOUT))
    def forward(self, x): return self.fc(self.conv(x))

class CNNLSTMEmotion(nn.Module):
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

def audio_to_mel_sequence(y):
    s, e = int(CLIP_START_SEC*SR), int(CLIP_END_SEC*SR)
    if len(y) < e: y = np.pad(y, (0, e-len(y)))
    y = y[s:e]
    mels = []
    for i in range(N_STEPS):
        chunk = y[i*SAMPLES_PER_WINDOW:(i+1)*SAMPLES_PER_WINDOW]
        m = librosa.feature.melspectrogram(y=chunk, sr=SR, n_fft=N_FFT,
            hop_length=HOP_LENGTH, n_mels=N_MELS, fmin=FMIN, fmax=FMAX, power=2.0)
        mels.append(librosa.power_to_db(m, ref=np.max))
    return np.stack(mels, 0)[:, None, :, :].astype(np.float32)

QUADRANTS = {(+1,+1):"happy/excited", (-1,+1):"tense/angry",
              (-1,-1):"sad",          (+1,-1):"calm/relaxed"}
MOOD_COLORS = {"happy/excited":"#f39c12","tense/angry":"#e74c3c",
                "sad":"#3498db","calm/relaxed":"#2ecc71"}

def quadrant(v, a):
    return QUADRANTS[(+1 if v>=LABEL_MEAN else -1, +1 if a>=LABEL_MEAN else -1)]

def group_timeline(times, moods, min_len=2.0):
    segs=[]; cm=moods[0]; cs=times[0]
    for i in range(1, len(moods)):
        if moods[i]!=cm:
            e=times[i]
            if e-cs>=min_len or not segs:
                segs.append({"start":float(cs),"end":float(e),"mood":cm})
            else:
                segs[-1]["end"]=float(e)
            cm=moods[i]; cs=times[i]
    segs.append({"start":float(cs),"end":float(times[-1]+WINDOW_SEC),"mood":cm})
    return segs

@st.cache_resource
def load_model():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = CNNLSTMEmotion().to(dev)
    m.load_state_dict(torch.load(CKPT_PATH, map_location=dev)["model"])
    m.eval()
    return m, dev

@torch.no_grad()
def predict(path, model, dev):
    y, _ = librosa.load(path, sr=SR, mono=True)
    total = len(y) / SR
    if total < CLIP_END_SEC:
        mel = audio_to_mel_sequence(y)
        pr = denormalize_label(model(torch.from_numpy(mel[None]).to(dev)).cpu().numpy()[0])
        times = np.arange(N_STEPS)*WINDOW_SEC + CLIP_START_SEC
    else:
        nwin = int(total/WINDOW_SEC); acc = np.zeros((nwin,2), np.float32); cnt = np.zeros(nwin, np.int32)
        L = CLIP_END_SEC-CLIP_START_SEC; stride = L/2; start = 0.0
        while start+L <= total:
            yc = y[int(start*SR):int((start+L)*SR)]
            yp = np.concatenate([np.zeros(int(CLIP_START_SEC*SR)), yc])
            mel = audio_to_mel_sequence(yp)
            pr_ = denormalize_label(model(torch.from_numpy(mel[None]).to(dev)).cpu().numpy()[0])
            idx0 = int(start/WINDOW_SEC)
            for i in range(N_STEPS):
                k = idx0+i
                if k<nwin: acc[k]+=pr_[i]; cnt[k]+=1
            start += stride
        cnt[cnt==0]=1; pr=acc/cnt[:,None]; times=np.arange(nwin)*WINDOW_SEC
    moods = [quadrant(v,a) for v,a in pr]
    segs = group_timeline(times, moods, 2.0)
    mv, ma = float(pr[:,0].mean()), float(pr[:,1].mean())
    insight = (f"Cam xuc trung binh: {quadrant(mv,ma)} (V={mv:.2f}, A={ma:.2f}). "
               + (f"Di qua {len(segs)} doan: " + " -> ".join(s['mood'] for s in segs) + "."
                  if len(set(moods))>1 else "Giu trang thai cam xuc xuyen suot."))
    return times, pr, segs, insight

st.set_page_config(page_title="Music Emotion Timeline", layout="wide")
st.title("He thong phan tich dien bien cam xuc trong am nhac")
st.caption("CNN + LSTM tren DEAM | Valence-Arousal timeline + mood insight")

up = st.file_uploader("Upload 1 file nhac (.mp3 / .wav)", type=["mp3","wav"])
if up is not None:
   
    tmp = Path(tempfile.gettempdir()) / up.name
    tmp.write_bytes(up.read())
    st.audio(str(tmp))
    with st.spinner("Dang phan tich ..."):
        model, dev = load_model()
        times, pr, segs, insight = predict(tmp, model, dev)
    st.success("Da phan tich xong!")
    st.subheader("Insight"); st.info(insight)

    col1, col2 = st.columns([2,1])
    with col1:
        fig, ax = plt.subplots(2,1, figsize=(10,6), gridspec_kw={"height_ratios":[2,1]})
        ax[0].plot(times, pr[:,0], label="Valence", color="#2980b9", lw=2)
        ax[0].plot(times, pr[:,1], label="Arousal", color="#c0392b", lw=2)
        ax[0].axhline(LABEL_MEAN, color="gray", ls="--", alpha=.5)
        ax[0].set_ylabel("V, A (1..9)"); ax[0].legend(); ax[0].grid(alpha=.3)
        for s in segs:
            c = MOOD_COLORS.get(s["mood"], "#95a5a6")
            ax[1].axvspan(s["start"], s["end"], color=c, alpha=.8)
            ax[1].text((s["start"]+s["end"])/2, .5, s["mood"], ha="center", va="center",
                        fontsize=8, color="white", fontweight="bold")
        ax[1].set_ylim(0,1); ax[1].set_yticks([])
        ax[1].set_xlim(times.min(), times.max()); ax[1].set_xlabel("Thoi gian (s)")
        st.pyplot(fig)
    with col2:
        fig2, ax2 = plt.subplots(figsize=(5,5))
        ax2.axhline(LABEL_MEAN, color="k", ls="--", alpha=.3)
        ax2.axvline(LABEL_MEAN, color="k", ls="--", alpha=.3)
        ax2.plot(pr[:,0], pr[:,1], color="#8e44ad", lw=1.2, alpha=.6)
        sc = ax2.scatter(pr[:,0], pr[:,1], c=times, cmap="viridis", s=15)
        plt.colorbar(sc, ax=ax2, label="Thoi gian (s)")
        ax2.set_xlabel("Valence"); ax2.set_ylabel("Arousal")
        ax2.set_xlim(LABEL_MIN, LABEL_MAX); ax2.set_ylim(LABEL_MIN, LABEL_MAX)
        ax2.set_title("Quy dao V-A")
        st.pyplot(fig2)

    st.subheader("Timeline (bang)")
    st.dataframe([{"start (s)":round(s["start"],2), "end (s)":round(s["end"],2),
                    "mood":s["mood"]} for s in segs])
