# 🎵 MUSIC EMOTION APP — Dynamic Music Emotion Recognition

> Hệ thống nhận diện cảm xúc âm nhạc theo thời gian thực sử dụng kiến trúc **CNN + BiLSTM + Multi-head Self-Attention** với hàm mất mát CCC (Concordance Correlation Coefficient).

---

## 📌 Giới thiệu

**MUSIC EMOTION APP** là ứng dụng nhận diện cảm xúc âm nhạc động (Dynamic Music Emotion Recognition — DMER), dự đoán cặp giá trị **Valence–Arousal** liên tục theo từng cửa sổ thời gian 0,5 giây trong suốt bản nhạc.

Khác với các hệ thống phân loại cảm xúc tĩnh (gán một nhãn duy nhất cho toàn bài), hệ thống này theo dõi **sự thay đổi cảm xúc theo thời gian** — phản ánh đúng hơn trải nghiệm nghe nhạc thực tế của con người.

Cảm xúc được biểu diễn trong **không gian Russell 2 chiều**:
- **Valence** — mức độ tích cực/tiêu cực (buồn ↔ vui)
- **Arousal** — mức độ kích thích/năng lượng (trầm lắng ↔ sôi động)

---

## 🏗️ Kiến trúc mô hình

```
Đầu vào: Audio (.mp3/.wav)
    ↓
[Tiền xử lý] Mel Spectrogram (SR=22050, N_MELS=64, N_FFT=1024, HOP_LENGTH=256)
    ↓
[CNN] Trích xuất đặc trưng cục bộ (local feature extraction)
    ↓
[BiLSTM] Mô hình hoá phụ thuộc thời gian hai chiều (HIDDEN=128, LAYERS=2)
    ↓
[Multi-head Self-Attention] Tập trung vào các bước thời gian quan trọng
    ↓
[FC Layer] Hồi quy dự đoán Valence & Arousal ∈ [-1, 1]
    ↓
Đầu ra: Chuỗi (Valence, Arousal) theo từng 0,5 giây
```

**Siêu tham số chính:**

| Tham số | Giá trị |
|---------|---------|
| Sample Rate | 22,050 Hz |
| N_MELS | 64 |
| N_FFT | 1,024 |
| HOP_LENGTH | 256 |
| WINDOW_SEC | 0.5 giây |
| N_STEPS (frames/window) | 60 |
| LSTM Hidden | 128 |
| LSTM Layers | 2 |
| Dropout | 0.3 |
| Loss Function | CCC Loss |

---

## 🗂️ Các checkpoint mô hình

| File | Mô tả |
|------|-------|
| `best.pt` | Mô hình tốt nhất huấn luyện trên DEAM (baseline) |
| `best_attention.pt` | Mô hình có Multi-head Self-Attention, huấn luyện trên DEAM |
| `best_attention_balanced.pt` | Mô hình Attention với dữ liệu cân bằng theo phân phối V-A |
| `best_pmemo_ft_head.pt` | Mô hình fine-tune trên PMEmo (chiến lược freeze CNN, chỉ train head) |

---

## 📊 Kết quả thực nghiệm

### Đánh giá trên DEAM (1,802 bài nhạc)

| Mô hình | CCC_Valence | CCC_Arousal |
|---------|-------------|-------------|
| Baseline CNN+BiLSTM | 0.612 | 0.741 |
| + Multi-head Self-Attention | **0.647** | **0.772** |

### Đánh giá trên PMEmo — Transfer Learning (794 chorus clips)

| Chiến lược | CCC_Valence | CCC_Arousal |
|-----------|-------------|-------------|
| Fine-tune toàn bộ | 0.651 | 0.798 |
| Freeze CNN (chỉ train head) | **0.686** | **0.823** |

> **CCC (Concordance Correlation Coefficient)** đo mức độ tương quan đồng thời giữa dự đoán và nhãn thực tế. Giá trị CCC ∈ [-1, 1]; càng gần 1 càng tốt.

---

## 📦 Cài đặt

```bash
# Clone repository
git clone https://github.com/TangNgocPhung/MUSIC_EMOTION_APP.git
cd MUSIC_EMOTION_APP

# Tạo môi trường ảo (khuyến nghị)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Cài đặt thư viện
pip install -r requirements.txt
```

---

## 🚀 Sử dụng

### Chạy ứng dụng chính

```bash
streamlit run app.py
```

### Chạy phiên bản thay thế

```bash
streamlit run app1.py
```

Sau khi khởi động, ứng dụng mở trên trình duyệt tại `http://localhost:8501`.

---

## 🎛️ Các chức năng chính

### 1. 🎧 Tải lên và phân tích bài nhạc
- Hỗ trợ định dạng `.mp3`, `.wav`
- Tự động trích xuất Mel Spectrogram và chia cửa sổ thời gian 0,5 giây
- Hiển thị waveform và spectrogram của bài nhạc

### 2. 📈 Biểu đồ Valence–Arousal theo thời gian
- Trực quan hóa chuỗi giá trị V và A liên tục suốt bài nhạc
- Cho thấy rõ sự thay đổi cảm xúc theo từng đoạn nhạc

### 3. 🗺️ Bản đồ cảm xúc Russell (V-A Space)
- Vẽ quỹ đạo chuyển động của cảm xúc trên mặt phẳng 2D
- Phân vùng 4 góc: vui–sôi động, vui–trầm lắng, buồn–sôi động, buồn–trầm lắng

### 4. 🔀 Lựa chọn mô hình
- Chọn một trong bốn checkpoint tùy theo mục đích sử dụng
- So sánh kết quả dự đoán giữa các mô hình

### 5. 🕹️ Phát nhạc đồng bộ
- Phát bài nhạc và theo dõi điểm cảm xúc hiện tại di chuyển theo thời gian thực trên biểu đồ

---

## 🛠️ Tech Stack

| Thành phần | Công nghệ |
|-----------|-----------|
| Deep Learning | PyTorch |
| Audio Processing | librosa, torchaudio |
| Web Interface | Streamlit |
| Visualization | Plotly, Matplotlib |
| Datasets | DEAM (Univ. of Geneva), PMEmo (SJTU) |
| Training Environment | Google Colab (GPU Tesla T4) |

---

## 📁 Cấu trúc thư mục

```
MUSIC_EMOTION_APP/
├── app.py                      # Ứng dụng Streamlit chính
├── app1.py                     # Phiên bản giao diện thay thế
├── best.pt                     # Checkpoint DEAM baseline
├── best_attention.pt           # Checkpoint DEAM + Attention
├── best_attention_balanced.pt  # Checkpoint DEAM + Attention + Balanced
├── best_pmemo_ft_head.pt       # Checkpoint PMEmo fine-tune
└── requirements.txt            # Danh sách thư viện
```

---

## 📚 Dữ liệu huấn luyện

- **DEAM** (MediaEval Database for Emotional Analysis of Music) — 1,802 bài nhạc, nhãn V-A được thu thập liên tục mỗi 0,5 giây từ nhiều annotator, Đại học Geneva.
- **PMEmo** — 794 đoạn nhạc phổ biến (chorus clips), nhãn V-A liên tục, Shanghai Jiao Tong University.

---

## 👩‍💻 Tác giả

**Tăng Ngọc Phụng**
- GitHub: [@TangNgocPhung](https://github.com/TangNgocPhung)
- Học viên Cao học — Trường Đại học Sư phạm TP. Hồ Chí Minh

---

## 📄 Giấy phép

Dự án phục vụ mục đích học thuật và nghiên cứu. Vui lòng liên hệ tác giả trước khi sử dụng thương mại.
