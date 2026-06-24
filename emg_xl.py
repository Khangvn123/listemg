"""
emg_xl.py — Merged EMG pipeline
================================
Gop 3 file thanh 1:
    00_emg_config.py        -> phan [CONFIG]
    01_emg_preprocessing.py -> phan [PREPROCESSING]
    02_emg_features.py      -> phan [FEATURE EXTRACTION]

Based on: "Machine Learning based Neuromuscular Disease Detection
           and Classification Using EMG Signal" (WIECON-ECE 2024)

Pipeline:
  1 — Load raw .asc files (163,840 samples, 5 s @ 32,768 Hz)
  2 — Filter: Notch 50/100 Hz + Butterworth bandpass 16–5000 Hz
  3 — Segment each recording into 5 × 1-second windows
  4 — Trich xuat 23 dac trung tu moi segment -> features_17.csv

Usage:
    python emg_xl.py
    # hoac trong code:
    from emg_xl import run_pipeline, extract_all
    records = run_pipeline()
    X, y, meta = extract_all(records)
"""

import os
import re
import glob
from collections import Counter

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch


# ═════════════════════════════════════════════════════════════════════════════
# [CONFIG]  (was 00_emg_config.py)
# ═════════════════════════════════════════════════════════════════════════════

# Data lives in the "02_Biceps Brachii" folder next to this file.
DATA_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "02_Biceps Brachii")

# Signal parameters
FS          = 32_768    # Hz — sampling frequency
DURATION_S  = 5         # seconds per recording
N_SAMPLES   = FS * DURATION_S   # 163,840
N_SEGMENTS  = 5
SEG_LEN     = FS * 1    # 32,768 samples per 1-second segment

# Butterworth bandpass
BP_LOW      = 16        # Hz — high-pass: removes EEG (<30 Hz) & ECG (<40 Hz)
BP_HIGH     = 5000      # Hz — low-pass:  removes high-freq noise
BP_ORDER    = 3

# Notch filters
NOTCH_FREQS = [50, 100] # Hz — power-line fundamental + 2nd harmonic
NOTCH_Q     = 30        # quality factor

CLASSES = {
    "Healthy":    0,
    "Myopathy":   1,
    "Neuropathy": 2,
}

CLASS_COLORS = {
    "Healthy":    "#2ca02c",
    "Myopathy":   "#d62728",
    "Neuropathy": "#1f77b4",
}


# ═════════════════════════════════════════════════════════════════════════════
# [PREPROCESSING]  (was 01_emg_preprocessing.py)
# ═════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
def load_signal(filepath: str) -> np.ndarray:
    """
    Read one .asc file → 1-D float64 array.
    Handles adjacent negative numbers without separator,
    e.g. '-7741.4000-11768.8000' → '-7741.4000 -11768.8000'.
    """
    with open(filepath, "r") as fh:
        content = fh.read()
    content = re.sub(r"(\d)-", r"\1 -", content)
    return np.fromstring(content, sep=" ")


def load_all_records(data_dir: str = DATA_DIR) -> list[dict]:
    """
    Walk Healthy / Myopathy / Neuropathy sub-folders.

    Returns a list of dicts, each with:
        signal   : np.ndarray  — raw signal trimmed to exactly 5 s
        label    : int         — 0 Healthy | 1 Myopathy | 2 Neuropathy
        class    : str
        filename : str
        filepath : str

    Recordings shorter than 5 s are skipped automatically
    (1 known Neuropathy file in the Mendeley dataset).
    """
    records = []
    for class_name, label in CLASSES.items():
        folder = os.path.join(data_dir, class_name)
        files  = sorted(glob.glob(os.path.join(folder, "*.asc")))
        for fp in files:
            sig = load_signal(fp)
            if sig.size < N_SAMPLES:
                print(f"  [SKIP] {os.path.basename(fp)} — {sig.size} samples < {N_SAMPLES}")
                continue
            records.append({
                "signal":   sig[:N_SAMPLES],
                "label":    label,
                "class":    class_name,
                "filename": os.path.basename(fp),
                "filepath": fp,
            })
    return records


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — FILTERING
# ─────────────────────────────────────────────────────────────────────────────
def _make_bandpass(low: float, high: float, fs: int, order: int):
    nyq  = fs / 2.0
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return b, a


def _make_notch(freq: float, fs: int, Q: float):
    b, a = iirnotch(freq / (fs / 2.0), Q)
    return b, a


def filter_signal(raw: np.ndarray, fs: int = FS) -> np.ndarray:
    """
    Two-stage zero-phase filter chain (uses filtfilt — no phase distortion).

    Stage A — Notch filters
        50 Hz  : power-line noise + ECG harmonics
        100 Hz : 2nd harmonic of power-line

    Stage B — Butterworth bandpass (order 3)
        High-pass 30 Hz : removes EEG (0.5–30 Hz) and ECG (0.5–40 Hz)
        Low-pass 450 Hz : removes high-frequency noise above the EMG band

    Returns clean EMG signal in the 30–450 Hz window.
    """
    sig = raw.copy()

    for f in NOTCH_FREQS:
        b, a = _make_notch(f, fs, Q=NOTCH_Q)
        sig  = filtfilt(b, a, sig)

    b, a = _make_bandpass(BP_LOW, BP_HIGH, fs, order=BP_ORDER)
    sig  = filtfilt(b, a, sig)

    return sig


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — SEGMENTATION
# ─────────────────────────────────────────────────────────────────────────────
def segment_signal(sig: np.ndarray,
                   seg_len: int = SEG_LEN,
                   n_seg:   int = N_SEGMENTS) -> list[np.ndarray]:
    """Split signal into n_seg non-overlapping 1-second windows."""
    return [sig[i * seg_len:(i + 1) * seg_len] for i in range(n_seg)]


# ─────────────────────────────────────────────────────────────────────────────
# FULL PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline(data_dir: str = DATA_DIR) -> list[dict]:
    """
    Execute Steps 1–3 and return the processed records list.

    Each record dict is extended with:
        filtered     : np.ndarray          — filtered full signal
        segments     : list[np.ndarray]    — 5 × filtered 1-second segments
        segments_raw : list[np.ndarray]    — 5 × raw 1-second segments (for display)
    """
    print("=" * 60)
    print("STEP 1 — Loading raw signals …")
    records = load_all_records(data_dir)
    counts  = Counter(r["class"] for r in records)
    for cls, n in counts.items():
        print(f"  {cls:12s}: {n:3d} recordings")
    print(f"  {'TOTAL':12s}: {len(records):3d} recordings")

    print(f"\nSTEP 2 — Filtering  [Notch {NOTCH_FREQS} Hz  +  BP {BP_LOW}–{BP_HIGH} Hz] …")
    for rec in records:
        rec["filtered"] = filter_signal(rec["signal"])

    print(f"\nSTEP 3 — Segmenting into {N_SEGMENTS} × 1-second windows …")
    for rec in records:
        rec["segments"]     = segment_signal(rec["filtered"])
        rec["segments_raw"] = segment_signal(rec["signal"])

    total_seg = sum(len(r["segments"]) for r in records)
    print(f"  Total segments: {total_seg}")
    print("=" * 60)
    return records


# ═════════════════════════════════════════════════════════════════════════════
# [FEATURE EXTRACTION]  (was 02_emg_features.py)
# ═════════════════════════════════════════════════════════════════════════════
#
# Trich xuat 23 dac trung tu moi segment da loc:
#
#   [Time domain]
#     MAV  - Mean Absolute Value           : trung binh tri tuyet doi
#     RMS  - Root Mean Square              : can bac hai trung binh binh phuong
#     VAR  - Variance                      : phuong sai bien do (EMG variant)
#     WL   - Waveform Length               : tong bien thien tuyet doi
#     SD   - Standard Deviation            : do lech chuan
#     ZC   - Zero Crossing                 : so lan tin hieu cat qua 0
#     SSC  - Slope Sign Change             : so lan dao ham doi dau
#     WAMP - Willison Amplitude            : so lan |diff| vuot nguong
#     IEMG - Integrated EMG                : tong tri tuyet doi
#     SSI  - Simple Square Integral        : tong binh phuong
#     MV   - Mean Value                    : gia tri trung binh
#     LOG  - Log Detector                  : exp(mean(log(|x|)))
#     MFL  - Maximum Fractal Length        : log10(sqrt(sum(diff^2)))
#     DAMV - Difference Absolute Mean Value: mean(|diff|)
#     TURN_AMP - Turn Amplitude            : bien do trung binh tai cac MUAP turn
#     P2P  - Peak-to-Peak                  : bien do dinh-dinh cua so truot 10ms
#     MMAV - Modified Mean Absolute Value  : MAV co trong so vi tri (giua=1, dau=0.5)
#     EWL  - Enhanced Waveform Length      : sum(|diff|^0.75) co trong so vi tri
#     SKEW - Skewness                      : do bat doi xung phan phoi bien do
#     KURT - Kurtosis (excess)             : do nho/phang phan phoi bien do
#
#   [Frequency domain]
#     MNF  - Mean Frequency                : tan so trung binh cua PSD
#     MDF  - Median Frequency              : tan so trung vi cua PSD
#     PKF  - Peak Frequency                : tan so co cong suat lon nhat

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
_HERE       = os.path.dirname(os.path.abspath(__file__))
OUTPUT_CSV  = os.path.join(_HERE, "features_17.csv")
OUTPUT_DIR  = _HERE


# ─────────────────────────────────────────────────────────────────────────────
# TIME-DOMAIN FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def mav(seg: np.ndarray) -> float:
    """Mean Absolute Value — MAV = (1/N) * sum(|x_i|)"""
    return float(np.mean(np.abs(seg)))


def rms(seg: np.ndarray) -> float:
    """Root Mean Square — RMS = sqrt((1/N) * sum(x_i^2))"""
    return float(np.sqrt(np.mean(seg ** 2)))


def var(seg: np.ndarray) -> float:
    """Variance (EMG) — VAR = (1/N) * sum(x_i^2)  (khong tru mean)"""
    return float(np.mean(seg ** 2))


def wl(seg: np.ndarray) -> float:
    """Waveform Length — WL = sum(|x[i+1] - x[i]|)"""
    return float(np.sum(np.abs(np.diff(seg))))


def sd(seg: np.ndarray) -> float:
    """Standard Deviation — SD = std(x)"""
    return float(np.std(seg))


def zc(seg: np.ndarray, threshold: float = 0.0) -> float:
    """
    Zero Crossing — so lan tin hieu cat qua muc 0.
    ZC = sum( sign(x[i]) != sign(x[i+1])  AND  |x[i] - x[i+1]| > threshold )
    """
    s = seg.astype(float)
    sign_change = (s[:-1] * s[1:] < 0)
    above_thresh = np.abs(s[:-1] - s[1:]) > threshold
    return float(np.sum(sign_change & above_thresh))


def ssc(seg: np.ndarray, threshold: float = 0.0) -> float:
    """
    Slope Sign Change — so lan dao ham doi dau.
    SSC = sum( d[i]*d[i-1] < 0  AND  |d| > threshold )
    """
    d = np.diff(seg.astype(float))
    changes = np.sum(
        (d[:-1] * d[1:] < -threshold) |
        ((np.abs(d[:-1]) > threshold) & (np.abs(d[1:]) > threshold) & (d[:-1] * d[1:] < 0))
    )
    return float(changes)


def wamp(seg: np.ndarray, threshold: float = 10.0) -> float:
    """
    Willison Amplitude — so lan |x[i+1] - x[i]| vuot nguong.
    Threshold mac dinh 10 (don vi cung voi tin hieu).
    """
    return float(np.sum(np.abs(np.diff(seg.astype(float))) > threshold))


def iemg(seg: np.ndarray) -> float:
    """Integrated EMG — IEMG = sum(|x_i|)"""
    return float(np.sum(np.abs(seg)))


def ssi(seg: np.ndarray) -> float:
    """Simple Square Integral — SSI = sum(x_i^2)"""
    return float(np.sum(seg ** 2))


def mv(seg: np.ndarray) -> float:
    """Mean Value — MV = mean(x)"""
    return float(np.mean(seg))


def log_detector(seg: np.ndarray) -> float:
    """
    Log Detector — LOG = exp( (1/N) * sum(log(|x_i|)) )
    Do lech trung binh log-amplitude.
    """
    return float(np.exp(np.mean(np.log(np.abs(seg) + 1e-9))))


def mfl(seg: np.ndarray) -> float:
    """
    Maximum Fractal Length — MFL = log10( sqrt( sum((x[i+1]-x[i])^2) ) )
    Do do phuc tap fractal cua tin hieu.
    """
    return float(np.log10(np.sqrt(np.sum(np.diff(seg.astype(float)) ** 2)) + 1e-9))


def damv(seg: np.ndarray) -> float:
    """Difference Absolute Mean Value — DAMV = mean(|x[i+1] - x[i]|)"""
    return float(np.mean(np.abs(np.diff(seg.astype(float)))))


def turn_amp(seg: np.ndarray, threshold: float = 10.0) -> float:
    """
    Turn Amplitude — bien do trung binh tai cac 'turn' (diem doi chieu do doc).
    Turn = noi dao ham doi dau VA |diff| > threshold (MUAP turn).
    Myopathy: thap (MUAP nho do soi co teo).
    Neuropathy: cao (MUAP lon do tai phan bo than kinh).
    """
    s  = seg.astype(float)
    d  = np.diff(s)
    sc = (d[:-1] * d[1:] < 0) & (np.abs(d[:-1]) > threshold)
    if sc.sum() == 0:
        return 0.0
    return float(np.mean(np.abs(s[1:-1][sc])))


def p2p(seg: np.ndarray, win_ms: float = 10.0) -> float:
    """
    Peak-to-Peak amplitude — bien do dinh-dinh trung binh tren cua so truot
    ~win_ms (mac dinh 10ms), do truc tiep do lon MUAP.
    Myopathy: nho (~800 uV). Neuropathy: lon (~1400 uV).
    """
    s = seg.astype(float)
    w = int(FS * win_ms / 1000.0)
    if w < 2:
        return float(s.max() - s.min())
    n = len(s) // w
    if n < 1:
        return float(s.max() - s.min())
    p = [s[i * w:(i + 1) * w].max() - s[i * w:(i + 1) * w].min() for i in range(n)]
    return float(np.mean(p))


def _pos_weight(N: int) -> np.ndarray:
    """Trong so vi tri: 1.0 cho mau giua (25%-75%), 0.5 hai dau (Phinyomark)."""
    w = np.full(N, 0.5)
    lo, hi = int(0.25 * N), int(0.75 * N)
    w[lo:hi] = 1.0
    return w


def mmav(seg: np.ndarray) -> float:
    """
    Modified Mean Absolute Value — MAV co trong so vi tri.
    Mau o giua segment (25%-75%) weight 1.0, hai dau 0.5 -> giam nhieu bien.
    Myopathy thap (~114), Neuropathy cao (~217) — d=1.15 tach Myo/Neuro tot.
    """
    s = np.abs(seg.astype(float))
    return float(np.sum(_pos_weight(len(s)) * s) / len(s))


def ewl(seg: np.ndarray, p: float = 0.75) -> float:
    """
    Enhanced Waveform Length — WL voi |diff|^p va trong so vi tri.
    Nhan manh bien thien o vung giua segment, do phuc tap dang song.
    """
    d = np.abs(np.diff(seg.astype(float)))
    return float(np.sum(_pos_weight(len(d)) * (d ** p)))


def skew(seg: np.ndarray) -> float:
    """
    Skewness — do bat doi xung cua phan phoi bien do.
    SKEW = mean( ((x - mean) / std)^3 ).
    """
    s = seg.astype(float)
    return float(np.mean(((s - np.mean(s)) / (np.std(s) + 1e-9)) ** 3))


def kurt(seg: np.ndarray) -> float:
    """
    Kurtosis (excess) — do nho/phang cua phan phoi bien do.
    KURT = mean( ((x - mean) / std)^4 ) - 3.
    """
    s = seg.astype(float)
    return float(np.mean(((s - np.mean(s)) / (np.std(s) + 1e-9)) ** 4) - 3.0)


# ─────────────────────────────────────────────────────────────────────────────
# FREQUENCY-DOMAIN FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def mnf(seg: np.ndarray) -> float:
    """Mean Frequency — MNF = sum(f_i * PSD_i) / sum(PSD_i)"""
    freqs = np.fft.rfftfreq(len(seg), 1.0 / FS)
    psd   = np.abs(np.fft.rfft(seg)) ** 2
    return float(np.sum(freqs * psd) / (np.sum(psd) + 1e-9))


def mdf(seg: np.ndarray) -> float:
    """Median Frequency — tan so tai do PSD tich luy dat 50%."""
    freqs  = np.fft.rfftfreq(len(seg), 1.0 / FS)
    psd    = np.abs(np.fft.rfft(seg)) ** 2
    cumsum = np.cumsum(psd)
    idx    = np.searchsorted(cumsum, cumsum[-1] / 2.0)
    idx    = min(idx, len(freqs) - 1)
    return float(freqs[idx])


def pkf(seg: np.ndarray) -> float:
    """Peak Frequency — PKF = tan so co cong suat PSD lon nhat."""
    freqs = np.fft.rfftfreq(len(seg), 1.0 / FS)
    psd   = np.abs(np.fft.rfft(seg)) ** 2
    return float(freqs[np.argmax(psd)])


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE REGISTRY  (thu tu: MAV RMS VAR WL SD ZC SSC WAMP IEMG SSI MV LOG MFL DAMV TURN_AMP P2P MMAV EWL SKEW KURT MNF MDF PKF)
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_FUNCS = {
    "MAV":  mav,
    "RMS":  rms,
    "VAR":  var,
    "WL":   wl,
    "SD":   sd,
    "ZC":   zc,
    "SSC":  ssc,
    "WAMP": wamp,
    "IEMG": iemg,
    "SSI":  ssi,
    "MV":   mv,
    "LOG":  log_detector,
    "MFL":  mfl,
    "DAMV": damv,
    "TURN_AMP": turn_amp,
    "P2P":  p2p,
    "MMAV": mmav,
    "EWL":  ewl,
    "SKEW": skew,
    "KURT": kurt,
    "MNF":  mnf,
    "MDF":  mdf,
    "PKF":  pkf,
}

FEATURE_NAMES = list(FEATURE_FUNCS.keys())
# 23 features total


# ─────────────────────────────────────────────────────────────────────────────
# EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────
def extract_segment(seg: np.ndarray) -> np.ndarray:
    """Trich xuat vector 23 dac trung tu 1 segment."""
    return np.array([fn(seg) for fn in FEATURE_FUNCS.values()])


def extract_all(records: list[dict]) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """
    Duyet qua tat ca records va segments, trich xuat dac trung.

    Tra ve:
        X    : np.ndarray  shape (n_samples, 23)
        y    : np.ndarray  shape (n_samples,)
        meta : list[dict]
    """
    X_rows, y_rows, meta = [], [], []

    for rec in records:
        for seg_idx, seg in enumerate(rec["segments"]):
            X_rows.append(extract_segment(seg))
            y_rows.append(rec["label"])
            meta.append({
                "class":    rec["class"],
                "label":    rec["label"],
                "filename": rec["filename"],
                "seg_idx":  seg_idx,
            })

    X = np.vstack(X_rows)
    y = np.array(y_rows)
    return X, y, meta


def to_dataframe(X: np.ndarray, y: np.ndarray, meta: list[dict]) -> pd.DataFrame:
    """Dong goi X, y va meta thanh DataFrame."""
    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df.insert(0, "class",    [m["class"]    for m in meta])
    df.insert(1, "label",    y)
    df.insert(2, "filename", [m["filename"] for m in meta])
    df.insert(3, "seg_idx",  [m["seg_idx"]  for m in meta])
    return df


# ═════════════════════════════════════════════════════════════════════════════
# STANDALONE RUN — preprocessing + feature extraction + luu CSV
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Running preprocessing pipeline ...")
    records = run_pipeline()

    print(f"\nExtracting features {FEATURE_NAMES} ...")
    X, y, meta = extract_all(records)

    labels_inv = {v: k for k, v in CLASSES.items()}

    print(f"\n{'='*60}")
    print(f"  Total samples : {X.shape[0]}")
    print(f"  Features      : {X.shape[1]}  {FEATURE_NAMES}")
    print(f"  Label dist.   :")
    unique, counts = np.unique(y, return_counts=True)
    for lbl, cnt in zip(unique, counts):
        print(f"    {labels_inv[lbl]:12s} (label={lbl}): {cnt} samples")
    print(f"{'='*60}")

    df = to_dataframe(X, y, meta)

    # ── Luu features_17.csv (dau vao chinh cua 03_feat17_pop_snn_holdout.py) ──
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved (all classes) : {OUTPUT_CSV}")

    # ── Luu tung lop ──────────────────────────────────────────────────────────
    class_file_map = {
        "Healthy":    "features_healthy.csv",
        "Myopathy":   "features_myopathy.csv",
        "Neuropathy": "features_neuropathy.csv",
    }
    for class_name, fname in class_file_map.items():
        df_cls = df[df["class"] == class_name].reset_index(drop=True)
        out_path = os.path.join(OUTPUT_DIR, fname)
        df_cls.to_csv(out_path, index=False)
        print(f"Saved ({class_name:10s}): {out_path}  [{len(df_cls)} rows]")

    # ── Thong ke mo ta ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Descriptive statistics per class (mean +/- std):")
    print(f"{'='*60}")
    for class_name in ["Healthy", "Myopathy", "Neuropathy"]:
        sub = df[df["class"] == class_name][FEATURE_NAMES]
        print(f"\n[{class_name}]")
        stat = sub.agg(["mean", "std"]).T
        stat.columns = ["mean", "std"]
        print(stat.round(4).to_string())
