"""
emg_infer.py - Suy luan 1 ban ghi EMG -> chan doan 3 lop
=========================================================
Module dung cho GUI (app.py). Tai dung pipeline emg_xl + model 2 tang da train.

Luong:
  bytes .asc -> load -> loc (Notch+bandpass) -> 5 segment 1s -> 23 dac trung/segment
            -> chuan hoa (normalizer Biceps-train) -> SNN 2 tang -> xac suat 3 lop
            -> gop segment -> nhan + do tin cay.

KHONG sua emg_xl.py / 05_two_stage_snn.py. Can:
  features_17.csv (fit normalizer), layer05_two_stage_snn.pth (trong so).
"""

import os
import re
import importlib.util

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split


HERE       = os.path.dirname(os.path.abspath(__file__))
BICEPS_CSV = os.path.join(HERE, "features_17.csv")
MODEL_PATH = os.path.join(HERE, "layer05_two_stage_snn.pth")


def _load(name, fn):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fn))
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

emg = _load("emg_xl", "emg_xl.py")
snn = _load("two_stage_snn", "05_two_stage_snn.py")

CLASS_NAMES = snn.CLASS_NAMES                       # ['Healthy','Myopathy','Neuropathy']
CLASS_VI    = {"Healthy": "Binh thuong (Healthy)",
               "Myopathy": "Benh co (Myopathy)",
               "Neuropathy": "Benh than kinh (Neuropathy)"}
S1F, S2F    = snn.STAGE1_FEATS, snn.STAGE2_FEATS
device      = snn.device

# Chi so cot cho tung tang trong vector 23 dac trung (thu tu emg.FEATURE_NAMES)
_FN   = emg.FEATURE_NAMES
_IDX1 = [_FN.index(f) for f in S1F]
_IDX2 = [_FN.index(f) for f in S2F]

# Nguong kiem tra mien du lieu = khoang cach Mahalanobis (da bien) toi phan bo train.
# Tren log-feature + chuan hoa + shrinkage cov. Hieu chinh: Biceps ~2.5, data_test ~5.3.
# T=3.8 -> false-reject Biceps ~5%, bat data_test ~79%. (Deltoid ~similar -> kho bat.)
DOMAIN_MAHA_TH = 3.8

# Cac dac trung de DIEN GIAI khi hien dac diem he thong (ten + nhan tieng Viet)
_DISPLAY_FEATS = [
    ("RMS",  "Biên độ hiệu dụng (RMS)"),
    ("MAV",  "Biên độ trung bình (MAV)"),
    ("WL",   "Độ dài dạng sóng (WL)"),
    ("ZC",   "Số lần cắt mức 0 (ZC)"),
    ("MNF",  "Tần số trung bình (MNF, Hz)"),
    ("MDF",  "Tần số trung vị (MDF, Hz)"),
]


# ─────────────────────────────────────────────────────────────────────────────
# NAP MODEL + NORMALIZER (mot lan)
# ─────────────────────────────────────────────────────────────────────────────
def build_predictor():
    """Nap 2 tang + normalizer Biceps-train. Goi 1 lan, cache lai o GUI."""
    ck = torch.load(MODEL_PATH, map_location=device)
    m1 = snn.PopSNN(len(S1F) * snn.N_POP, 2, n_hidden=snn.HIDDEN).to(device)
    m2 = snn.PopSNN(len(S2F) * snn.N_POP, 2, n_hidden=snn.S2_HIDDEN).to(device)
    m1.load_state_dict(ck["stage1"]); m2.load_state_dict(ck["stage2"])
    m1.eval(); m2.eval()

    # Normalizer = fit tren TRAIN-split cua Biceps (trung khit luc train)
    df_b, y_b, keys_b = snn.load_dataset(BICEPS_CSV)
    uk = np.unique(keys_b); ul = np.array([y_b[keys_b == k][0] for k in uk])
    k_tr, _, _, _ = train_test_split(
        uk, ul, test_size=(snn.VALID_RATIO + snn.TEST_RATIO),
        stratify=ul, random_state=snn.SPLIT_SEED)
    tr_m = np.isin(keys_b, k_tr)
    norm1 = snn.fit_normalizer(df_b[S1F].to_numpy(np.float32)[tr_m])
    norm2 = snn.fit_normalizer(df_b[S2F].to_numpy(np.float32)[tr_m])

    # Tham chieu kiem tra mien du lieu (tu toan bo Biceps):
    #   - p1/p99 tung dac trung -> giai thich dac trung nao lech
    #   - Mahalanobis (mu, sd, Ci) tren log-feature -> quyet dinh hop le/khong
    Xtr = df_b[_FN].to_numpy(np.float64)
    Xtr = np.sign(Xtr) * np.log1p(np.abs(Xtr))          # nen dynamic range
    mu  = Xtr.mean(0)
    sd  = Xtr.std(0) + 1e-9
    Z   = (Xtr - mu) / sd
    C   = np.cov(Z.T)
    lam = 0.2                                            # shrinkage -> cov on dinh, kha nghich dao
    C   = (1 - lam) * C + lam * np.eye(len(C)) * np.trace(C) / len(C)
    ref = {
        "p1":  df_b[_FN].quantile(0.01).to_numpy(np.float64),
        "p99": df_b[_FN].quantile(0.99).to_numpy(np.float64),
        "maha": {"mu": mu, "sd": sd, "Ci": np.linalg.inv(C)},
    }
    return {"m1": m1, "m2": m2, "norm1": norm1, "norm2": norm2, "ref": ref}


# ─────────────────────────────────────────────────────────────────────────────
# DOC + TIEN XU LY + TRICH DAC TRUNG
# ─────────────────────────────────────────────────────────────────────────────
def signal_from_bytes(data: bytes) -> np.ndarray:
    """Doc noi dung .asc (bytes) -> mang 1-D, giong emg.load_signal."""
    content = data.decode("utf-8", errors="ignore")
    content = re.sub(r"(\d)-", r"\1 -", content)
    return np.fromstring(content, sep=" ")


def features_from_signal(sig: np.ndarray):
    """sig raw -> (features (n_seg,23), filtered_signal). Raise neu < 5 giay."""
    if sig.size < emg.N_SAMPLES:
        raise ValueError(
            f"Tin hieu chi co {sig.size} mau (< {emg.N_SAMPLES} = 5s @ {emg.FS}Hz). "
            f"File qua ngan de phan tich.")
    filt  = emg.filter_signal(sig[:emg.N_SAMPLES])           # Notch + bandpass
    segs  = emg.segment_signal(filt)                          # 5 x 1s
    feats = np.vstack([emg.extract_segment(s) for s in segs]) # (5, 23)
    return feats, filt


def _enc(feats_cols, norm):
    p1, denom = norm
    return snn.population_encode(snn.apply_normalizer(feats_cols.astype(np.float32), p1, denom))


# ─────────────────────────────────────────────────────────────────────────────
# SUY LUAN 2 TANG -> XAC SUAT 3 LOP
# ─────────────────────────────────────────────────────────────────────────────
def predict(predictor, feats: np.ndarray) -> dict:
    """feats (n_seg,23) -> dict ket qua chan doan."""
    m1, m2 = predictor["m1"], predictor["m2"]
    p1 = snn.get_probs(m1, _enc(feats[:, _IDX1], predictor["norm1"]))   # (n,2) Healthy/Sick
    p2 = snn.get_probs(m2, _enc(feats[:, _IDX2], predictor["norm2"]))   # (n,2) Myo/Neuro

    # Xac suat 3 lop / segment: P(H)=p1_h ; P(Myo)=p1_sick*p2_myo ; P(Neu)=p1_sick*p2_neu
    prob3 = np.stack([p1[:, 0], p1[:, 1] * p2[:, 0], p1[:, 1] * p2[:, 1]], axis=1)  # (n,3)
    avg   = prob3.mean(axis=0)                                # gop segment
    avg   = avg / avg.sum()
    pred_i = int(np.argmax(avg))
    seg_preds = prob3.argmax(axis=1)

    return {
        "predicted":    CLASS_NAMES[pred_i],
        "predicted_vi": CLASS_VI[CLASS_NAMES[pred_i]],
        "confidence":   float(avg[pred_i]),
        "probs":        {CLASS_NAMES[i]: float(avg[i]) for i in range(3)},
        "n_segments":   int(feats.shape[0]),
        "seg_agreement": float(np.mean(seg_preds == pred_i)),   # ty le segment dong thuan
        "stage1_sick":  float(p1[:, 1].mean()),                 # P(Sick) trung binh tang 1
    }


# ─────────────────────────────────────────────────────────────────────────────
# KIEM TRA MIEN DU LIEU (co cung "loai" voi data train khong?)
# ─────────────────────────────────────────────────────────────────────────────
def check_domain(predictor, feats: np.ndarray) -> dict:
    """
    Kiem tra ban ghi co cung "loai" voi tap train khong (Mahalanobis da bien).
    Tra ve:
      valid     : khoang cach Mahalanobis <= DOMAIN_MAHA_TH
      compat    : diem tuong thich 0..1 (de hien thi; >=0.5 <=> valid)
      distance  : khoang cach Mahalanobis
      offenders : dac trung nam ngoai [p1,p99] (de giai thich cai gi khac)
    """
    x = feats.mean(axis=0)                       # gop 5 segment -> 1 vector (23,)
    ref = predictor["ref"]
    m = ref["maha"]

    xl   = np.sign(x) * np.log1p(np.abs(x))
    z    = (xl - m["mu"]) / m["sd"]
    dist = float(np.sqrt(max(float(z @ m["Ci"] @ z), 0.0)))
    valid = dist <= DOMAIN_MAHA_TH
    # diem tuong thich: dist=0 ->100%, dist=T ->50%, dist=2T ->0%
    compat = max(0.0, min(1.0, 1.0 - dist / (2.0 * DOMAIN_MAHA_TH)))

    p1, p99 = ref["p1"], ref["p99"]
    within = (x >= p1) & (x <= p99)
    offenders = []
    for i, ok in enumerate(within):
        if ok:
            continue
        rng = max(p99[i] - p1[i], 1e-9)
        sev = (p1[i] - x[i]) / rng if x[i] < p1[i] else (x[i] - p99[i]) / rng
        offenders.append({
            "feat": _FN[i], "value": float(x[i]),
            "lo": float(p1[i]), "hi": float(p99[i]),
            "dir": "thấp hơn" if x[i] < p1[i] else "cao hơn",
            "severity": float(sev),
        })
    offenders.sort(key=lambda d: d["severity"], reverse=True)
    return {"valid": bool(valid), "compat": compat, "distance": dist,
            "threshold": DOMAIN_MAHA_TH, "n_out": len(offenders), "offenders": offenders}


def system_profile(predictor) -> dict:
    """Dac diem du lieu he thong duoc train tren (de hien thi cho nguoi dung)."""
    p1, p99 = predictor["ref"]["p1"], predictor["ref"]["p99"]
    rows = []
    for feat, label in _DISPLAY_FEATS:
        i = _FN.index(feat)
        rows.append({"Đặc trưng": label, "Khoảng mong đợi (p1–p99)": f"{p1[i]:,.1f}  –  {p99[i]:,.1f}"})
    return {
        "muscle":      "Biceps Brachii (bộ Mendeley)",
        "fs":          emg.FS,
        "duration_s":  emg.DURATION_S,
        "n_samples":   emg.N_SAMPLES,
        "n_features":  len(_FN),
        "feature_ranges": rows,
        "compat_min":  0.5,        # diem tuong thich toi thieu de hop le
    }


def analyze(predictor, data: bytes) -> dict:
    """bytes .asc -> dict {ok, ...} hoac {ok: False, error}. Khong nem loi ra ngoai."""
    try:
        sig = signal_from_bytes(data)
        feats, filt = features_from_signal(sig)
        res = predict(predictor, feats)
        res["domain"] = check_domain(predictor, feats)         # kiem tra mien du lieu
        # Tin hieu da loc, 1 giay dau, downsample ~2000 diem de ve
        seg0 = filt[:emg.SEG_LEN]
        step = max(1, len(seg0) // 2000)
        res["signal_preview"] = seg0[::step].astype(float).tolist()
        res["raw_samples"]    = int(sig.size)
        res["ok"] = True
        return res
    except Exception as e:
        return {"ok": False, "error": str(e)}
