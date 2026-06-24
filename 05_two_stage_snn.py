"""
Layer 07 [TWO-STAGE] — SNN phan tang (Healthy/Sick) -> (Myo/Neuro)
==================================================================
Y tuong: tach bai toan 3 lop thanh 2 tang nhi phan, moi tang dung BO FEATURE
RIENG toi uu cho nhiem vu cua no.

  TANG 1 — Healthy vs Sick (Myo+Neuro gop lai)
      Dung feature pho thong -> giu kha nang nhan Healthy (von da tot, F1 ~90%).

  TANG 2 — Myopathy vs Neuropathy   (CHI chay khi tang 1 ra "Sick")
      Dung feature BIEN DO manh (Cohen d > 1.0): RMS, SD, MAV, IEMG, P2P,
      TURN_AMP, SSI  -> tap trung tach dung cap kho.

Du doan cuoi cung (3 lop):
      tang1 = Healthy           -> Healthy
      tang1 = Sick, tang2 = Myo -> Myopathy
      tang1 = Sick, tang2 = Neu -> Neuropathy

Moi tang la 1 SNN nhi phan (tai dung PopSNN + TTFS + ATan giong file 03,
chi doi n_out=2). Train RIENG tung tang tren cung subject-split.

LUU Y error propagation: recall Myopathy cuoi = recall(Sick|Myo) tai tang1
                          x recall(Myo|Sick) tai tang2. Loi tang1 khong sua duoc.

Doc features_17.csv (can cot TURN_AMP, P2P -> chay 02_emg_features.py truoc).
Khong ghi de model file 03 (luu rieng layer05_stage*).
"""

import os
import re
import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix


# ─────────────────────────────────────────────────────────────────────────────
# CẤU HÌNH
# ─────────────────────────────────────────────────────────────────────────────
CSV_PATH      = 'features_17.csv'

# ONLY_STAGE1 = True  -> BO QUA tang 2, chi train + danh gia rieng tang 1
#   (Healthy vs Sick). Dung de thi nghiem nhanh tang 1.
ONLY_STAGE1   = False
# ONLY_STAGE2 = True  -> BO QUA tang 1, chi train + danh gia rieng tang 2
#   (Myo vs Neuro tren mau Sick THAT). Dung de thi nghiem nhanh tang 2.
ONLY_STAGE2   = False
# LUU Y: KHONG dat ca hai cung True (loai tru nhau). Ca hai False = chay 2 tang.
assert not (ONLY_STAGE1 and ONLY_STAGE2), \
    "ONLY_STAGE1 va ONLY_STAGE2 khong duoc cung True (loai tru nhau)."

# Tang 1 (Healthy vs Sick): feature pho thong
STAGE1_FEATS  = ['MAV', 'RMS', 'VAR', 'WL', 'SD', 'ZC', 'SSC', 'WAMP',
                 'IEMG', 'SSI', 'MV', 'MFL', 'DAMV', 'MNF', 'MDF', 'PKF']

# Tang 2 (Myo vs Neuro): feature bien do manh (Cohen d cao)
#   MMAV d=1.148, RMS/SD d=1.233, P2P d=1.038, TURN_AMP d=0.926, EWL d=0.670
STAGE2_FEATS  = ['RMS',  'ZC', 'SSC','TURN_AMP',
                 'MMAV', 'EWL','VAR', 'SKEW', 'KURT']

LABEL_MAP     = {'Healthy': 0, 'Myopathy': 1, 'Neuropathy': 2}
CLASS_NAMES   = ['Healthy', 'Myopathy', 'Neuropathy']
N_CLASSES     = 3

# Population coding
N_POP         = 14
SIGMA_POP     = 1.0 / (N_POP - 1) * 1.5

# TTFS spike encoding
T_STEPS       = 40
TTFS_THR      = 0.05

# Model
HIDDEN        = 200
BETA_INIT     = 0.9
THRESHOLD     = 0.4

# Train (mac dinh / TANG 1)
EPOCHS        = 300
LR            = 5e-4
BATCH         = 32
PATIENCE      = 80
GAMMA         = 2.0     # FocalLoss gamma mac dinh (tang 1)

# ── Sieu tham so RIENG cho TANG 2 (Myo vs Neuro) ──────────────────────────────
# Tang 2 la bai toan NHI PHAN don gian hon -> mang nho hon, it overfit hon.
S2_HIDDEN     = 64      # nho hon HIDDEN tang 1 (200): bai toan 2 lop, it feature
S2_LR         = 1e-3    # LR cao hon: hoi tu nhanh tren bai toan don gian
S2_GAMMA      = 1.5     # gamma thap hon: 2 lop kha can bang, bot ep mau kho
S2_EPOCHS     = 300
S2_PATIENCE   = 100     # kien nhan hon: cho tim minimum tot

SPLIT_SEED    = 42
TRAIN_SEED    = 42
TRAIN_RATIO   = 0.70
VALID_RATIO   = 0.15
TEST_RATIO    = 0.15

torch.manual_seed(TRAIN_SEED)
np.random.seed(SPLIT_SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
def _get_subject_id(filename: str) -> str:
    base = os.path.splitext(os.path.basename(filename))[0]
    m = re.search(r'(\d+)\s*_\s*(\d+)', base)
    return f"{m.group(1)}_{m.group(2)}" if m else base


def load_dataset(csv_path: str):
    """Tra ve df-feature (giu ca cot), y (0/1/2), keys."""
    df = pd.read_csv(csv_path)
    df = df[df['class'].isin(LABEL_MAP.keys())].reset_index(drop=True)
    y    = df['class'].map(LABEL_MAP).to_numpy(dtype=np.int64)
    subj = df['filename'].apply(_get_subject_id).to_numpy()
    keys = np.array([f"{c}_{s}" for c, s in zip(df['class'], subj)])
    return df, y, keys


# ─────────────────────────────────────────────────────────────────────────────
# CHUẨN HÓA + POPULATION CODING + TTFS + ATan + LIF  (giong file 03)
# ─────────────────────────────────────────────────────────────────────────────
def fit_normalizer(X):
    p1  = np.percentile(X, 1, axis=0).astype(np.float32)
    p99 = np.percentile(X, 99, axis=0).astype(np.float32)
    denom = np.where(p99 - p1 > 0, p99 - p1, 1.0).astype(np.float32)
    return p1, denom


def apply_normalizer(X, p1, denom):
    return np.clip((X - p1) / denom, 0.0, 1.0).astype(np.float32)


MU_POP = np.linspace(0.0, 1.0, N_POP, dtype=np.float32)


def population_encode(X_norm):
    x_exp = X_norm[:, :, None]; mu = MU_POP[None, None, :]
    resp = np.exp(-((x_exp - mu) ** 2) / (2 * SIGMA_POP ** 2))
    return resp.reshape(X_norm.shape[0], -1).astype(np.float32)


def ttfs_spikes(rate, T, thr=TTFS_THR):
    t_fire = ((1.0 - rate) * (T - 1)).long().clamp(0, T - 1)
    spikes = torch.zeros(T, *rate.shape, device=rate.device)
    spikes.scatter_(0, t_fire.unsqueeze(0), 1.0)
    return spikes * (rate > thr).float().unsqueeze(0)


class AtanSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v, thr):
        ctx.save_for_backward(v - thr)
        return (v >= thr).float()

    @staticmethod
    def backward(ctx, grad_out):
        (dv,) = ctx.saved_tensors
        return grad_out / (1.0 + (math.pi * dv) ** 2), None


def spike(v, thr=THRESHOLD):
    return AtanSpike.apply(v, thr)


class TemporalLIF(nn.Module):
    def __init__(self, n_in, n_out, beta=BETA_INIT, thr=THRESHOLD):
        super().__init__()
        self.fc  = nn.Linear(n_in, n_out, bias=False)
        self.thr = thr
        raw = float(np.log(beta / (1.0 - beta + 1e-8)))
        self.raw_beta = nn.Parameter(torch.full((n_out,), raw))

    def forward(self, x_seq):
        T, B, _ = x_seq.shape
        beta = torch.sigmoid(self.raw_beta)
        mem  = torch.zeros(B, self.fc.out_features, device=x_seq.device)
        spk  = torch.zeros_like(mem)
        out  = []
        for t in range(T):
            cur = self.fc(x_seq[t])
            mem = beta * mem + cur - self.thr * spk
            spk = spike(mem, self.thr)
            out.append(spk)
        return torch.stack(out, dim=0)


class PopSNN(nn.Module):
    """SNN nhi phan/da lop tong quat (n_out tham so hoa)."""
    def __init__(self, n_in, n_out, n_hidden=HIDDEN, T=T_STEPS):
        super().__init__()
        self.T = T
        self.lif1 = TemporalLIF(n_in, n_hidden)
        self.lif2 = TemporalLIF(n_hidden, n_out)

    def forward(self, rate):
        spk1 = self.lif1(ttfs_spikes(rate, self.T))
        spk2 = self.lif2(spk1)
        out  = spk2.sum(dim=0) / self.T
        return torch.log_softmax(out, dim=1)


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# LOSS
# ─────────────────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma; self.weight = weight

    def forward(self, log_prob, target):
        prob = log_prob.exp()
        p_t  = prob.gather(1, target.view(-1, 1)).squeeze(1)
        fl   = -((1 - p_t) ** self.gamma) * p_t.log()
        if self.weight is not None:
            fl = fl * self.weight[target]
        return fl.mean()


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN / EVAL  (mot tang nhi phan)
# ─────────────────────────────────────────────────────────────────────────────
def run_epoch(model, opt, crit, X, y, training=True):
    model.train(training)
    idx = np.random.permutation(len(X)) if training else np.arange(len(X))
    losses, lp_all = [], []
    for i in range(0, len(X), BATCH):
        b = idx[i:i + BATCH]
        xb = torch.tensor(X[b], device=device)
        yb = torch.tensor(y[b], device=device)
        lp = model(xb)
        loss = crit(lp, yb)
        if training:
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        losses.append(loss.item()); lp_all.append(lp.detach().cpu())
    return float(np.mean(losses)), torch.cat(lp_all, dim=0)


def get_probs(model, X):
    model.eval()
    with torch.no_grad():
        out = [model(torch.tensor(X[i:i + BATCH], device=device)).exp().cpu()
               for i in range(0, len(X), BATCH)]
    return torch.cat(out, dim=0).numpy()


def train_stage(name, n_in, n_out, X_tr, y_tr, X_vl, y_vl, cls_w, seed=TRAIN_SEED,
                hidden=HIDDEN, lr=LR, gamma=GAMMA, epochs=EPOCHS, patience=PATIENCE):
    """Train 1 tang. Sieu tham so co the override rieng cho tung tang."""
    torch.manual_seed(seed); np.random.seed(seed)
    model = PopSNN(n_in, n_out, n_hidden=hidden).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit  = FocalLoss(gamma=gamma, weight=cls_w)
    # KHONG early-stop: chay het toan bo `epochs`. Van luu model tot nhat theo val
    # (best checkpoint) de tra ve — `patience` giu lai cho tuong thich chu ky goi.
    best_f1, best_state = 0.0, None
    print(f"\n  [TANG: {name}] train PopSNN({n_in}->{hidden}->{n_out}) "
          f"lr={lr} gamma={gamma}  [chay het {epochs} ep, KHONG early-stop] ...")
    for ep in range(1, epochs + 1):
        tr_loss, _     = run_epoch(model, opt, crit, X_tr, y_tr, True)
        vl_loss, vl_lp = run_epoch(model, opt, crit, X_vl, y_vl, False)
        sched.step()
        vl_f1 = f1_score(y_vl, vl_lp.argmax(1).numpy(), average='macro', zero_division=0)
        if ep % 20 == 0 or ep == 1:
            print(f"      Ep {ep:3d} | tr={tr_loss:.3f} vl={vl_loss:.3f} vl_F1={vl_f1*100:.1f}%")
        if vl_f1 > best_f1:
            best_f1 = vl_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    print(f"      Done {epochs} ep (no early-stop) | Best Val F1={best_f1*100:.1f}%")
    model.load_state_dict(best_state)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: chuan bi 1 nhom feature (normalize tu train + encode)
# ─────────────────────────────────────────────────────────────────────────────
def prep_features(df, feats, tr_m, vl_m, te_m):
    X = df[feats].to_numpy(dtype=np.float32)
    p1, denom = fit_normalizer(X[tr_m])
    enc = lambda m: population_encode(apply_normalizer(X[m], p1, denom))
    return enc(tr_m), enc(vl_m), enc(te_m), (p1, denom)


def print_binary_cm(model, X_te, y_te_bin, class_names, title):
    """In confusion matrix + bao cao cho 1 tang nhi phan tren TEST."""
    pred = get_probs(model, X_te).argmax(1)
    f1   = f1_score(y_te_bin, pred, average='macro', zero_division=0)
    acc  = (pred == y_te_bin).mean()
    cm   = confusion_matrix(y_te_bin, pred, labels=[0, 1])
    print(f"\n  {'-'*60}")
    print(f"  CONFUSION MATRIX — {title}  (TEST, {len(y_te_bin)} mau)")
    print(f"  {'-'*60}")
    cw = 14
    print(f"  {'':16s}" + "".join(f"Pred {n:<{cw-5}}" for n in class_names))
    sep = "  " + "-" * (16 + cw * 2)
    print(sep)
    for i, n in enumerate(class_names):
        row = f"  True {n:<11s}"
        for j in range(2):
            row += f"{cm[i, j]:<{cw}d}"
        print(row)
    print(sep)
    print()
    print(classification_report(y_te_bin, pred, target_names=class_names,
                                digits=3, zero_division=0))
    print(f"  Accuracy = {acc*100:.1f}%   F1 Macro = {f1*100:.1f}%")


def print_cm_by_subject(model, X_te, y_te_bin, keys_te, class_names, title):
    """
    Danh gia THEO NGUOI: gop cac segment cua moi subject (theo keys_te) ->
    majority vote -> 1 nhan/nguoi. In CM + bao cao tren so NGUOI.
    keys_te: mang dinh danh subject cho tung segment (cung do dai y_te_bin).
    """
    pred_seg = get_probs(model, X_te).argmax(1)
    uniq = sorted(set(keys_te.tolist()))
    y_subj, pred_subj = [], []
    for k in uniq:
        m = (keys_te == k)
        # nhan that cua nguoi = nhan da so segment (thuc te dong nhat)
        y_subj.append(np.bincount(y_te_bin[m], minlength=2).argmax())
        pred_subj.append(np.bincount(pred_seg[m], minlength=2).argmax())
    y_subj, pred_subj = np.array(y_subj), np.array(pred_subj)

    f1  = f1_score(y_subj, pred_subj, average='macro', zero_division=0)
    acc = (pred_subj == y_subj).mean()
    cm  = confusion_matrix(y_subj, pred_subj, labels=[0, 1])
    print(f"\n  {'-'*60}")
    print(f"  CONFUSION MATRIX [THEO NGUOI] — {title}  ({len(uniq)} nguoi)")
    print(f"  {'-'*60}")
    cw = 14
    print(f"  {'':16s}" + "".join(f"Pred {n:<{cw-5}}" for n in class_names))
    sep = "  " + "-" * (16 + cw * 2)
    print(sep)
    for i, n in enumerate(class_names):
        row = f"  True {n:<11s}"
        for j in range(2):
            row += f"{cm[i, j]:<{cw}d}"
        print(row)
    print(sep)
    print()
    print(classification_report(y_subj, pred_subj, target_names=class_names,
                                digits=3, zero_division=0))
    print(f"  Accuracy [nguoi] = {acc*100:.1f}%   F1 Macro [nguoi] = {f1*100:.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 70)
    print("  LAYER 07 [TWO-STAGE] — (Healthy/Sick) -> (Myopathy/Neuropathy)")
    print("=" * 70)
    print(f"  Device  : {device}")
    print(f"  Tang 1 (Healthy/Sick): {len(STAGE1_FEATS)} feat -> N_IN={len(STAGE1_FEATS)*N_POP}")
    print(f"  Tang 2 (Myo/Neuro)   : {len(STAGE2_FEATS)} feat {STAGE2_FEATS}")
    print(f"                          -> N_IN={len(STAGE2_FEATS)*N_POP}")

    df, y_all, keys_all = load_dataset(CSV_PATH)
    print(f"\n  Tổng số segment: {len(y_all)}")

    # ── Subject-level holdout split ───────────────────────────────────────────
    uniq_keys   = np.unique(keys_all)
    uniq_labels = np.array([y_all[keys_all == k][0] for k in uniq_keys])
    for cls_id, cname in enumerate(CLASS_NAMES):
        print(f"    {cname:12s}: {int((uniq_labels == cls_id).sum())} subject")

    k_tr, k_tmp, _, lb_tmp = train_test_split(
        uniq_keys, uniq_labels, test_size=(VALID_RATIO + TEST_RATIO),
        stratify=uniq_labels, random_state=SPLIT_SEED)
    vt = VALID_RATIO / (VALID_RATIO + TEST_RATIO)
    k_vl, k_te = train_test_split(k_tmp, test_size=(1.0 - vt),
                                  stratify=lb_tmp, random_state=SPLIT_SEED)
    tr_m = np.isin(keys_all, k_tr)
    vl_m = np.isin(keys_all, k_vl)
    te_m = np.isin(keys_all, k_te)

    y_tr, y_vl, y_te = y_all[tr_m], y_all[vl_m], y_all[te_m]
    keys_te = keys_all[te_m]                     # dinh danh subject cho moi segment test

    # ── NHAN cho moi tang ─────────────────────────────────────────────────────
    # Tang 1: 0=Healthy, 1=Sick (Myo hoac Neuro)
    to_sick = lambda y: (y != LABEL_MAP['Healthy']).astype(np.int64)
    y1_tr, y1_vl = to_sick(y_tr), to_sick(y_vl)
    # Tang 2: chi mau Sick. 0=Myopathy, 1=Neuropathy
    to_myoneu = lambda y: (y == LABEL_MAP['Neuropathy']).astype(np.int64)  # Myo->0, Neu->1
    sick_tr = (y_tr != LABEL_MAP['Healthy'])
    sick_vl = (y_vl != LABEL_MAP['Healthy'])

    # ── Chuan bi feature 2 tang ───────────────────────────────────────────────
    X1_tr, X1_vl, X1_te, _ = prep_features(df, STAGE1_FEATS, tr_m, vl_m, te_m)
    X2_tr, X2_vl, X2_te, _ = prep_features(df, STAGE2_FEATS, tr_m, vl_m, te_m)

    # Du lieu tang 2 (chi mau Sick)
    X2_tr_s, y2_tr_s = X2_tr[sick_tr], to_myoneu(y_tr[sick_tr])
    X2_vl_s, y2_vl_s = X2_vl[sick_vl], to_myoneu(y_vl[sick_vl])
    c2 = np.bincount(y2_tr_s, minlength=2).astype(np.float32)
    w2 = torch.tensor(c2.sum() / (2 * c2), device=device)

    # nhan test cho moi tang
    y1_te = to_sick(y_te)                       # tang 1: 0=Healthy, 1=Sick
    sick_te = (y_te != LABEL_MAP['Healthy'])
    y2_te_s = to_myoneu(y_te[sick_te])          # tang 2: 0=Myo, 1=Neuro (mau Sick that)

    # ===================== TANG 1: Healthy vs Sick =====================
    # (Bo qua neu ONLY_STAGE2 — chi muon thi nghiem rieng tang 2)
    m1 = None
    if not ONLY_STAGE2:
        print("\n  >>> TRAIN TANG 1 (Healthy vs Sick) <<<")
        c1 = np.bincount(y1_tr, minlength=2).astype(np.float32)
        w1 = torch.tensor(c1.sum() / (2 * c1), device=device)
        m1 = train_stage("Healthy/Sick", X1_tr.shape[1], 2,
                         X1_tr, y1_tr, X1_vl, y1_vl, w1)
        # CM tang 1 theo NGUOI (majority vote segment cua moi subject)
        print_cm_by_subject(m1, X1_te, y1_te, keys_te,
                            ['Healthy', 'Sick'], "TANG 1: Healthy vs Sick")

    # ===================== CHE DO CHI TANG 1 ===========================
    # (CM tang 1 da in o tren). Chi luu model + thoat, KHONG train tang 2.
    if ONLY_STAGE1:
        torch.save(m1.state_dict(), 'layer05_stage1_only.pth')
        print(f"\n  Model tang 1: layer05_stage1_only.pth")
        import sys; sys.exit(0)

    # ===================== TANG 2: Myo vs Neuro (chi mau Sick) =========
    # Dung sieu tham so RIENG cho tang 2 (mang nho hon, LR cao hon, gamma thap hon)
    print("\n  >>> TRAIN TANG 2 (Myopathy vs Neuropathy) <<<")
    m2 = train_stage("Myo/Neuro", X2_tr.shape[1], 2,
                     X2_tr_s, y2_tr_s, X2_vl_s, y2_vl_s, w2,
                     hidden=S2_HIDDEN, lr=S2_LR, gamma=S2_GAMMA,
                     epochs=S2_EPOCHS, patience=S2_PATIENCE)
    # CM tang 2 theo NGUOI (majority vote segment cua moi subject Sick)
    print_cm_by_subject(m2, X2_te[sick_te], y2_te_s, keys_te[sick_te],
                        ['Myopathy', 'Neuropathy'], "TANG 2: Myopathy vs Neuropathy")

    # ===================== CHE DO CHI TANG 2 ===========================
    # (CM tang 2 da in o tren). Chi luu model + thoat.
    if ONLY_STAGE2:
        torch.save(m2.state_dict(), 'layer05_stage2_only.pth')
        print(f"\n  Model tang 2: layer05_stage2_only.pth")
        import sys; sys.exit(0)

    # ── Luu ca 2 model (moi tang da co CM rieng o tren: segment + nguoi) ──────
    torch.save({'stage1': m1.state_dict(), 'stage2': m2.state_dict()},
               'layer05_two_stage_snn.pth')
    print(f"\n  Model: layer05_two_stage_snn.pth")
