"""
app.py - GUI Streamlit DA TRANG: Chan doan EMG + Lich su + Loi khuyen Gemini
=============================================================================
Chay:   python -m streamlit run app.py
Yeu cau: pip install streamlit google-genai

Dieu huong (sidebar): 2 trang rieng
  - 🩺 Chan doan : upload .asc -> kiem tra mien -> phan loai -> loi khuyen AI
  - 📚 Lich su   : bang luu tru cac lan chan doan (history.json)

Tinh nang khac: API key nho qua F5 (bo nho server), luu lich su ra file.
"""

import os
import json
import base64
from datetime import datetime

import streamlit as st
import pandas as pd

import emg_infer
import advisor

HERE         = os.path.dirname(os.path.abspath(__file__))
LOGO         = os.path.join(HERE, "ft.png")
LOGO2        = os.path.join(HERE, "logo_Truong.jpg")     # logo goc phai header
HISTORY_FILE = os.path.join(HERE, "history.json")
HISTORY_MAX  = 200

VI = {"Healthy": "Bình thường", "Myopathy": "Bệnh cơ (Myopathy)", "Neuropathy": "Bệnh thần kinh (Neuropathy)"}

st.set_page_config(page_title="Hệ thống Chẩn đoán EMG", page_icon="🩺",
                   layout="centered", initial_sidebar_state="expanded")


# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.stApp { background-color: #eef2f7; }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stAppDeployButton"] { display: none; }
[data-testid="stMainMenu"], #MainMenu { display: none; }
[data-testid="stToolbarActions"] { display: none; }
[data-testid="stSidebarCollapsedControl"] { display: flex !important; }
.block-container { padding-top: 1.1rem; padding-bottom: 3rem; max-width: 940px; }
h1, h2, h3 { color: #0b3d5c; }

.app-header {
  display:flex; align-items:center; gap:18px;
  background: linear-gradient(135deg,#0b3d5c 0%, #0e7c86 100%);
  padding:18px 26px; border-radius:16px; margin-bottom:20px;
  box-shadow:0 8px 26px rgba(11,61,92,.28);
}
.app-logo { height:62px; width:auto; border-radius:12px; background:#fff; padding:6px; box-shadow:0 2px 8px rgba(0,0,0,.15); }
.corner-logo { position:fixed; top:12px; right:22px; z-index:1000; height:96px; width:auto;
  background:#fff; border-radius:14px; padding:6px; box-shadow:0 3px 14px rgba(0,0,0,.20); }
.app-title { color:#ffffff; font-size:1.5rem; font-weight:800; line-height:1.18; letter-spacing:.3px; margin:0; }
.app-sub { color:#cfe8ec; font-size:.92rem; margin-top:5px; }

[data-testid="stVerticalBlockBorderWrapper"] {
  background:#ffffff; border-radius:14px; border:1px solid #e2e9f1;
  box-shadow:0 2px 12px rgba(11,61,92,.06);
}
.sec-title { font-size:1.08rem; font-weight:700; color:#0b3d5c; margin:.1rem 0 .7rem; }

.stButton > button {
  background: linear-gradient(135deg,#0e7c86,#0b6b73); color:#fff; border:0;
  border-radius:10px; padding:.55rem 1.2rem; font-weight:600;
  box-shadow:0 4px 14px rgba(14,124,134,.32);
}
.stButton > button:hover { filter:brightness(1.08); color:#fff; }

.badge { display:inline-block; padding:.45rem 1.1rem; border-radius:999px; font-weight:800; font-size:1.12rem; letter-spacing:.3px; }
.badge-ok   { background:#e6f7ee; color:#0a7a3f; border:1px solid #9fdcc0; }
.badge-warn { background:#fff3e0; color:#b3590a; border:1px solid #f3c38a; }

.ai-badge { display:inline-block; background:#eaf3ff; color:#1565c0; border:1px solid #b9d8f5;
  border-radius:999px; padding:.22rem .8rem; font-size:.78rem; font-weight:700; white-space:nowrap; }
.ai-empty { background:#f3f7fb; border:1px dashed #c4d4e3; border-radius:12px; padding:20px;
  text-align:center; color:#5a7184; font-size:.92rem; }
[data-testid="stChatMessage"] { background:#f5f9ff; border:1px solid #dbe8f5; border-radius:14px; padding:10px 14px; margin-top:6px; }
[data-testid="stChatMessage"] h3 { color:#0b3d5c; font-size:1.02rem; margin-top:.6rem; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Logo + header (hien tren moi trang)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def _logo_b64(path):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return None

_b64   = _logo_b64(LOGO)
_b64_2 = _logo_b64(LOGO2)
_logo_html = f'<img src="data:image/png;base64,{_b64}" class="app-logo"/>' if _b64 else ""

# Logo Truong: co dinh o goc phai man hinh (NGOAI khung xanh)
if _b64_2:
    st.markdown(f'<img src="data:image/jpeg;base64,{_b64_2}" class="corner-logo"/>',
                unsafe_allow_html=True)

st.markdown(f"""
<div class="app-header">
  {_logo_html}
  <div>
    <div class="app-title">HỆ THỐNG CHẨN ĐOÁN EMG THẦN KINH – CƠ</div>
    <div class="app-sub">Phân loại Healthy · Myopathy · Neuropathy bằng SNN 2 tầng &nbsp;|&nbsp; Tư vấn bằng AI (Gemini)</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Bo nho server + lich su + model
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def _key_store():
    return {"gemini": ""}

def load_history():
    try:
        with open(HISTORY_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def save_history(items):
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(items[-HISTORY_MAX:], f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def add_history(rec):
    items = load_history()
    items.append(rec)
    save_history(items)

@st.cache_resource(show_spinner="Đang nạp mô hình SNN 2 tầng...")
def load_predictor():
    return emg_infer.build_predictor()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers render
# ─────────────────────────────────────────────────────────────────────────────
def render_diagnosis(result):
    cls = result["predicted"]
    badge_cls = "badge-ok" if cls == "Healthy" else "badge-warn"
    icon = "✅" if cls == "Healthy" else "⚠️"
    st.markdown('<div class="sec-title">🩺 Kết quả chẩn đoán</div>', unsafe_allow_html=True)
    st.markdown(f'<span class="badge {badge_cls}">{icon}&nbsp; {VI[cls]}</span>', unsafe_allow_html=True)
    if cls != "Healthy":
        st.caption("Mô hình nghi ngờ dấu hiệu bệnh lý — nên đi khám chuyên khoa thần kinh - cơ.")
    else:
        st.caption("Mô hình phân loại tín hiệu là bình thường.")
    with st.expander("Chi tiết kỹ thuật"):
        st.write(f"- Số segment phân tích: **{result['n_segments']}** (mỗi segment 1 giây)")
        st.write(f"- Tỷ lệ segment đồng thuận với nhãn cuối: **{result['seg_agreement']*100:.0f}%**")
        st.write(f"- P(Sick) trung bình ở Tầng 1: **{result['stage1_sick']*100:.1f}%**")
        st.write(f"- Số mẫu tín hiệu gốc: **{result['raw_samples']:,}**")
        st.caption("Tín hiệu đã lọc (Notch 50/100 Hz + bandpass 16-5000 Hz), 1 giây đầu:")
        st.line_chart(pd.DataFrame({"EMG (đã lọc)": result["signal_preview"]}), height=180)


def render_system_profile(predictor):
    p = emg_infer.system_profile(predictor)
    st.markdown("**📋 Đặc điểm dữ liệu hệ thống yêu cầu**")
    st.markdown(
        f"- Loại cơ: **{p['muscle']}**\n"
        f"- Tần số lấy mẫu: **{p['fs']:,} Hz** · thời lượng **≥ {p['duration_s']} giây** "
        f"(**{p['n_samples']:,} mẫu**)\n"
        f"- Số đặc trưng/segment: **{p['n_features']}**\n"
        f"- Yêu cầu: **điểm tương thích ≥ {p['compat_min']*100:.0f}%** "
        f"(khoảng cách Mahalanobis tới phân bố huấn luyện)"
    )
    st.dataframe(pd.DataFrame(p["feature_ranges"]), hide_index=True, width="stretch")


# ─────────────────────────────────────────────────────────────────────────────
# Bo nho API key dung chung (sidebar dung o cuoi file, sau khi dinh nghia trang)
# ─────────────────────────────────────────────────────────────────────────────
store = _key_store()


# ═════════════════════════════════════════════════════════════════════════════
# TRANG 1 — CHAN DOAN
# ═════════════════════════════════════════════════════════════════════════════
def page_diagnosis():
    predictor = load_predictor()

    # BOX — Tai file
    with st.container(border=True):
        st.markdown('<div class="sec-title">📤 Tải tín hiệu EMG</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("Chọn file `.asc` (5 giây @ 32768 Hz)", type=["asc"],
                                    label_visibility="collapsed")
        st.caption("Định dạng: file `.asc` chứa tín hiệu EMG một cột, ≥ 163,840 mẫu (5 giây).")

    if uploaded is None:
        st.info("⬆️ Hãy tải lên một file `.asc` để bắt đầu chẩn đoán.")
        with st.expander("📋 Xem đặc điểm dữ liệu hệ thống"):
            render_system_profile(predictor)
        return

    with st.spinner("Đang lọc, trích đặc trưng và phân loại..."):
        result = emg_infer.analyze(predictor, uploaded.getvalue())

    # Ghi lich su khi doi file
    if st.session_state.get("file_name") != uploaded.name:
        st.session_state["file_name"] = uploaded.name
        st.session_state.pop("advice", None)
        if result["ok"]:
            add_history({
                "time":   datetime.now().strftime("%Y-%m-%d %H:%M"),
                "file":   uploaded.name,
                "result": result["predicted_vi"],
                "valid":  bool(result["domain"]["valid"]),
                "compat": float(result["domain"]["compat"]),
            })

    if not result["ok"]:
        st.error(f"❌ Không phân tích được file: {result['error']}")
        return

    dom = result["domain"]

    # Du lieu KHONG phu hop
    if not dom["valid"]:
        with st.container(border=True):
            st.error(
                f"⚠️ **DỮ LIỆU EMG KHÔNG PHÙ HỢP VỚI HỆ THỐNG** — "
                f"điểm tương thích chỉ **{dom['compat']*100:.0f}%** (cần ≥ 50%)."
            )
            st.markdown(
                "Tín hiệu tải lên có đặc điểm **khác** với dữ liệu hệ thống được huấn luyện "
                "(khác loại cơ, khác thiết bị đo, khác thang biên độ/tần số). "
                "Kết quả phân loại sẽ **không đáng tin**. Vui lòng:\n"
                "- Kiểm tra lại file (đúng định dạng EMG, FS = 32768 Hz, ≥ 5 giây), **hoặc**\n"
                "- Đưa lại bộ dữ liệu phù hợp với đặc điểm hệ thống bên dưới."
            )
            off = dom["offenders"][:8]
            if off:
                st.markdown(f"**Các đặc trưng lệch khỏi khoảng huấn luyện ({dom['n_out']}/{len(emg_infer._FN)}):**")
                st.dataframe(pd.DataFrame([{
                    "Đặc trưng": o["feat"],
                    "Giá trị file": f"{o['value']:,.1f}",
                    "Khoảng mong đợi": f"{o['lo']:,.1f} – {o['hi']:,.1f}",
                    "Lệch": o["dir"],
                } for o in off]), hide_index=True, width="stretch")
        with st.container(border=True):
            render_system_profile(predictor)
        with st.expander("Vẫn xem kết quả phân loại (KHÔNG khuyến nghị — dữ liệu không phù hợp)"):
            render_diagnosis(result)
        return

    # Du lieu hop le
    st.success(f"✓ Dữ liệu phù hợp với hệ thống · điểm tương thích **{dom['compat']*100:.0f}%**")
    with st.container(border=True):
        render_diagnosis(result)
    with st.expander("📋 Đặc điểm dữ liệu hệ thống (tham khảo)"):
        render_system_profile(predictor)

    # BOX — Loi khuyen AI
    with st.container(border=True):
        hc1, hc2 = st.columns([3, 1])
        hc1.markdown('<div class="sec-title">💬 Trợ lý Tư vấn AI</div>', unsafe_allow_html=True)
        hc2.markdown('<div style="text-align:right;padding-top:4px">'
                     '<span class="ai-badge">⚡ Gemini Flash</span></div>', unsafe_allow_html=True)
        st.caption("Lời khuyên chăm sóc cá nhân hoá dựa trên kết quả chẩn đoán, sinh tự động bằng AI.")
        if st.button("✨  Sinh lời khuyên cho người bệnh", type="primary", width="stretch"):
            with st.spinner("Trợ lý AI đang phân tích kết quả và soạn lời khuyên..."):
                st.session_state["advice"] = advisor.get_advice(
                    diagnosis_vi=result["predicted_vi"], confidence=result["confidence"],
                    probs=result["probs"], api_key=(store.get("gemini") or None),
                )
        adv = st.session_state.get("advice")
        if adv and adv["ok"]:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(adv["text"])
                st.caption(f"💡 Sinh bởi {adv.get('model', 'Gemini')} · nội dung mang tính tham khảo")
        elif adv and not adv["ok"]:
            st.error(adv["error"])
        else:
            st.markdown('<div class="ai-empty">🤖 Nhấn nút phía trên để nhận lời khuyên về '
                        'chăm sóc, sinh hoạt và theo dõi triệu chứng từ trợ lý AI.</div>',
                        unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# TRANG 2 — LICH SU
# ═════════════════════════════════════════════════════════════════════════════
def page_history():
    items = load_history()
    with st.container(border=True):
        c1, c2 = st.columns([3, 1])
        c1.markdown('<div class="sec-title">📚 Lịch sử chẩn đoán</div>', unsafe_allow_html=True)
        if c2.button("🗑️ Xóa lịch sử", width="stretch"):
            save_history([])
            st.rerun()
        if not items:
            st.caption("Chưa có lịch sử. Kết quả các lần chẩn đoán ở trang Chẩn đoán sẽ được lưu tại đây.")
            return
        df = pd.DataFrame(items[::-1])
        df["valid"] = df["valid"].map({True: "✓ Hợp lệ", False: "✗ Không hợp lệ"})
        df = df.drop(columns=["compat"], errors="ignore")
        df = df.rename(columns={"time": "Thời gian", "file": "File",
                                "result": "Kết quả", "valid": "Trạng thái"})
        st.dataframe(df, hide_index=True, width="stretch", height=420)
        st.caption(f"Tổng cộng **{len(items)}** lần chẩn đoán đã lưu.")


# ─────────────────────────────────────────────────────────────────────────────
# Dieu huong da trang (menu o sidebar — bam de chuyen trang)
# ─────────────────────────────────────────────────────────────────────────────
diag_page = st.Page(page_diagnosis, title="Chẩn đoán", icon="🩺", default=True)
hist_page = st.Page(page_history,   title="Lịch sử",   icon="📚")
pg = st.navigation([diag_page, hist_page], position="hidden")     # an menu mac dinh (tu dung ben duoi)

with st.sidebar:
    # GHI CONG — tren cung sidebar (thay cho logo fetel nho)
    st.markdown(
        '<div style="font-size:.9rem;color:#0b3d5c;line-height:1.75;margin-bottom:2px">'
        '🎓 <b>Khóa luận tốt nghiệp</b><br>'
        '<span style="color:#33506a"><b>Sinh viên:</b> Phạm Xuân Khang<br>'
        '<b>MSSV:</b> 22200080<br>'
        '<b>GVHD:</b> Nguyễn Thị Thiên Trang</span>'
        '</div>', unsafe_allow_html=True)
    st.divider()

    # DIEU HUONG TRANG
    st.page_link(diag_page, label="Chẩn đoán", icon="🩺")
    st.page_link(hist_page, label="Lịch sử",   icon="📚")
    st.divider()

    # CAU HINH
    st.header("⚙️ Cấu hình")
    _default_key = store.get("gemini") or os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")
    api_key = st.text_input(
        "Google Gemini API key", value=_default_key, type="password",
        help="Lấy MIỄN PHÍ tại aistudio.google.com/app/apikey. Nhập 1 lần, nhớ qua F5 (chỉ mất khi tắt server).")
    if api_key:
        store["gemini"] = api_key
    if store.get("gemini"):
        st.caption("🔑 API key đã lưu (giữ qua F5).")
    st.caption("Model lời khuyên: **Gemini Flash** (free tier)")
    st.divider()
    st.caption("⚠️ Công cụ **sàng lọc hỗ trợ**, không thay thế chẩn đoán của bác sĩ.")

pg.run()
