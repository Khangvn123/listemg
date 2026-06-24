"""
advisor.py - Goi Google Gemini (free tier) sinh loi khuyen tu ket qua chan doan EMG
====================================================================================
Dung SDK chinh thuc moi `google-genai` (goi: from google import genai).
Cai dat:  pip install google-genai
API key:  lay MIEN PHI tai https://aistudio.google.com/app/apikey  (khong can the tin dung)
          -> dat bien moi truong GEMINI_API_KEY (hoac GOOGLE_API_KEY), hoac nhap o sidebar.

LUU Y AN TOAN: day la cong cu HO TRO SANG LOC tu tin hieu EMG, KHONG phai
chan doan y khoa. Loi khuyen luon kem khuyen cao gap bac si chuyen khoa.
"""

import os

# Cac model Flash mien phi, thu lan luot (ten model Gemini thay doi theo thoi gian)
MODEL_CANDIDATES = ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-1.5-flash", "gemini-flash-latest"]

SYSTEM_PROMPT = """\
Ban la tro ly y khoa ho tro giai thich ket qua SANG LOC than kinh - co tu tin hieu EMG \
(dien co do), phuc vu nguoi benh noi tieng Viet. Ket qua dau vao do mot mo hinh hoc may \
(mang neuron xung - SNN) phan loai, KHONG phai chan doan cua bac si.

Nguyen tac:
- Tra loi bang tieng Viet, ro rang, dong cam, de hieu voi nguoi khong chuyen.
- LUON nhan manh: day chi la ket qua sang loc tu mo hinh AI, khong thay the tham kham \
  lam sang; nguoi benh CAN gap bac si chuyen khoa than kinh - co de duoc chan doan chinh thuc.
- TUYET DOI KHONG ke don thuoc cu the, lieu luong, hay chi dinh dieu tri thay bac si.
- Loi khuyen PHAI PHU HOP RIENG voi tinh trang da phan loai (Healthy / Myopathy / Neuropathy):
  moi tinh trang co loi khuyen sinh hoat, van dong va dau hieu canh bao KHAC NHAU. TUYET DOI
  khong dua loi khuyen chung chung giong het nhau cho moi tinh trang.
- KHONG neu con so phan tram / do tin cay / xac suat cu the trong cau tra loi. Neu muc do
  chac chan thap, chi noi CHUNG rang ket qua chua chac chan va cang can di kham them.

Dinh huong theo tung tinh trang (tu dieu chinh ngon ngu, dung sao chep nguyen van):
- Binh thuong (Healthy): tap trung DUY TRI suc khoe co - than kinh (van dong deu dan, dinh duong,
  nghi ngoi, phong ngua); KHONG noi nhu nguoi benh dang mac benh.
- Benh co (Myopathy): bao ve co bap, TRANH gang suc / qua tai; uu tien van dong nhe va vat ly
  tri lieu phu hop; theo doi yeu co tien trien, met moi co, kho thuc hien dong tac.
- Benh than kinh (Neuropathy): chu y giam / roi loan cam giac -> cham soc da va ban chan, phong
  tranh te nga va chan thuong; theo doi te bi, yeu, teo co; kiem soat nguyen nhan nen neu co.

KHONG nhac lai ket qua phan loai, KHONG giai thich tinh trang la gi. Vao thang loi khuyen.
Toan bo cau tra loi viet bang tieng Viet CO DAU day du (ke ca cac tieu de muc).
Trinh bay theo cau truc Markdown, CHI gom dung 3 muc sau, dung CHINH XAC cac tieu de co dau nay:
### 1. Lời khuyên sinh hoạt & chăm sóc
### 2. Dấu hiệu cần đi khám ngay
### 3. Lưu ý quan trọng (khuyến cáo gặp bác sĩ chuyên khoa)

Giu do dai vua phai, khong lan man."""


def _prompt(diagnosis_vi: str, confidence: float, probs: dict | None) -> str:
    # Chi dua MUC DO chac chan dinh tinh (khong dua con so) -> AI khong nhac lai phan tram
    level = "thap" if confidence < 0.5 else ("trung binh" if confidence < 0.75 else "cao")
    return (
        f"Ket qua phan loai tu mo hinh EMG: **{diagnosis_vi}**\n"
        f"Muc do chac chan cua mo hinh: {level}.\n\n"
        f"Hay dua ra loi khuyen phu hop cho nguoi benh dua tren ket qua nay. "
        f"Khong nhac lai con so phan tram nao."
    )


def get_advice(diagnosis_vi: str, confidence: float, probs: dict | None = None,
               api_key: str | None = None, model: str | None = None) -> dict:
    """
    Goi Gemini sinh loi khuyen. Tra ve {ok, text, model} hoac {ok: False, error}.
    api_key: None -> doc tu GEMINI_API_KEY / GOOGLE_API_KEY (bien moi truong).
    """
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        return {"ok": False, "error": "Chua cai SDK. Chay: pip install google-genai"}

    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        return {"ok": False, "error": "Chua co API key. Lay MIEN PHI tai "
                                      "aistudio.google.com/app/apikey roi dat GEMINI_API_KEY "
                                      "hoac nhap o sidebar."}

    try:
        client = genai.Client(api_key=key)
    except Exception as e:
        return {"ok": False, "error": f"API key khong hop le: {e}"}

    # Cho phep noi dung y te (loi khuyen sang loc hop le) -> noi long bo loc an toan
    safety = [
        types.SafetySetting(category=c, threshold="BLOCK_NONE")
        for c in ("HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
                  "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT")
    ]
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        max_output_tokens=2048,
        temperature=0.7,
        safety_settings=safety,
    )
    user_prompt = _prompt(diagnosis_vi, confidence, probs)

    candidates = [model] if model else MODEL_CANDIDATES
    last_err = None
    for name in candidates:
        try:
            resp = client.models.generate_content(model=name, contents=user_prompt, config=config)
            text = (getattr(resp, "text", None) or "").strip()
            if text:
                return {"ok": True, "text": text, "model": name}
            fb = getattr(resp, "prompt_feedback", None)
            last_err = f"Phan hoi rong/bi chan (model {name}, feedback={fb})."
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            continue   # thu model tiep theo (vd ten model khong ton tai)

    return {"ok": False, "error": f"Khong sinh duoc loi khuyen. {last_err}"}
