import re
import time
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from datetime import datetime

import requests
import streamlit as st
from ddgs import DDGS


# =========================
# App Meta
# =========================
APP_VERSION = "v4.13"
CREDIT_LINE = f"RealityBot {APP_VERSION} • gebaut für Dominik"


# =========================
# Streamlit Setup
# =========================
st.set_page_config(
    page_title="RealityBot Deep-Dive",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# =========================
# Data Model
# =========================
@dataclass
class SourceDoc:
    title: str
    url: str
    snippet: str
    excerpt: str = ""


# =========================
# CSS: Responsive + WhatsApp/In-App Browser Fix
# =========================
def apply_global_css(sidebar_px_desktop: int = 460) -> None:
    st.markdown(
        f"""
<style>
html, body, [data-testid="stAppViewContainer"] {{
  overflow-x: hidden !important;
}}

/* MOBILE / TOUCH */
@media (pointer: coarse), (hover: none) {{
  section[data-testid="stSidebar"] {{
    width: 100vw !important;
    max-width: 100vw !important;
  }}
  section[data-testid="stSidebar"] > div {{
    width: 100vw !important;
    max-width: 100vw !important;
  }}

  .main .block-container {{
    padding-left: 1rem !important;
    padding-right: 1rem !important;
    padding-top: 1.25rem !important;
  }}

  h1 {{ font-size: 1.85rem !important; line-height: 1.15 !important; }}
  h2 {{ font-size: 1.3rem !important; }}
  p, li {{ font-size: 1.02rem !important; }}

  .stButton > button, .stDownloadButton > button {{
    padding-top: 0.85rem !important;
    padding-bottom: 0.85rem !important;
  }}
}}

/* DESKTOP */
@media (pointer: fine) and (min-width: 900px) {{
  section[data-testid="stSidebar"] {{
    width: {sidebar_px_desktop}px !important;
  }}
  section[data-testid="stSidebar"] > div {{
    width: {sidebar_px_desktop}px !important;
  }}
}}

section[data-testid="stSidebar"] .stMarkdown p {{
  margin-bottom: 0.55rem;
}}
</style>
        """,
        unsafe_allow_html=True,
    )


def apply_blur_main_desktop_only(locked: bool) -> None:
    # Blur nur auf Desktop (pointer fine). Auf Handy NIE blurren.
    if locked:
        st.markdown(
            """
<style>
@media (pointer: fine) {
  section.main,
  div[data-testid="stMain"],
  div[data-testid="stMainBlockContainer"],
  div[data-testid="stAppViewContainer"] section.main {
    filter: blur(4px);
    opacity: 0.16;
    pointer-events: none;
    user-select: none;
  }
}
</style>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
<style>
section.main,
div[data-testid="stMain"],
div[data-testid="stMainBlockContainer"],
div[data-testid="stAppViewContainer"] section.main {
  filter: none !important;
  opacity: 1 !important;
  pointer-events: auto !important;
  user-select: auto !important;
}
</style>
            """,
            unsafe_allow_html=True,
        )


# =========================
# Utility: remove internal markers like [1 pos 4 neg]
# =========================
_INTERNAL_TAG_RE = re.compile(
    r"\[\s*\d+\s*(?:pos|neg)\s*(?:,\s*\d+\s*(?:pos|neg)\s*)*\]",
    flags=re.IGNORECASE
)

def strip_internal_markers(text: str) -> str:
    text = _INTERNAL_TAG_RE.sub("", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


# =========================
# Robust parsing: RB markers (fixes "nicht gefunden")
# =========================
def extract_rb_block(md: str, name: str) -> str:
    # [[RB:NAME]] ... [[/RB:NAME]]
    pattern = rf"(?s)\[\[RB:{re.escape(name)}\]\]\s*(.*?)\s*\[\[/RB:{re.escape(name)}\]\]"
    m = re.search(pattern, md)
    return m.group(1).strip() if m else ""


# =========================
# DDGS Search
# =========================
def safe_ddgs_search(query: str, max_results: int = 5, retries: int = 3, backoff: float = 1.4) -> List[Dict]:
    last_err: Optional[Exception] = None
    for attempt in range(retries):
        try:
            with DDGS() as ddgs:
                return list(ddgs.text(query, max_results=max_results))
        except Exception as e:
            last_err = e
            time.sleep(backoff * (attempt + 1))
    raise last_err if last_err else RuntimeError("Unbekannter DDGS-Fehler")


def normalize_results(results: List[Dict]) -> List[SourceDoc]:
    docs: List[SourceDoc] = []
    for r in results:
        title = (r.get("title") or r.get("heading") or "Ohne Titel").strip()
        url = (r.get("href") or r.get("url") or "").strip()
        snippet = (r.get("body") or r.get("snippet") or "").strip()
        if url:
            docs.append(SourceDoc(title=title, url=url, snippet=snippet))
    return docs


# =========================
# Optional: Fetch page text (cached)
# =========================
@st.cache_data(ttl=3600)
def fetch_page_text(url: str, timeout: int = 12, max_chars: int = 7000) -> str:
    headers = {"User-Agent": "Mozilla/5.0 (RealityBot; Streamlit) AppleWebKit/537.36"}
    res = requests.get(url, headers=headers, timeout=timeout)
    res.raise_for_status()
    html = res.text

    html = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", html)
    text = re.sub(r"(?s)<.*?>", " ", html)
    text = (
        text.replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&quot;", '"')
        .replace("&#39;", "'")
    )
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_chars:
        text = text[:max_chars] + "…"
    return text


def attach_excerpts(docs: List[SourceDoc], top_n: int, excerpt_chars: int) -> List[SourceDoc]:
    for i, d in enumerate(docs):
        if i >= top_n:
            break
        try:
            txt = fetch_page_text(d.url)
            d.excerpt = txt[:excerpt_chars].strip()
        except Exception:
            d.excerpt = ""
    return docs


# =========================
# Prompt helpers
# =========================
def format_sources_block(label: str, docs: List[SourceDoc], per_doc_limit: int = 1400) -> str:
    lines = [f"## {label}"]
    for i, d in enumerate(docs, 1):
        parts = [f"[{i}] {d.title}", f"URL: {d.url}"]
        if d.snippet:
            parts.append(f"Snippet: {d.snippet}")
        if d.excerpt:
            parts.append(f"Auszug: {d.excerpt}")
        chunk = "\n".join(parts).strip()
        if len(chunk) > per_doc_limit:
            chunk = chunk[:per_doc_limit] + "…"
        lines.append(chunk + "\n")
    return "\n".join(lines)


def build_prompt(topic: str, pos_docs: List[SourceDoc], neg_docs: List[SourceDoc]) -> str:
    pos_block = format_sources_block("POSITIVE QUELLEN (Tipps/Chancen)", pos_docs)
    neg_block = format_sources_block("NEGATIVE QUELLEN (Reibung/Probleme)", neg_docs)

    return f"""
Du bist RealityBot. Du baust ein NAHBAR-DOSSIER aus Web-Quellen.
Ton: alltagstauglich, direkt, warm, realistisch (nicht wissenschaftlich, nicht literarisch).
Ziel: Es soll sich anfühlen, als hätte die Person „{topic}“ gedanklich schon einmal durchlebt.

ABSOLUT WICHTIG:
- KEINE internen Marker wie [1 pos 4 neg], KEIN Quellen-Scoring, KEINE Meta-Auswertungen im Text.

THEMA: {topic}

WEB-QUELLEN:
{pos_block}

{neg_block}

REGELN:
- Schreibe in 2. Person („du“).
- Paraphrasiere, keine langen Zitate.
- Wenn etwas nicht aus Quellen ableitbar ist: **Annahme** markieren.
- Die **Essenz** darf NICHT die Bulletpoints aus den anderen Abschnitten wiederholen.
  Sie soll das Erlebnis verdichten, ohne aufgeblasen zu wirken.

WICHTIG FÜR TECHNIK/PARSING:
- Gib die Abschnitte GENAU in diesem Marker-Format aus.
- KEINE zusätzlichen Marker, keine Änderungen an den Marker-Namen.

AUSGABE-FORMAT (genau so):

[[RB:BRIEFING]]
10–16 Bulletpoints mit hoher Erfahrungsdichte. (Nur Bulletpoints)
[[/RB:BRIEFING]]

[[RB:CHANCES_RISKS]]
- 5 Chancen (Alltagssprache + kurzer Kontext)
- 5 Risiken (Alltagssprache + kurzer Kontext)
[[/RB:CHANCES_RISKS]]

[[RB:MINEFIELD]]
6–10 Bulletpoints: „das kann dir passieren“. Klar, konkret.
[[/RB:MINEFIELD]]

[[RB:ESSENZ]]
KEINE Bulletpoints. Fließtext. Keine Wiederholung der Listen oben.
Stil: nahbar, direkt, realistisch. Keine Poesie, keine Roman-Sprache.
Kurze Absätze (meist 1–3 Sätze). Keine langen Satzketten.
Länge: ca. 500–900 Wörter (lieber dichter als länger).
Inhalt MUSS rein:
- 3–5 „Das passiert fast immer“-Momente (konkret)
- 2–3 „Wenn du X merkst, mach Y“-Sätze (Abkürzungen)
- 2–3 typische Gedanken („du denkst kurz…“) – kurz, nicht dramatisch
Wenn etwas nicht aus Quellen ableitbar ist: **Annahme** markieren.

QUALITÄTSBREMSE:
Wenn du zu blumig/„buchig“ wirst: kürze, werde konkreter, nimm Metaphern raus.
[[/RB:ESSENZ]]

[[RB:CHECKLIST]]
Checkboxen (- [ ]) passend zum Thema.
12–20 Checkboxen, nutze nur passende Blöcke:
- 🛒 Einkauf / Tools (falls relevant)
- 📅 Erste 7 Tage (falls relevant)
- 🧠 Mentale Hygiene (falls relevant)
[[/RB:CHECKLIST]]

[[RB:SOURCES]]
Nummerierte Liste: Titel + URL. Nur Quellen aus den Blöcken oben.
[[/RB:SOURCES]]
""".strip()


# =========================
# Gemini Call
# =========================
def call_gemini(prompt: str, api_key: str, model: str, timeout: int = 60) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    res = requests.post(url, json=payload, timeout=timeout)
    if res.status_code != 200:
        raise RuntimeError(f"Gemini API Fehler {res.status_code}: {res.text[:600]}")
    data = res.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        raise RuntimeError(f"Gemini Antwort unerwartet: {str(data)[:900]}")


# =========================
# API Key Validation (paste-safe)
# =========================
@st.cache_data(ttl=300)
def validate_gemini_key(api_key: str) -> Tuple[bool, str]:
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    res = requests.get(url, timeout=12)
    if res.status_code == 200:
        return True, "✅ API-Key gültig (Gemini)."
    return False, f"❌ Gemini: {res.status_code} – {res.text[:220]}"


def looks_like_complete_key(key: str) -> bool:
    k = key.strip()
    if len(k) < 20:
        return False
    return (k.startswith("AIza") and len(k) >= 30) or (len(k) >= 30)


def key_validation_engine(api_key: str, debounce_s: float = 0.6) -> None:
    if "key_status" not in st.session_state:
        st.session_state["key_status"] = None
    if "key_prev" not in st.session_state:
        st.session_state["key_prev"] = ""
    if "key_prev_len" not in st.session_state:
        st.session_state["key_prev_len"] = 0
    if "key_changed_ts" not in st.session_state:
        st.session_state["key_changed_ts"] = 0.0
    if "key_last_validated" not in st.session_state:
        st.session_state["key_last_validated"] = ""

    if not api_key:
        st.session_state["key_status"] = None
        st.session_state["key_prev"] = ""
        st.session_state["key_prev_len"] = 0
        st.session_state["key_last_validated"] = ""
        return

    api_key = api_key.strip()
    prev = st.session_state["key_prev"]
    prev_len = st.session_state["key_prev_len"]

    if api_key != prev:
        st.session_state["key_prev"] = api_key
        st.session_state["key_prev_len"] = len(api_key)
        st.session_state["key_changed_ts"] = time.time()
        st.session_state["key_status"] = ("pending", "⏳ Prüfe API-Key…")

        jumped = abs(len(api_key) - prev_len) >= 10 or (prev_len == 0 and len(api_key) >= 25)

        if jumped or looks_like_complete_key(api_key):
            if api_key != st.session_state["key_last_validated"]:
                try:
                    ok, msg = validate_gemini_key(api_key)
                    st.session_state["key_status"] = ("ok" if ok else "bad", msg)
                    st.session_state["key_last_validated"] = api_key
                except Exception as e:
                    st.session_state["key_status"] = ("bad", f"❌ Prüfung fehlgeschlagen: {e}")
        return

    ks = st.session_state.get("key_status")
    if ks and ks[0] == "pending":
        if (time.time() - st.session_state["key_changed_ts"]) < debounce_s:
            return

        if api_key == st.session_state["key_last_validated"]:
            return

        try:
            ok, msg = validate_gemini_key(api_key)
            st.session_state["key_status"] = ("ok" if ok else "bad", msg)
            st.session_state["key_last_validated"] = api_key
        except Exception as e:
            st.session_state["key_status"] = ("bad", f"❌ Prüfung fehlgeschlagen: {e}")


# =========================
# PDF export (pretty)
# =========================
def build_pdf_bytes_pretty(topic: str, blocks: Dict[str, str]) -> bytes:
    try:
        from io import BytesIO
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import cm
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem, PageBreak
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    except Exception as e:
        raise RuntimeError("PDF-Export benötigt 'reportlab'. Installiere mit: pip install reportlab") from e

    def esc(s: str) -> str:
        return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    h2 = ParagraphStyle("RBH2", parent=styles["Heading2"], spaceBefore=10, spaceAfter=6)
    body = ParagraphStyle("RBBody", parent=styles["BodyText"], leading=14, spaceAfter=6)
    small = ParagraphStyle("RBSmall", parent=styles["BodyText"], fontSize=9, leading=11, textColor="#666666")
    warn = ParagraphStyle("RBWarn", parent=body, backColor="#3b0f12", borderPadding=8, spaceBefore=6, spaceAfter=6)

    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        title=f"RealityBot – {topic}",
    )

    story = []
    story.append(Paragraph(esc("RealityBot – Deep-Dive Dossier"), title_style))
    story.append(Spacer(1, 6))
    story.append(Paragraph(esc(topic), styles["Heading1"]))
    story.append(Spacer(1, 10))
    story.append(Paragraph(esc(CREDIT_LINE), small))
    story.append(Paragraph(esc(datetime.now().strftime("%d.%m.%Y • %H:%M")), small))
    story.append(Spacer(1, 18))

    toc = [
        ("🧭 Wie es sich wirklich anfühlt", blocks.get("BRIEFING", "")),
        ("🔎 Chancen vs. Risiken", blocks.get("CHANCES_RISKS", "")),
        ("🚨 Das Minenfeld", blocks.get("MINEFIELD", "")),
        ("🧩 Maximale Essenz", blocks.get("ESSENZ", "")),
        ("✅ Praxis-Checkliste", blocks.get("CHECKLIST", "")),
        ("🌐 Quellen", blocks.get("SOURCES", "")),
    ]
    story.append(Paragraph(esc("Inhaltsübersicht"), styles["Heading3"]))
    story.append(Spacer(1, 6))
    story.append(ListFlowable(
        [ListItem(Paragraph(esc(t[0]), body)) for t in toc if t[1].strip()],
        bulletType="bullet",
        leftIndent=16
    ))
    story.append(PageBreak())

    def render_block(title: str, content: str, is_warning: bool = False):
        story.append(Paragraph(esc(title), h2))
        lines = [ln.rstrip() for ln in (content or "").splitlines()]
        bullets = []
        paras = []

        def flush_para():
            nonlocal paras
            if paras:
                p = " ".join([p.strip() for p in paras]).strip()
                if p:
                    story.append(Paragraph(esc(p), warn if is_warning else body))
                paras = []

        def flush_bullets():
            nonlocal bullets
            if bullets:
                story.append(ListFlowable(
                    [ListItem(Paragraph(esc(b), body)) for b in bullets],
                    bulletType="bullet",
                    leftIndent=16
                ))
                story.append(Spacer(1, 6))
                bullets = []

        for ln in lines:
            s = ln.strip()
            if not s:
                flush_para()
                flush_bullets()
                continue
            if s.startswith("- [ ]"):
                flush_para()
                bullets.append("☐ " + s[5:].strip())
                continue
            if s.startswith("- "):
                flush_para()
                bullets.append(s[2:].strip())
                continue
            if re.match(r"^\d+[\)\.]\s+", s):
                flush_para()
                bullets.append(s)
                continue
            paras.append(s)

        flush_para()
        flush_bullets()
        story.append(Spacer(1, 6))

    render_block("🧭 Wie es sich wirklich anfühlt", blocks.get("BRIEFING", ""))
    render_block("🔎 Chancen vs. Risiken", blocks.get("CHANCES_RISKS", ""))
    render_block("🚨 Das Minenfeld", blocks.get("MINEFIELD", ""), is_warning=True)
    render_block("🧩 Maximale Essenz", blocks.get("ESSENZ", ""))
    render_block("✅ Praxis-Checkliste", blocks.get("CHECKLIST", ""))
    render_block("🌐 Quellen", blocks.get("SOURCES", ""))

    story.append(Spacer(1, 8))
    story.append(Paragraph("— Ende des Dossiers —", small))
    doc.build(story)
    return buf.getvalue()


# =========================
# Global CSS + Sidebar UI
# =========================
apply_global_css(sidebar_px_desktop=460)

with st.sidebar:
    st.title("🔑 Aktivierung")

    st.markdown(
        """
**API-Key (kurz & simpel):**  
Ein **Zugangscode**, damit RealityBot bei Gemini deine Analyse anfragen darf.  
Er hängt an **deinem** Google-Konto (Limits/Kosten laufen darüber).  
👉 **Nicht teilen** – du kannst ihn jederzeit löschen.
"""
    )

    st.link_button("🔗 Gemini-API-Key erstellen", "https://aistudio.google.com/app/apikey", use_container_width=True)

    api_key = st.text_input("Dein Gemini API-Key:", type="password")
    key_validation_engine(api_key)

    ks = st.session_state.get("key_status")
    if ks is not None and ks[0] == "pending":
        try:
            st.autorefresh(interval=500, limit=8, key="rb_key_pending_autorefresh")
        except Exception:
            pass

    if ks is None:
        st.info("Key einfügen (Copy/Paste) – er wird automatisch geprüft.")
    else:
        state, msg = ks
        if state == "pending":
            st.warning(msg)
        elif state == "ok":
            st.success(msg)
        else:
            st.error(msg)

    st.markdown("---")
    st.markdown(
        """
**🔐 Datenschutz & Praxis-Tipp**  
- Dein Key wird **nicht gespeichert** (nur in dieser laufenden Session genutzt).  
- Je konkreter dein Thema, desto besser der Deep-Dive:
  *„Erstes Mal Festivalcamping (alleine, 3 Tage, Zelt)“* > *„Festival“*.
"""
    )

    st.markdown("---")
    with st.expander("🧰 Pro-Tools (optional)", expanded=False):
        # Standard etwas "schneller", damit Mobile nicht ewig lädt
        preset = st.radio("Deep-Dive Profil", ["Standard", "Deep", "Ultra-Deep", "Manuell"], index=0)

        if preset == "Standard":
            per_side, retries, backoff = 5, 3, 1.4
            fetch_enabled, fetch_top_n, excerpt_chars = True, 2, 650
        elif preset == "Deep":
            per_side, retries, backoff = 8, 4, 1.6
            fetch_enabled, fetch_top_n, excerpt_chars = True, 5, 1100
        elif preset == "Ultra-Deep":
            per_side, retries, backoff = 10, 5, 1.8
            fetch_enabled, fetch_top_n, excerpt_chars = True, 7, 1500
        else:
            per_side = st.slider("Treffer je Seite", 3, 10, 5, 1)
            retries = st.slider("Stabilität (Retries)", 1, 5, 3, 1)
            backoff = st.slider("Backoff", 1.0, 3.0, 1.4, 0.1)
            fetch_enabled = st.checkbox("Mehr Kontext (Seiten-Auszüge)", value=True)
            fetch_top_n = st.slider("Top-Links fetchen", 0, 8, 2, 1)
            excerpt_chars = st.slider("Auszug-Länge", 300, 2000, 650, 100)

        gemini_model = st.selectbox(
            "Gemini Modell",
            ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-lite"],
            index=0
        )

        demo_mode = st.checkbox("Demo ohne KI", value=False)


# Defaults if expander never opened
if "gemini_model" not in locals():
    gemini_model = "gemini-2.5-flash"
if "demo_mode" not in locals():
    demo_mode = False
if "per_side" not in locals():
    per_side, retries, backoff = 5, 3, 1.4
    fetch_enabled, fetch_top_n, excerpt_chars = True, 2, 650


# =========================
# Unlock logic
# =========================
key_ok = (st.session_state.get("key_status") is not None and st.session_state["key_status"][0] == "ok")
unlocked = demo_mode or key_ok

# === Mobile: Sidebar automatisch "schließen" (via CSS ausblenden), sobald freigeschaltet ===
if "rb_hide_sidebar_mobile" not in st.session_state:
    st.session_state["rb_hide_sidebar_mobile"] = False

# Sobald unlocked zum ersten Mal True ist, Sidebar auf Mobile automatisch ausblenden
if unlocked and not st.session_state["rb_hide_sidebar_mobile"]:
    st.session_state["rb_hide_sidebar_mobile"] = True
    st.rerun()

# CSS hide (Mobile only)
if st.session_state.get("rb_hide_sidebar_mobile", False):
    st.markdown(
        """
<style>
@media (pointer: coarse), (hover: none) {
  section[data-testid="stSidebar"] { display: none !important; }
  div[data-testid="stAppViewContainer"] { margin-left: 0 !important; }
}
</style>
        """,
        unsafe_allow_html=True
    )

# Blur on desktop only while locked
apply_blur_main_desktop_only(locked=not unlocked)


# =========================
# Main UI
# =========================
st.title("🧠 RealityBot: Deep-Dive")

st.markdown(
    """
RealityBot ist kein „frag die KI“-Tool.  
Er sammelt im Hintergrund echte Web-Erfahrungen – **Tipps UND Schattenseiten** – und verdichtet das so,
dass du den Kontext fühlst, nicht nur eine oberflächliche Erklärung bekommst.

Du bekommst:
- **Erfahrungs-Briefing**
- **Chancen + Risiken**
- **Minenfeld**
- **Maximale Essenz**
- **Praxis-Checkliste**
"""
)

# If locked
if not unlocked:
    st.error("🔒 Erst aktivieren: Tippe oben links auf **☰**, öffne die Sidebar und füge deinen **Gemini API-Key** ein.")
    st.caption(CREDIT_LINE)
    st.stop()

# Mobile helper: reopen settings (unhide sidebar)
with st.expander("⚙️ Einstellungen (Key/Pro-Tools)", expanded=False):
    st.write("Wenn du den Key ändern oder die Pro-Tools nutzen willst:")
    if st.button("Sidebar wieder anzeigen", use_container_width=True):
        st.session_state["rb_hide_sidebar_mobile"] = False
        st.rerun()
    st.info("Auf dem Handy dann links oben **☰** tippen, um die Sidebar zu öffnen.")

# Default mobile layout ON, without forcing users to understand it
if "mobile_layout" not in st.session_state:
    st.session_state["mobile_layout"] = True

st.session_state["mobile_layout"] = st.checkbox(
    "📱 Mobile-Ansicht (Tabs statt langer Seite)",
    value=st.session_state.get("mobile_layout", True),
    help="Auf dem Handy sind Tabs oft angenehmer als eine sehr lange Scroll-Seite.",
)


# =========================
# Form: ENTER submits
# =========================
with st.form("topic_form", clear_on_submit=False):
    topic = st.text_input(
        "Was planst du zum ersten Mal?",
        placeholder="z.B. erstes Mal Festivalcamping (alleine, Zelt, 3 Tage) / erste eigene Wohnung / …",
    )
    submitted = st.form_submit_button("Umfassende Analyse starten", use_container_width=True)

colA, colB = st.columns([1, 1])
with colA:
    if st.button("🧹 Ergebnis zurücksetzen", use_container_width=True):
        for k in ["final_report", "raw_sources_block", "topic_value"]:
            st.session_state.pop(k, None)
with colB:
    st.caption("💡 Tipp: Thema tippen + **Enter** drücken.")


def run_analysis(topic_value: str) -> None:
    if not topic_value.strip():
        st.error("Bitte gib ein Thema ein.")
        return

    neg_query = f"{topic_value} Probleme Risiken Schattenseiten Kritik Warnung Erfahrungen"
    pos_query = f"{topic_value} Tipps Erfahrungen was man wissen sollte häufige Fehler Vorbereitung"

    with st.status("🛰️ RealityBot arbeitet im Hintergrund…", expanded=False) as status:
        status.update(label="🛰️ Sammle Erfahrungswissen…", state="running")

        try:
            neg_raw = safe_ddgs_search(neg_query, max_results=per_side, retries=retries, backoff=backoff)
            time.sleep(0.35)
            pos_raw = safe_ddgs_search(pos_query, max_results=per_side, retries=retries, backoff=backoff)
        except Exception as e:
            status.update(label="⚠️ Recherche konnte nicht abgeschlossen werden", state="error")
            st.error(f"Recherche-Fehler: {e}")
            return

        neg_docs = normalize_results(neg_raw)
        pos_docs = normalize_results(pos_raw)

        if not neg_docs and not pos_docs:
            status.update(label="⚠️ Keine brauchbaren Quellen gefunden", state="error")
            st.warning("Keine Treffer mit URLs gefunden. Tipp: Thema anders formulieren.")
            return

        if fetch_enabled and fetch_top_n > 0:
            status.update(label="📚 Verdichte Kontext…", state="running")
            attach_excerpts(neg_docs, top_n=min(fetch_top_n, len(neg_docs)), excerpt_chars=excerpt_chars)
            attach_excerpts(pos_docs, top_n=min(fetch_top_n, len(pos_docs)), excerpt_chars=excerpt_chars)

        status.update(label="🧠 Schreibe Dossier…", state="running")
        prompt = build_prompt(topic_value.strip(), pos_docs, neg_docs)

        try:
            report = call_gemini(prompt, api_key.strip(), gemini_model)
        except Exception as e:
            report = f"[[RB:BRIEFING]]\n⚠️ KI-Analyse fehlgeschlagen: {str(e)}\n[[/RB:BRIEFING]]"

        report = strip_internal_markers(report)

        raw_sources = (
            "### Quellenprüfung (Rohdaten – gesammelt)\n\n"
            "**Tipps/Chancen-Quellen:**\n" + "\n".join([f"- {d.title}\n  - {d.url}" for d in pos_docs]) +
            "\n\n**Reibung/Probleme-Quellen:**\n" + "\n".join([f"- {d.title}\n  - {d.url}" for d in neg_docs])
        )

        st.session_state.final_report = report
        st.session_state.raw_sources_block = raw_sources
        st.session_state.topic_value = topic_value.strip()

        status.update(label="✅ Dossier fertig.", state="complete")


if submitted:
    run_analysis(topic)


# =========================
# Output rendering
# =========================
if "final_report" in st.session_state:
    report_md = st.session_state.final_report
    topic_value = st.session_state.get("topic_value", "RealityBot Dossier")
    raw_sources_block = st.session_state.get("raw_sources_block", "")

    blocks = {
        "BRIEFING": extract_rb_block(report_md, "BRIEFING"),
        "CHANCES_RISKS": extract_rb_block(report_md, "CHANCES_RISKS"),
        "MINEFIELD": extract_rb_block(report_md, "MINEFIELD"),
        "ESSENZ": extract_rb_block(report_md, "ESSENZ"),
        "CHECKLIST": extract_rb_block(report_md, "CHECKLIST"),
        "SOURCES": extract_rb_block(report_md, "SOURCES"),
    }

    # Fallback: wenn Marker fehlen (z.B. KI weicht ab), zeige komplett
    marker_missing = all(not v.strip() for v in blocks.values())

    st.divider()

    if marker_missing:
        st.warning("Hinweis: Die KI hat das Ausgabe-Format leicht verändert. Ich zeige deshalb das komplette Dossier.")
        st.markdown(report_md)
    else:
        if st.session_state.get("mobile_layout", True):
            tabs = st.tabs(["🧭 Briefing", "🔎 5+5", "🚨 Minenfeld", "🧩 Essenz", "✅ Checkliste", "🌐 Quellen"])
            with tabs[0]:
                st.markdown(blocks["BRIEFING"] or "_(nicht gefunden)_")
            with tabs[1]:
                st.markdown(blocks["CHANCES_RISKS"] or "_(nicht gefunden)_")
            with tabs[2]:
                if blocks["MINEFIELD"]:
                    st.error(blocks["MINEFIELD"])
                else:
                    st.markdown("_(nicht gefunden)_")
            with tabs[3]:
                st.markdown(blocks["ESSENZ"] or "_(nicht gefunden)_")
            with tabs[4]:
                st.markdown(blocks["CHECKLIST"] or "_(nicht gefunden)_")
            with tabs[5]:
                st.markdown(blocks["SOURCES"] or "_(nicht gefunden)_")
                with st.expander("🔎 Quellenprüfung (Rohdaten) – gesammelt"):
                    st.markdown(raw_sources_block)
        else:
            st.markdown("## 🧭 Wie es sich wirklich anfühlt")
            st.markdown(blocks["BRIEFING"] or "_(nicht gefunden)_")

            st.markdown("## 🔎 Chancen vs. Risiken (5 + 5)")
            st.markdown(blocks["CHANCES_RISKS"] or "_(nicht gefunden)_")

            st.markdown("## 🚨 Das Minenfeld")
            if blocks["MINEFIELD"]:
                st.error(blocks["MINEFIELD"])
            else:
                st.markdown("_(nicht gefunden)_")

            st.markdown("## 🧩 Maximale Essenz")
            st.markdown(blocks["ESSENZ"] or "_(nicht gefunden)_")

            st.markdown("## ✅ Praxis-Checkliste")
            st.markdown(blocks["CHECKLIST"] or "_(nicht gefunden)_")

            st.divider()
            st.markdown("## 🌐 Quellen")
            st.markdown(blocks["SOURCES"] or "_(nicht gefunden)_")
            with st.expander("🔎 Quellenprüfung (Rohdaten) – gesammelt"):
                st.markdown(raw_sources_block)

    export_txt = (
        f"RealityBot – Deep-Dive Dossier\n\n"
        f"THEMA:\n{topic_value}\n\n"
        f"RAW_REPORT:\n{report_md}\n"
    )

    st.download_button(
        "📄 Dossier als .txt speichern",
        data=export_txt.encode("utf-8"),
        file_name=f"RealityBot_{re.sub(r'[^a-zA-Z0-9_-]+', '_', topic_value)[:40]}.txt",
        mime="text/plain",
        use_container_width=True,
    )

    try:
        pdf_bytes = build_pdf_bytes_pretty(topic_value, blocks if not marker_missing else {"BRIEFING": report_md})
        st.download_button(
            "🧾 Dossier als PDF speichern",
            data=pdf_bytes,
            file_name=f"RealityBot_{re.sub(r'[^a-zA-Z0-9_-]+', '_', topic_value)[:40]}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    except Exception as e:
        st.warning(str(e))

st.caption(CREDIT_LINE)
