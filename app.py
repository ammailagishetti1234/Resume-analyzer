import re
import tempfile
from typing import List, Tuple, Dict

import streamlit as st
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------
# CONFIG
# -----------------------------
DEFAULT_THRESHOLD = 40

DEFAULT_JOB_DESCRIPTION = """
We are looking for a .NET Full Stack Developer with experience in C#, ASP.NET Core, Web API,
Azure (Functions, App Service), SQL, Git, and basic front-end (HTML, CSS, JavaScript).
Experience with CI/CD, REST APIs, Postman, and Microservices is a plus.
"""


# -----------------------------
# TEXT UTILS
# -----------------------------
STOPWORD_LIKE = {
    "a", "an", "the", "and", "or", "to", "of", "for", "with", "in", "on", "at", "as", "is", "are",
    "be", "will", "we", "you", "your", "our", "they", "their", "from", "by", "this", "that",
    "experience", "years", "year", "plus", "strong", "good", "excellent", "looking", "role",
    "developer", "engineer", "required", "requirements", "responsibilities"
}

def clean_text(text: str) -> str:
    """Normalize text for matching."""
    text = text.lower()
    # keep c#, .net, +, #; remove other punctuation
    text = re.sub(r"[^a-z0-9\.\+#\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def contains_phrase(haystack_clean: str, phrase_clean: str) -> bool:
    # word-boundary-ish matching for short phrases
    return f" {phrase_clean} " in f" {haystack_clean} "


# -----------------------------
# FILE TEXT EXTRACTION
# -----------------------------
def extract_text_from_upload(uploaded_file) -> str:
    """
    Extract text from uploaded resume file: PDF/DOCX/TXT
    Note: scanned image PDFs will return empty unless OCR is used.
    """
    filename = uploaded_file.name.lower()
    file_bytes = uploaded_file.getvalue()

    if filename.endswith(".pdf"):
        from pdfminer.high_level import extract_text as pdf_extract_text
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        return pdf_extract_text(tmp_path) or ""

    if filename.endswith(".docx"):
        import docx2txt
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        return docx2txt.process(tmp_path) or ""

    if filename.endswith(".txt"):
        return file_bytes.decode("utf-8", errors="ignore")

    raise ValueError("Unsupported file type. Please upload PDF, DOCX, or TXT.")


# -----------------------------
# KEYWORD / PHRASE EXTRACTION FROM JD (No predefined SKILLS)
# -----------------------------
def extract_keywords_from_jd(jd_text: str, top_k: int = 35) -> List[str]:
    """
    Extract important 1-3 gram keywords/phrases from JD using TF-IDF on the JD text itself.
    This mimics "online analyzer" keyword lists (ATS-like).
    """
    jd_clean = clean_text(jd_text)
    if not jd_clean:
        return []

    # TF-IDF on single doc still works; scores become essentially term frequency weighted.
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 3),
        min_df=1
    )
    X = vectorizer.fit_transform([jd_clean])
    features = vectorizer.get_feature_names_out()
    scores = X.toarray().flatten()

    # rank by tf-idf score
    ranked = sorted(zip(features, scores), key=lambda x: x[1], reverse=True)

    keywords = []
    for term, sc in ranked:
        term = term.strip()
        if sc <= 0:
            continue
        # filter junk
        if len(term) < 2:
            continue
        if term in STOPWORD_LIKE:
            continue
        if all(ch.isdigit() for ch in term):
            continue
        # remove very generic single words
        if " " not in term and term in STOPWORD_LIKE:
            continue

        # keep unique and avoid too-similar duplicates
        if term not in keywords:
            keywords.append(term)

        if len(keywords) >= top_k:
            break

    # A small cleanup: prefer keeping tech tokens like c#, asp.net, .net
    # Also remove phrases that are substrings of a longer already-selected phrase (optional)
    final = []
    for kw in keywords:
        if any((kw != other and kw in other) for other in keywords):
            # if it's a substring, keep it only if it's a known tech token
            if kw in {"c#", ".net", "asp.net", "sql", "azure", "javascript"}:
                final.append(kw)
        else:
            final.append(kw)

    # ensure stable order & limit
    return final[:top_k]


def keyword_coverage(resume_text: str, keywords: List[str]) -> Tuple[float, List[str], List[str]]:
    """Percent of JD keywords present in resume."""
    resume_clean = clean_text(resume_text)
    if not keywords:
        return 0.0, [], []

    matched, missing = [], []
    for kw in keywords:
        kw_clean = clean_text(kw)
        if contains_phrase(resume_clean, kw_clean):
            matched.append(kw)
        else:
            missing.append(kw)

    score = (len(matched) / max(len(keywords), 1)) * 100.0
    return round(score, 2), matched, missing


# -----------------------------
# SEMANTIC SIMILARITY (BERT if available, else TF-IDF)
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_embedding_model():
    """
    Try to load sentence-transformers model.
    If not installed, return None and we will use TF-IDF fallback.
    """
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        return None


def semantic_similarity(resume_text: str, jd_text: str, use_bert: bool = True) -> float:
    """
    Returns similarity in percent (0-100).
    - If BERT is available and enabled: uses embeddings
    - Else: TF-IDF cosine similarity fallback
    """
    resume_clean = clean_text(resume_text)
    jd_clean = clean_text(jd_text)
    if not resume_clean or not jd_clean:
        return 0.0

    if use_bert:
        model = load_embedding_model()
        if model is not None:
            emb = model.encode([resume_text, jd_text], normalize_embeddings=True)
            sim = float((emb[0] * emb[1]).sum())  # cosine because normalized
            return round(sim * 100.0, 2)

    # fallback TF-IDF
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform([resume_clean, jd_clean])
    score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    return round(score * 100.0, 2)


# -----------------------------
# FINAL HYBRID SCORING (More like online analyzers)
# -----------------------------
def hybrid_score(
    semantic_pct: float,
    keyword_pct: float,
    phrase_pct: float,
    w_sem: float = 0.55,
    w_key: float = 0.35,
    w_phrase: float = 0.10
) -> float:
    """
    Weighted score (0-100). Weights should sum to 1.0.
    """
    total = (w_sem * semantic_pct) + (w_key * keyword_pct) + (w_phrase * phrase_pct)
    return round(total, 2)


def phrase_coverage(resume_text: str, keywords: List[str]) -> float:
    """
    Give extra credit for multi-word phrases (2-3 grams) matched.
    Many ATS tools value phrases more than single tokens.
    """
    resume_clean = clean_text(resume_text)
    phrases = [k for k in keywords if " " in k]  # multi-word only
    if not phrases:
        return 0.0

    matched = 0
    for p in phrases:
        p_clean = clean_text(p)
        if contains_phrase(resume_clean, p_clean):
            matched += 1

    return round((matched / max(len(phrases), 1)) * 100.0, 2)


# -----------------------------
# VISUALS
# -----------------------------
def plot_donut(score: float, threshold: float):
    fig, ax = plt.subplots(figsize=(5, 5))
    score = max(0, min(100, score))
    remain = 100 - score
    color = "#2ecc71" if score >= threshold else "#e74c3c"
    ax.pie([score, remain], colors=[color, "#ecf0f1"], startangle=90,
           wedgeprops={"width": 0.35, "edgecolor": "white"})
    ax.text(0, 0, f"{score:.1f}%", ha="center", va="center", fontsize=18, fontweight="bold")
    ax.set_title("Overall Match")
    return fig


def plot_breakdown(semantic_pct: float, keyword_pct: float, phrase_pct: float):
    fig, ax = plt.subplots(figsize=(7, 4))
    items = ["Semantic", "Keyword Coverage", "Phrase Coverage"]
    vals = [semantic_pct, keyword_pct, phrase_pct]
    ax.bar(items, vals, color=["#3498db", "#9b59b6", "#f1c40f"])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Score (%)")
    ax.set_title("Score Breakdown")
    for i, v in enumerate(vals):
        ax.text(i, v + 2, f"{v:.1f}%", ha="center", fontweight="bold")
    return fig


# -----------------------------
# STREAMLIT UI
# -----------------------------
def main():
    st.set_page_config(page_title="Resume Analyzer", page_icon="📄", layout="centered")
    st.title("📄 Resume Analyzer (Online-style: Hybrid ATS + Semantic Matching)")

    st.write(
        "This version behaves more like typical online analyzers:\n"
        "✅ Extracts important JD keywords automatically (no fixed SKILLS list)\n"
        "✅ Uses semantic similarity (BERT if available)\n"
        "✅ Shows matched & missing keywords + charts\n"
    )

    with st.expander("⚙️ Settings", expanded=False):
        threshold = st.slider("Reject Threshold (%)", 0, 100, DEFAULT_THRESHOLD, 1)

        use_bert = st.checkbox(
            "Use Semantic (BERT) similarity if available (recommended)",
            value=True,
            help="If sentence-transformers is installed, this gives much better matching than TF-IDF."
        )

        top_k = st.slider("How many JD keywords to extract", 10, 80, 35, 5)

        st.markdown("### Weighting (like ATS scoring)")
        w_sem = st.slider("Weight: Semantic Similarity", 0.0, 1.0, 0.55, 0.05)
        w_key = st.slider("Weight: Keyword Coverage", 0.0, 1.0, 0.35, 0.05)
        w_phrase = st.slider("Weight: Phrase Coverage", 0.0, 1.0, 0.10, 0.05)

        # normalize weights safely
        s = w_sem + w_key + w_phrase
        if s == 0:
            w_sem, w_key, w_phrase = 0.55, 0.35, 0.10
        else:
            w_sem, w_key, w_phrase = w_sem / s, w_key / s, w_phrase / s

        st.caption(f"Normalized weights → Semantic: {w_sem:.2f}, Keywords: {w_key:.2f}, Phrases: {w_phrase:.2f}")

    st.subheader("1) Upload Resume")
    uploaded_file = st.file_uploader("Upload Resume (PDF / DOCX / TXT)", type=["pdf", "docx", "txt"])

    st.subheader("2) Job Description")
    jd_text = st.text_area("Paste Job Description here", value=DEFAULT_JOB_DESCRIPTION, height=180)

    analyze_btn = st.button("🔍 Analyze")

    if analyze_btn:
        if not uploaded_file:
            st.error("Please upload a resume file first.")
            return

        try:
            resume_text = extract_text_from_upload(uploaded_file)
        except Exception as e:
            st.error(f"Could not read file: {e}")
            return

        if not resume_text.strip():
            st.warning(
                "Resume text extraction returned empty.\n\n"
                "If your PDF is scanned (image-based), text extraction won't work without OCR.\n"
                "Try uploading DOCX/TXT or ask me to add OCR support."
            )
            return

        # --- Extract JD keywords automatically ---
        jd_keywords = extract_keywords_from_jd(jd_text, top_k=top_k)

        # --- Keyword and phrase coverage ---
        keyword_pct, matched_kw, missing_kw = keyword_coverage(resume_text, jd_keywords)
        phrase_pct = phrase_coverage(resume_text, jd_keywords)

        # --- Semantic similarity (BERT if available) ---
        semantic_pct = semantic_similarity(resume_text, jd_text, use_bert=use_bert)

        # --- Final hybrid score ---
        overall = hybrid_score(semantic_pct, keyword_pct, phrase_pct, w_sem, w_key, w_phrase)
        decision = "✅ SELECTED (Shortlisted)" if overall >= threshold else "❌ REJECTED"

        st.subheader("✅ Results")
        st.metric("Overall Match %", f"{overall}%")
        if overall >= threshold:
            st.success(f"Decision: {decision}")
        else:
            st.error(f"Decision: {decision}")

        # Charts
        c1, c2 = st.columns(2)
        with c1:
            st.pyplot(plot_donut(overall, threshold))
        with c2:
            st.pyplot(plot_breakdown(semantic_pct, keyword_pct, phrase_pct))

        st.markdown("### 🧾 Detailed Scores")
        st.write(f"**Semantic Similarity:** {semantic_pct}%")
        st.write(f"**Keyword Coverage:** {keyword_pct}%")
        st.write(f"**Phrase Coverage:** {phrase_pct}%")

        # Matched / Missing keywords
        st.subheader("🧠 Keyword Match (Auto-extracted from JD)")

        col1, col2 = st.columns(2)
        with col1:
            st.write("✅ Matched Keywords")
            if matched_kw:
                st.write(", ".join(matched_kw))
            else:
                st.write("None")
        with col2:
            st.write("❌ Missing Keywords")
            if missing_kw:
                st.write(", ".join(missing_kw))
            else:
                st.write("None")

        # Debug extracted text
        with st.expander("📝 View Extracted Resume Text (Debug)", expanded=False):
            st.text_area("Extracted resume text (first 12k chars):", value=resume_text[:12000], height=250)

        st.caption(
            "Note: This is a relevance score (hybrid ATS-like), not a true ML 'accuracy'. "
            "You should calibrate the reject threshold using several sample resumes."
        )


if __name__ == "__main__":
    main()