"""
DataIntern — RAG Chatbot for CRM & Business Data (Gemini version, Drive-linked)

Before running, set these environment variables in your terminal:
    set GEMINI_API_KEY=your_gemini_key
    set DRIVE_API_KEY=your_google_cloud_drive_api_key
    (optional) set DRIVE_FOLDER_ID=your_folder_id   -- defaults to Arun's shared folder

Then run: streamlit run app.py

Click "Sync from Drive" in the sidebar to pull in every file currently in the
shared folder. Click it again any time you add new files to Drive.
"""

import io
import os
import json
import re
import contextlib

import requests
import streamlit as st
import pandas as pd
import numpy as np
import faiss
import plotly.express as px
import pdfplumber
import docx
from sentence_transformers import SentenceTransformer
from google import genai
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
DRIVE_API_KEY = st.secrets["DRIVE_API_KEY"]
# ----------------------------------------------------------------------------
# Basic setup
# ----------------------------------------------------------------------------
MODEL_NAME = "gemini-2.5-flash"

st.set_page_config(page_title="DataIntern", layout="wide")

# session_state = Streamlit's way of remembering things between clicks
if "chunks" not in st.session_state:
    st.session_state.chunks = []          # text pieces for RAG search
if "index" not in st.session_state:
    st.session_state.index = None          # the FAISS search index
if "embedder" not in st.session_state:
    st.session_state.embedder = SentenceTransformer("all-MiniLM-L6-v2")
if "dataframes" not in st.session_state:
    st.session_state.dataframes = {}       # name -> pandas table, for math questions
if "messages" not in st.session_state:
    st.session_state.messages = []         # chat history shown on screen


# ----------------------------------------------------------------------------
# SIDEBAR: API key + file upload
# ----------------------------------------------------------------------------
api_key = os.environ.get("GEMINI_API_KEY")
DRIVE_API_KEY = os.environ.get("DRIVE_API_KEY")
DRIVE_FOLDER_ID = os.environ.get("DRIVE_FOLDER_ID", "1T4mGVK9XqEMQ9tS8XZrVnJp2-2sQRpo7")

with st.sidebar:
    st.header("Data source: Google Drive")
    st.caption(f"Folder ID: {DRIVE_FOLDER_ID}")
    sync_clicked = st.button("🔄 Sync from Drive")

    st.divider()
    st.header("Or upload manually")
    uploaded_files = st.file_uploader(
        "CSV, XLSX, PDF, DOCX, JSON, TSV",
        accept_multiple_files=True,
        type=["csv", "tsv", "xlsx", "pdf", "docx", "json"],
    )
    process_clicked = st.button("Process uploaded files")

if not api_key:
    st.error(
        "GEMINI_API_KEY environment variable not found.\n\n"
        "Close this app, and in your terminal run:\n\n"
        "set GEMINI_API_KEY=your_key_here\n\n"
        "then run: streamlit run app.py  (again, in the same terminal window)."
    )
    st.stop()

client = genai.Client(api_key=api_key)


# ----------------------------------------------------------------------------
# Google Drive: list files in the shared folder + download each one
# ----------------------------------------------------------------------------
GOOGLE_NATIVE_EXPORTS = {
    "application/vnd.google-apps.document": (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "docx"),
    "application/vnd.google-apps.spreadsheet": (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "xlsx"),
    "application/vnd.google-apps.presentation": ("application/pdf", "pdf"),
}
SUPPORTED_EXT = {"csv", "tsv", "xlsx", "pdf", "docx", "json"}


def list_drive_files(folder_id, api_key):
    url = "https://www.googleapis.com/drive/v3/files"
    params = {
        "q": f"'{folder_id}' in parents and trashed = false",
        "key": api_key,
        "fields": "files(id, name, mimeType)",
        "pageSize": 100,
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json().get("files", [])


def download_drive_file(file_id, mime_type, api_key):
    """Returns (bytes, filename_extension_to_use) for a single Drive file."""
    if mime_type in GOOGLE_NATIVE_EXPORTS:
        export_mime, ext = GOOGLE_NATIVE_EXPORTS[mime_type]
        url = f"https://www.googleapis.com/drive/v3/files/{file_id}/export"
        params = {"mimeType": export_mime, "key": api_key}
    else:
        url = f"https://www.googleapis.com/drive/v3/files/{file_id}"
        params = {"alt": "media", "key": api_key}
        ext = None
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    return resp.content, ext


def sync_from_drive(folder_id, api_key):
    files_meta = list_drive_files(folder_id, api_key)
    new_chunks = []
    processed, skipped = [], []
    for meta in files_meta:
        name = meta["name"]
        ext = name.split(".")[-1].lower() if "." in name else ""
        try:
            content, forced_ext = download_drive_file(meta["id"], meta["mimeType"], api_key)
            if forced_ext:  # Google-native file, use exported extension + keep base name
                base_name = name if "." in name else name
                name = f"{base_name}.{forced_ext}"
                ext = forced_ext
            if ext not in SUPPORTED_EXT:
                skipped.append(meta["name"])
                continue
            file_like = io.BytesIO(content)
            new_chunks.extend(process_drive_file(file_like, name))
            processed.append(name)
        except Exception as e:
            skipped.append(f"{name} ({e})")
    return new_chunks, processed, skipped


# ----------------------------------------------------------------------------
# File parsing -> chunks (for text search) + dataframes (for tabular questions)
# ----------------------------------------------------------------------------
def parse_csv_xlsx(file, name):
    ext = name.split(".")[-1].lower()
    sheets = {}
    if ext in ("csv", "tsv"):
        sep = "\t" if ext == "tsv" else ","
        sheets[name] = pd.read_csv(file, sep=sep)
    else:
        xl = pd.read_excel(file, sheet_name=None)
        for sheet_name, df in xl.items():
            sheets[f"{name} · {sheet_name}"] = df

    new_chunks = []
    for key, df in sheets.items():
        st.session_state.dataframes[key] = df
        schema_text = (
            f"Table '{key}' has columns: {', '.join(df.columns.astype(str))}. "
            f"It has {len(df)} rows. Sample rows:\n{df.head(5).to_string(index=False)}"
        )
        new_chunks.append({"text": schema_text, "source": f"{key} · schema"})
        for i, row in df.head(200).iterrows():
            row_text = ", ".join(f"{c}={row[c]}" for c in df.columns)
            new_chunks.append({"text": f"Row {i} in {key}: {row_text}", "source": f"{key} · row {i}"})
    return new_chunks


def parse_pdf(file, name):
    new_chunks = []
    with pdfplumber.open(file) as pdf:
        for pnum, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                new_chunks.append({"text": text, "source": f"{name} · page {pnum}"})
    return new_chunks


def parse_docx(file, name):
    d = docx.Document(file)
    new_chunks, buf, para_start = [], [], 1
    for i, para in enumerate(d.paragraphs, start=1):
        if para.text.strip():
            buf.append(para.text)
        if len(buf) >= 10:
            new_chunks.append({"text": "\n".join(buf), "source": f"{name} · paragraphs {para_start}-{i}"})
            buf, para_start = [], i + 1
    if buf:
        new_chunks.append({"text": "\n".join(buf), "source": f"{name} · paragraphs {para_start}-end"})
    return new_chunks


def parse_json(file, name):
    data = json.load(file)
    new_chunks = []
    if isinstance(data, list) and data and isinstance(data[0], dict):
        df = pd.json_normalize(data)
        st.session_state.dataframes[name] = df
        new_chunks.append({
            "text": f"JSON '{name}' has fields: {', '.join(df.columns.astype(str))}, {len(df)} records.",
            "source": f"{name} · schema",
        })
        for i, row in df.head(200).iterrows():
            row_text = ", ".join(f"{c}={row[c]}" for c in df.columns)
            new_chunks.append({"text": f"Record {i} in {name}: {row_text}", "source": f"{name} · record {i}"})
    else:
        new_chunks.append({"text": json.dumps(data, indent=2)[:4000], "source": name})
    return new_chunks


def process_drive_file(file_obj, name):
    ext = name.split(".")[-1].lower()
    try:
        if ext in ("csv", "tsv", "xlsx"):
            return parse_csv_xlsx(file_obj, name)
        elif ext == "pdf":
            return parse_pdf(file_obj, name)
        elif ext == "docx":
            return parse_docx(file_obj, name)
        elif ext == "json":
            return parse_json(file_obj, name)
        else:
            st.warning(f"Unsupported file type: {name}")
            return []
    except Exception as e:
        st.error(f"Error parsing {name}: {e}")
        return []


def process_uploaded_file(uploaded_file):
    return process_drive_file(uploaded_file, uploaded_file.name)


def rebuild_index():
    texts = [c["text"] for c in st.session_state.chunks]
    if not texts:
        return
    embeddings = st.session_state.embedder.encode(texts, show_progress_bar=False)
    embeddings = np.array(embeddings).astype("float32")
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    st.session_state.index = index


if sync_clicked:
    if not DRIVE_API_KEY:
        st.error(
            "DRIVE_API_KEY environment variable not found.\n\n"
            "In your terminal (same window as GEMINI_API_KEY), run:\n\n"
            "set DRIVE_API_KEY=your_drive_api_key_here\n\n"
            "then run: streamlit run app.py again."
        )
    else:
        with st.spinner("Fetching files from Drive and indexing..."):
            new_chunks, processed, skipped = sync_from_drive(DRIVE_FOLDER_ID, DRIVE_API_KEY)
            st.session_state.chunks.extend(new_chunks)
            rebuild_index()
        st.success(f"Synced {len(processed)} file(s) from Drive: {', '.join(processed) or 'none'}")
        if skipped:
            st.warning(f"Skipped: {', '.join(skipped)}")

if process_clicked and uploaded_files:
    with st.spinner("Reading and indexing your files..."):
        for f in uploaded_files:
            st.session_state.chunks.extend(process_uploaded_file(f))
        rebuild_index()
    st.success(f"Indexed {len(st.session_state.chunks)} chunks from {len(uploaded_files)} file(s).")

with st.sidebar:
    if st.session_state.dataframes:
        st.subheader("Loaded tables")
        for name in st.session_state.dataframes:
            st.caption(name)


# ----------------------------------------------------------------------------
# Search (retrieval) over the indexed chunks
# ----------------------------------------------------------------------------
def retrieve(query, k=5):
    if st.session_state.index is None:
        return []
    q_emb = st.session_state.embedder.encode([query]).astype("float32")
    faiss.normalize_L2(q_emb)
    scores, ids = st.session_state.index.search(q_emb, k)
    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx == -1:
            continue
        results.append({**st.session_state.chunks[idx], "score": float(score)})
    return results


# ----------------------------------------------------------------------------
# Deciding what kind of question this is
# ----------------------------------------------------------------------------
CHART_KEYWORDS = ["chart", "graph", "plot", "visuali", "show me", "trend", "breakdown", "bar", "pie", "compare"]
AGG_KEYWORDS = ["total", "sum", "average", "avg", "count", "how many", "revenue", "by rep", "by region",
                "group by", "closed-won", "closed won", "pipeline"]


def wants_chart(q):
    return any(k in q.lower() for k in CHART_KEYWORDS)


def wants_tabular(q):
    return any(k in q.lower() for k in AGG_KEYWORDS) and st.session_state.dataframes


# ----------------------------------------------------------------------------
# Tabular question -> Gemini writes a pandas snippet -> we run it -> real number
# ----------------------------------------------------------------------------
def run_tabular_query(question):
    schemas = [f"- dfs['{name}'] columns: {list(df.columns)} ({len(df)} rows)"
               for name, df in st.session_state.dataframes.items()]
    schema_block = "\n".join(schemas)

    prompt = f"""You are given pandas DataFrames accessible via a dict called dfs.
{schema_block}

Write a short Python snippet (code only, no explanation) that computes the answer to:
"{question}"

Rules:
- Use only dfs[...] and pandas/numpy operations.
- Assign the final answer to a variable named result.
- If the result should be charted, keep result as a DataFrame.
- No imports, no file/network access, no exec/eval.
Return ONLY the code, no markdown fences."""

    # --- THE NEW TRY/EXCEPT BLOCK GOES HERE ---
    try:
        response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
    except Exception as e:
        # If Google rejects the API call (e.g., bad key, region block), return the exact error.
        return None, "", f"Gemini API connection failed: {str(e)}"
    # ------------------------------------------

    code = response.text.strip()
    code = re.sub(r"^```python|```$", "", code, flags=re.MULTILINE).strip()

    safe_globals = {"pd": pd, "np": np, "dfs": st.session_state.dataframes}
    local_vars = {}
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            exec(code, safe_globals, local_vars)
        return local_vars.get("result"), code, None
    except Exception as e:
        return None, code, str(e)


# ----------------------------------------------------------------------------
# Plain-text RAG answer with citations
# ----------------------------------------------------------------------------
def answer_with_rag(question, history):
    chunks = retrieve(question, k=6)
    if not chunks:
        return "I don't see that in your files.", []

    context = "\n\n".join(f"[{c['source']}]\n{c['text']}" for c in chunks)
    history_block = "\n".join(f"{m['role']}: {m['content']}" for m in history[-6:])

    prompt = f"""You are a business data assistant. Answer ONLY using the context below.
If the answer isn't in the context, say exactly: "I don't see that in your files."
Cite the source in square brackets like [filename · sheet/page/row] after each claim.

Conversation so far:
{history_block}

Context:
{context}

Question: {question}
Answer:"""

    placeholder = st.empty()
    full_text = ""
    for chunk in client.models.generate_content_stream(model=MODEL_NAME, contents=prompt):
        if chunk.text:
            full_text += chunk.text
            placeholder.markdown(full_text)
    return full_text, chunks


# ----------------------------------------------------------------------------
# Charts
# ----------------------------------------------------------------------------
def make_chart(df, question):
    prompt = f"""Given a table with columns {list(df.columns)} and this request: "{question}",
respond ONLY with JSON like: {{"chart_type": "bar", "x": "<column name>", "y": "<column name or null>"}}
chart_type must be one of: bar, line, pie, scatter."""
    response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
    spec_text = re.sub(r"^```json|```$", "", response.text.strip(), flags=re.MULTILINE).strip()
    try:
        spec = json.loads(spec_text)
    except Exception:
        spec = {"chart_type": "bar", "x": df.columns[0], "y": df.columns[1] if len(df.columns) > 1 else None}

    ctype, x, y = spec.get("chart_type", "bar"), spec.get("x"), spec.get("y")
    if ctype == "pie":
        fig = px.pie(df, names=x, values=y)
    elif ctype == "line":
        fig = px.line(df, x=x, y=y)
    elif ctype == "scatter":
        fig = px.scatter(df, x=x, y=y)
    else:
        fig = px.bar(df, x=x, y=y)
    return fig


# ----------------------------------------------------------------------------
# Chat UI
# ----------------------------------------------------------------------------
st.title("📊 DataIntern — chat with your business files")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

question = st.chat_input("Ask a question about your data...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if wants_tabular(question) or wants_chart(question):
            result, code, err = run_tabular_query(question)
            with st.expander("Show pandas query used"):
                st.code(code, language="python")

            if err:
                st.warning(f"Query failed ({err}); falling back to text search.")
                answer, sources = answer_with_rag(question, st.session_state.messages)
            else:
                if isinstance(result, pd.DataFrame) and wants_chart(question):
                    fig = make_chart(result, question)
                    st.plotly_chart(fig, use_container_width=True)
                    answer = f"Chart generated from computed data (columns: {list(result.columns)})."
                    st.markdown(answer)
                else:
                    answer = f"**{result}**"
                    st.markdown(answer)
                sources = []
        else:
            answer, sources = answer_with_rag(question, st.session_state.messages)

        if sources:
            with st.expander("Sources"):
                for s in sources:
                    st.caption(f"{s['source']} (score {s['score']:.2f})")

    st.session_state.messages.append({"role": "assistant", "content": answer})
