
import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

load_dotenv()

load_dotenv()

# Read API key from Streamlit Cloud Secrets or local .env
if "GROQ_API_KEY" in st.secrets:
    groq_key = st.secrets["GROQ_API_KEY"]
else:
    groq_key = os.getenv("GROQ_API_KEY", "")

st.set_page_config(page_title="RAG Document QA", page_icon="📄", layout="wide")
st.title("📄 RAG Document Question Answering System")


uploaded_files = st.sidebar.file_uploader(
    "Upload PDF / TXT / DOCX",
    type=["pdf", "txt", "docx"],
    accept_multiple_files=True
)

chunk_size = st.sidebar.slider("Chunk Size", 200, 1000, 500)
chunk_overlap = st.sidebar.slider("Chunk Overlap", 0, 200, 50)
k = st.sidebar.slider("Top K", 1, 10, 4)

@st.cache_resource(show_spinner=False)
def build_vector_db(file_bytes, file_name, chunk_size, chunk_overlap):
    suffix = "." + file_name.split(".")[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        path = tmp.name

    if suffix == ".pdf":
        docs = PyPDFLoader(path).load()
    elif suffix == ".txt":
        docs = TextLoader(path, encoding="utf-8").load()
    else:
        docs = Docx2txtLoader(path).load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.from_documents(chunks, embeddings)

if uploaded_files:
    all_docs = []
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    for file in uploaded_files:
        suffix = "." + file.name.split(".")[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file.read())
            path = tmp.name

        if suffix == ".pdf":
            docs = PyPDFLoader(path).load()
        elif suffix == ".txt":
            docs = TextLoader(path, encoding="utf-8").load()
        else:
            docs = Docx2txtLoader(path).load()

        all_docs.extend(docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = splitter.split_documents(all_docs)
    db = FAISS.from_documents(chunks, embeddings)

    st.success(f"Indexed {len(chunks)} chunks.")

    if groq_key:
        llm = ChatGroq(
            groq_api_key=groq_key,
            model_name="llama-3.3-70b-versatile",
            temperature=0
        )

        qa = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=db.as_retriever(search_kwargs={"k": k}),
            return_source_documents=True
        )

        question = st.text_input("Ask a question about your documents")

        if st.button("Generate Answer") and question:
            with st.spinner("Thinking..."):
                result = qa.invoke({"query": question})

            st.subheader("Answer")
            st.write(result["result"])

            st.subheader("Retrieved Sources")
            for i, doc in enumerate(result["source_documents"], start=1):
                with st.expander(f"Chunk {i}"):
                    st.write(doc.page_content)
                    st.caption(doc.metadata)
    else:
        st.warning("Enter your Groq API key in the sidebar.")
else:
    st.info("Upload one or more documents to begin.")
