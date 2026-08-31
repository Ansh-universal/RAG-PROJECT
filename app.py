import os
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
from mistralai.client import Mistral
from dotenv import load_dotenv
load_dotenv()


st.set_page_config(
    page_title="RAG PDF Assistant",
    page_icon="📚"
)

st.title("📚 RAG PDF Assistant")
st.write("Upload a PDF and ask questions about its content.")


mistral_api_key = st.secrets["MISTRAL_API_KEY"]
client = Mistral(api_key=mistral_api_key)


@st.cache_resource
def load_embedding_model():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model

embedding_model = load_embedding_model()


@st.cache_resource
def create_chroma():
    chroma_client = chromadb.Client()
    collection = chroma_client.get_or_create_collection(name="pdf_rag")
    return collection

collection = create_chroma()


uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])


if uploaded_file is not None:
    st.success("PDF uploaded successfully!")
    reader = PdfReader(uploaded_file)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"


    st.write(f"Total characters extracted: {len(text):,}")


    chunk_size = 1000
    overlap = 200
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap

    st.write(f"Total chunks created: {len(chunks)}")

    embeddings = embedding_model.encode(chunks).tolist()

    try:
        collection.delete(where={"source": "uploaded_pdf"})

    except:
        pass

    ids = [f"chunk_{i}"for i in range(len(chunks))]
    collection.add(ids=ids,documents=chunks,embeddings=embeddings,metadatas=[{"source": "uploaded_pdf"}for _ in chunks])
    st.success("PDF processed and stored in ChromaDB!")

    question = st.text_input("Ask a question about the PDF:")

    if question:
        query_embedding = embedding_model.encode(question).tolist()

        results = collection.query(
            query_embeddings=[query_embedding],n_results=5)

        relevant_chunks = results["documents"][0]

        context = "\n\n".join(relevant_chunks)

        prompt = f"""
You are a helpful question-answering assistant.

Answer the user's question using ONLY the information
provided in the context below.

If the answer cannot be found in the context,
say that the information is not available in the PDF.

Context:

{context}

Question:

{question}
"""

        with st.spinner("Thinking..."):
            response = client.chat.complete(model="mistral-small-latest",messages=[{"role": "user","content": prompt}])

        answer = response.choices[0].message.content
        st.subheader("Answer")
        st.write(answer)

        with st.expander("View retrieved PDF chunks"):
            for i, chunk in enumerate(relevant_chunks):
                st.write(f"### Chunk {i + 1}")
                st.write(chunk)