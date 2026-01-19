from qdrant_client import QdrantClient
from pypdf import PdfReader
from io import BytesIO
from llama_index.core.schema import Document
from generation import prompt_template_generation, create_query_engine
from doc_processing import Custom_transformation


def reset_qdrant_collection():
    client = QdrantClient(url="http://localhost:6333")
    client.delete_collection("new_rag_docs")


def extract_pdf_text(pdf_bytes: bytes, pdf_name: str):
    reader = PdfReader(BytesIO(pdf_bytes))
    pages = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            pages.append({
                "page": i + 1,
                "text": text,
                "source": pdf_name,
            })
    return pages


def index_documents(uploaded_pdfs: list[tuple[str, bytes]]):
    """
    uploaded_pdfs = [(filename, pdf_bytes)]
    """
    all_pages = []

    for filename, pdf_bytes in uploaded_pdfs:
        pages = extract_pdf_text(pdf_bytes, filename)
        all_pages.extend(pages)

    documents = [
        Document(
            text=page["text"],
            metadata={
                "filename": page["source"],
                "source": "web",
            }
        )
        for page in all_pages
    ]

    Custom_transformation(documents)
    return {"status": "indexed", "pages": len(all_pages)}


def ask_question(query: str):
    prompt_gen = prompt_template_generation()
    prompt = prompt_gen.prompt_generation(query=query)
    response = create_query_engine(prompt)
    return response

