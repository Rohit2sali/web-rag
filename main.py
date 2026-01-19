from fastapi import FastAPI, UploadFile, File
from rag_main import reset_qdrant_collection, index_documents, ask_question

app = FastAPI(title="RAG Backend API")


@app.on_event("startup")
def startup():
    reset_qdrant_collection()


@app.post("/upload")
async def upload_pdfs(files: list[UploadFile] = File(...)):
    uploaded = []

    for file in files:
        content = await file.read()
        uploaded.append((file.filename, content))

    result = index_documents(uploaded)
    return result


@app.post("/ask")
def ask(query: str):
    answer = ask_question(query)
    return {"answer": answer}

@app.get("/health")
def health():
    return {"status": "ok"}
