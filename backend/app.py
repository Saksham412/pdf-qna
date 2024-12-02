from fastapi import FastAPI, UploadFile
from pdf_processor import extract_text_from_pdf, chunk_text
from vector_store import store_embeddings, search_vector_db, embed_text
from chatbot import generate_answer

app = FastAPI()

@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile):
    text = extract_text_from_pdf(file.file)
    chunks = chunk_text(text)
    embeddings = [embed_text(chunk) for chunk in chunks]
    store_embeddings(embeddings, {"file": file.filename})
    return {"message": "PDF uploaded and processed."}

@app.get("/ask/")
async def ask_question(question: str):
    results = search_vector_db(question)
    answer = generate_answer(question, results)
    return {"answer": answer}
