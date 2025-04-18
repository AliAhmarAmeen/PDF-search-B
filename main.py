import os
import numpy as np
import requests
from typing import List
from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
import pdfplumber
from nltk.tokenize import sent_tokenize
import nltk
nltk.download('punkt')
from dotenv import load_dotenv
import tempfile

# -----------------

nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')

# -----------------

load_dotenv()

# Cross Origin Policy
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["https://askfrompdf.netlify.app/"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# -----------Model API's ------------------------------------------
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
API_HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}

# Helper functions for API calls
def hf_embeddings_api(texts: List[str]) -> List[List[float]]:
    response = requests.post(
        "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2",
        headers=API_HEADERS,
        json={"inputs": texts, "options": {"wait_for_model": True}}
    )
    if response.status_code != 200:
        raise HTTPException(500, "Embedding API error")
    return response.json()

def hf_qa_api(context: str, question: str) -> dict:
    response = requests.post(
        "https://api-inference.huggingface.co/models/bert-large-uncased-whole-word-masking-finetuned-squad",
        headers=API_HEADERS,
        json={
            "inputs": {
                "question": question,
                "context": context
            },
            "options": {"wait_for_model": True}
        }
    )
    if response.status_code != 200:
        raise HTTPException(500, "QA API error")
    return response.json()

# ----------API Ends------------------------------------------------

# Helper Functions
def find_relevant_chunks(context: str, question: str, top_k: int = 3) -> list:
    chunks = [p.strip() for p in context.split('\n\n') if len(p.strip()) > 50]
    if not chunks:
        return [context]
    if len(chunks) < top_k:
        return chunks
    
    # Get embeddings via API
    embeddings = hf_embeddings_api([question] + chunks)
    question_embedding = embeddings[0]
    chunk_embeddings = embeddings[1:]
    
    # Calculate cosine similarity
    scores = []
    for emb in chunk_embeddings:
        norm_q = np.linalg.norm(question_embedding)
        norm_c = np.linalg.norm(emb)
        score = np.dot(question_embedding, emb) / (norm_q * norm_c)
        scores.append(score)
    
    top_indices = np.argsort(scores)[-top_k:][::-1]
    return [chunks[i] for i in top_indices]

# Helper Function
def expand_answer(answer: str, context: str) -> str:
    """Add surrounding sentences to create detailed answers"""
    sentences = sent_tokenize(context)
    answer_sentences = []
    for i, sent in enumerate(sentences):
        if any(word in sent.lower() for word in answer.lower().split()):
            start = max(0, i-2)
            end = min(len(sentences), i+2)
            answer_sentences.extend(sentences[start:end])
    seen = set()
    return ' '.join([sent for sent in answer_sentences if not (sent in seen or seen.add(sent))][:4])

# ---------------------------------

@app.get("/")
def health_check():
    return {"status": "App is running"}



# Now time to Find answer from PDF
@app.post("/ask")
async def ask_question(payload: dict = Body(...)):
    try:
        relevant_chunks = find_relevant_chunks(payload['context'], payload['question'])
        combined_context = "\n".join(relevant_chunks)
        
        # Get answer via API
        api_response = hf_qa_api(combined_context, payload['question'])
        answer = api_response['answer']
        score = api_response['score']
        
        expanded_answer = expand_answer(answer, combined_context)
        return {
            "answer": expanded_answer,
            "confidence": round(score, 4)
        }
    except Exception as e:
        raise HTTPException(500, f"Error processing question: {str(e)}")

# Extracting text from Given PDF to TEMP pdf
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if file.content_type != 'application/pdf':
        raise HTTPException(400, "Invalid file type")
    
    with tempfile.NamedTemporaryFile(delete=False) as temp_pdf:
        content = await file.read()
        temp_pdf.write(content)
        temp_pdf_path = temp_pdf.name

    try:
        text = ""
        # Using pdfplumber instead of PyPDF2
        with pdfplumber.open(temp_pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:  # Handle empty pages
                    text += page_text + "\n\n"
        
        os.unlink(temp_pdf_path)
        return {"text": text.strip()}
    except Exception as e:
        os.unlink(temp_pdf_path)
        raise HTTPException(500, f"Error processing PDF: {str(e)}")
    finally:
        if os.path.exists(temp_pdf_path):
            os.unlink(temp_pdf_path)
    