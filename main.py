import os
import numpy as np
import requests
from typing import List
from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
import pdfplumber
from nltk.tokenize import sent_tokenize
import nltk
nltk.data.path = [os.path.join(os.getcwd(), 'nltk_data')] + nltk.data.path
from dotenv import load_dotenv
import tempfile
import logging
import sys

# Configure logging properly for Railway
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

print("Starting application initialization...")  # Add this for debug

# NLTK setup function
def setup_nltk():
    try:
        print("Setting up NLTK...")
        nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
        os.makedirs(nltk_data_path, exist_ok=True)
        
        if nltk_data_path not in nltk.data.path:  
             nltk.data.path.insert(0, nltk_data_path)
        
        resources = ['punkt', 'averaged_perceptron_tagger']
        for resource in resources:
            try:
                nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' else f'taggers/{resource}')
                print(f"NLTK {resource} data already available")
            except LookupError:
                try:
                    nltk.download(resource, download_dir=nltk_data_path, quiet=True)
                    print(f"NLTK {resource} data downloaded successfully")
                except Exception as e:
                    print(f"Could not download NLTK {resource}: {e}")
        print("NLTK setup complete")
    except Exception as e:
        print(f"Error in NLTK setup: {e}")
        raise e

# Run NLTK setup
setup_nltk()

print("Loading environment variables...")
load_dotenv()

print("Creating FastAPI application...")
app = FastAPI()

print("Setting up CORS middleware...")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    # allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Check if HF token is available
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
if not HF_API_TOKEN:
    print("WARNING: HF_API_TOKEN not found in environment variables!")
else:
    print("HF_API_TOKEN found successfully")

API_HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}

# Helper functions for API calls
def hf_embeddings_api(texts: List[str]) -> List[List[float]]:
    try:
        response = requests.post(
            "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2",
            headers=API_HEADERS,
            json={"inputs": texts, "options": {"wait_for_model": True}}
        )
        if response.status_code != 200:
            raise HTTPException(500, f"Embedding API error: {response.text}")
        return response.json()
    except Exception as e:
        logger.error(f"Embedding API error: {str(e)}")
        raise HTTPException(500, f"Embedding API error: {str(e)}")

def hf_qa_api(context: str, question: str) -> dict:
    try:
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
            raise HTTPException(500, f"QA API error: {response.text}")
        return response.json()
    except Exception as e:
        logger.error(f"QA API error: {str(e)}")
        raise HTTPException(500, f"QA API error: {str(e)}")

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
        logger.error(f"Error processing question: {str(e)}")
        raise HTTPException(500, f"Error processing question: {str(e)}")


# Extracting text from Given PDF to TEMP pdf
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        if file.content_type != 'application/pdf':
            raise HTTPException(400, "Invalid file type")
        
        with tempfile.NamedTemporaryFile(delete=False) as temp_pdf:
            content = await file.read()
            temp_pdf.write(content)
            temp_pdf_path = temp_pdf.name

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
        if 'temp_pdf_path' in locals() and os.path.exists(temp_pdf_path):
            os.unlink(temp_pdf_path)
        logger.error(f"Error processing PDF: {str(e)}")
        raise HTTPException(500, f"Error processing PDF: {str(e)}")
    
# ----------------------------------------------------------------

@app.get("/")
def health_check():
    return {"status": "App is running", "message": "Health check successful"}

# ----------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
