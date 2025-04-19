import os
import numpy as np
import requests
from typing import List
from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
import pdfplumber
from nltk.tokenize import sent_tokenize
import nltk
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
            nltk.data.path.append(nltk_data_path)
        
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
    allow_origins=["https://askfrompdf.netlify.app"],
    allow_credentials=True,
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

# Rest of your code remains the same...

@app.get("/")
def health_check():
    return {"status": "App is running", "message": "Health check successful"}

# Add a simple test endpoint
@app.get("/test")
def test_endpoint():
    return {"message": "Test endpoint working"}

# Your other endpoints remain the same...

if __name__ == "__main__":
    print("Starting server from main...")
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="info")