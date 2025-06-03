from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import tempfile
import shutil
import os
from dotenv import load_dotenv

from manuscript_analysis_quadrant import run_analysis

# Carica variabili da .env se presente
load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze")
async def analyze_manuscript(
    file: UploadFile,
    accepted_genres: str = Form(...),
    trend_keywords: str = Form(...)
):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        temp_path = tmp.name

    try:
        genres = [g.strip().lower() for g in accepted_genres.split(",")]
        keywords = [k.strip().lower() for k in trend_keywords.split(",")]
        result = run_analysis(temp_path, genres, keywords)
        return JSONResponse(content=result)
    finally:
        os.remove(temp_path)

if __name__ == "__main__":
    # Assicurati che la variabile OPENROUTER_API_KEY sia settata
    if not os.getenv("OPENROUTER_API_KEY"):
        raise RuntimeError("La variabile d'ambiente OPENROUTER_API_KEY non è definita.")
    uvicorn.run(app, host="0.0.0.0", port=8005)
