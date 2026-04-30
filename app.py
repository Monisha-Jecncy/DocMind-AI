import os
import shutil

from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.ingestion import ingest_documents
from src.rag_chain import ask_bot

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")
DATA_DIR = os.path.join(BASE_DIR, "data")


os.environ["ANONYMIZED_TELEMETRY"] = "False"

app = FastAPI()

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload")
async def upload_pdf(files: list[UploadFile] = File(...)):
    os.makedirs(DATA_DIR, exist_ok=True)

    saved_files = []

    for file in files:
        if not file.filename.endswith(".pdf"):
            continue

        path = os.path.join(DATA_DIR, file.filename)

        with open(path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        saved_files.append(file.filename)

    # rebuild DB
    ingest_documents()

    return {"files": saved_files}


@app.post("/ask")
async def ask(query: str = Form(...), file_name: str = Form(None)):
    try:
        result = ask_bot(query, file_name)
        return JSONResponse(content=result)
    except Exception as e:
        print("ERROR:", e)
        return JSONResponse(content={"answer": "Server Error", "sources": []})
