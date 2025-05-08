 
from fastapi import FastAPI, UploadFile, File
from huggingface_hub import InferenceClient
import pdfplumber
from openpyxl import load_workbook

app = FastAPI()

# Sostituisci con il tuo HuggingFace Token personale
HUGGINGFACE_TOKEN = "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"
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

client = InferenceClient(model=model_name, token=HUGGINGFACE_TOKEN)

@app.get("/")
def home():
    return {"messaggio": "Carica PDF ed Excel tramite endpoint /upload/"}

@app.post("/upload/")
async def analizza_documenti(pdf: UploadFile = File(...), excel: UploadFile = File(...)):
    testo_pdf = ""
    with pdfplumber.open(pdf.file) as pdf_file:
        for pagina in pdf_file.pages:
            testo_pdf += pagina.extract_text()

    wb = load_workbook(excel.file, data_only=True)
    foglio = wb.active
    dati_excel = []
    for riga in foglio.iter_rows(values_only=True):
        dati_excel.append(riga)

    prompt_ai = f"""
    Dati estratti dal PDF:
    {testo_pdf}

    Dati estratti dall'Excel:
    {dati_excel}

    Identifica chiaramente eventuali differenze, incongruenze o errori.
    """

    risposta_ai = client.text_generation(prompt_ai, max_new_tokens=500)

    return {"Analisi effettuata da AI": risposta_ai}
