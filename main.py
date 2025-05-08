from fastapi import FastAPI, UploadFile, File, HTTPException
from transformers import pipeline
import pdfplumber
from openpyxl import load_workbook
from io import BytesIO
import traceback

app = FastAPI()

# Creazione della pipeline di text-generation
generator = pipeline("text-generation", model="tiiuae/falcon-rw-1b")

@app.get("/")
def home():
    return {"messaggio": "Carica PDF ed Excel tramite endpoint /upload/"}

@app.post("/upload/")
async def analizza_documenti(pdf: UploadFile = File(...), excel: UploadFile = File(...)):
    try:
        # Estrazione testo dal PDF
        testo_pdf = ""
        with pdfplumber.open(pdf.file) as pdf_file:
            for pagina in pdf_file.pages:
                estratto = pagina.extract_text()
                if estratto:
                    testo_pdf += estratto
                else:
                    testo_pdf += "[Pagina vuota o non leggibile]\n"

        # Caricamento dell'Excel
        wb = load_workbook(BytesIO(await excel.read()), data_only=True)
        foglio = wb.active
        dati_excel = []
        for riga in foglio.iter_rows(values_only=True):
            dati_excel.append(riga)

        # Creazione del prompt per l'AI
        prompt_ai = f"""
        Dati estratti dal PDF:
        {testo_pdf}

        Dati estratti dall'Excel:
        {dati_excel}

        Identifica chiaramente eventuali differenze, incongruenze o errori.
        """

        # Generazione del testo
        output = generator(prompt_ai, max_new_tokens=200)
        risposta_ai = output[0]["generated_text"]

        return {"Analisi effettuata da AI": risposta_ai}

    except Exception as e:
        errore_dettagliato = traceback.format_exc()
        print(f"Errore interno: {errore_dettagliato}")
        raise HTTPException(status_code=500, detail=errore_dettagliato)
