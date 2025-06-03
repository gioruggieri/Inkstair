import fitz
import re
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph
from typing import TypedDict

from reportlab.lib.pagesizes import A4
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle, PageBreak
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.enums import TA_LEFT
from reportlab.lib import colors

from langchain_openai import ChatOpenAI
import os

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
import uuid

from dotenv import load_dotenv

# Load environment variables (OpenRouter API key)
load_dotenv()
qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")

class GraphState(TypedDict):
    chunk: str
    result: str
    corrections: str
    
# 1. Estrazione testo dal PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    return "".join(page.get_text() for page in doc).strip()

# 2. Chunking del testo
def split_text_into_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    return splitter.create_documents([text])

# 3. Embedding + Retriever
def embed_chunks(chunks):
    # 1. Embedding model da Hugging Face
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 2. Connessione al tuo cluster Qdrant Cloud
    qdrant_client = QdrantClient(
        url = qdrant_url,
        api_key = qdrant_api_key
    )

    collection_name = "rag-frizioni-" + str(uuid.uuid4())[:8]

    # 3. Crea una collection (se non esiste)
    qdrant_client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)  # 384 per MiniLM
    )

    # 4. Carica i documenti nel vector store
    vectorstore = Qdrant.from_documents(
        documents=chunks,
        embedding=embeddings,
        url = qdrant_url,     
        api_key = qdrant_api_key,
        collection_name=collection_name,
    )


    # 5. Ritorna il retriever
    return vectorstore.as_retriever()

# 4. Prompt per l'analisi
analysis_prompt = ChatPromptTemplate.from_messages([
    ("system", "Sei un esperto editor. Analizza il testo seguente e individua i punti di frizione secondo i criteri dati."),
    ("human", """Testo da analizzare:
---
{text}
---

Criteri da usare per identificare i punti di frizione:
1. Descrizioni troppo lunghe o non funzionali.
2. Riferimenti culturali non condivisi dal lettore target.
3. Linguaggio troppo complesso per il pubblico previsto.
4. Dialoghi innaturali o eccessivamente informativi.
5. Cambi di tempo o punto di vista poco chiari.
6. Sovraccarico di nomi o concetti nuovi senza spiegazione.
7. Ritmo narrativo sbilanciato.
8. Presupposizioni di conoscenze specialistiche.

Per ogni punto:
- Riporta l’estratto (max 5 righe)
- Indica il tipo di frizione
- Spiega brevemente il motivo""")
])

# 5. Prompt per la correzione
correction_prompt = ChatPromptTemplate.from_messages([
    ("system", "Sei un editor professionista. Suggerisci una versione migliorata di ogni passaggio problematico."),
    ("human", """Analisi dei punti di frizione:
---
{analysis}
---

Per ciascun passaggio:
- Riscrivi in modo più chiaro e adatto al target.""")
])

# 6. LangGraph: Nodo 1
def analyze_chunk(data):
    MODEL_NAME = 'nvidia/llama-3.1-nemotron-ultra-253b-v1:free'
    llm = ChatOpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url='https://openrouter.ai/api/v1',
        model=MODEL_NAME,
        )
    #llm = Ollama(model="qwen2.5:7b")
    chain = analysis_prompt | llm | StrOutputParser()
    return {"result": chain.invoke({"text": data["chunk"]})}

# 7. LangGraph: Nodo 2
def suggest_corrections(data):
    # llm = Ollama(model="qwen2.5:7b")
    MODEL_NAME = 'nvidia/llama-3.1-nemotron-ultra-253b-v1:free'
    llm = ChatOpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url='https://openrouter.ai/api/v1',
        model=MODEL_NAME,
        )
    chain = correction_prompt | llm | StrOutputParser()
    return {"corrections": chain.invoke({"analysis": data["result"]})}

# 8. Costruzione grafo
builder = StateGraph(GraphState)
builder.add_node("analyze_chunk", analyze_chunk)
builder.add_node("suggest_corrections", suggest_corrections)
builder.set_entry_point("analyze_chunk")
builder.add_edge("analyze_chunk", "suggest_corrections")
builder.set_finish_point("suggest_corrections")
graph = builder.compile()

# 9. Estrazione estratti e tipi
def estrai_estratti(text):
    blocchi = re.findall(r"(?:Estratto:|“|\"|')(.{30,400}?)(?:”|\"|')", text)
    if not blocchi:
        blocchi = re.findall(r"(?:- |• )(.{30,400})", text)
    return blocchi[:5]

def estrai_tipo_frizione(text):
    frizioni = re.findall(r"(?i)(?:tipo di frizione|tipo):\s*(.+)", text)
    return [f.strip().lower() for f in frizioni[:5]]

# 10. Markdown
def salva_markdown(results, output_path="report_frizioni.md"):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# 📘 Report dei Punti di Frizione e Correzioni\n\n")
        for i, item in enumerate(results):
            f.write(f"## 🧩 Chunk {i+1}\n\n")
            f.write("### 🔍 Punti di Frizione:\n")
            f.write("```\n" + item["result"].strip() + "\n```\n\n")
            f.write("### ✏️ Correzioni:\n")
            f.write("```\n" + item["corrections"].strip() + "\n```\n\n")

# 11. PDF con indice
def genera_pdf_annotato(testo_originale, results, output_path="annotazioni_frizioni.pdf"):
    doc = SimpleDocTemplate(output_path, pagesize=A4)
    styles = getSampleStyleSheet()
    toc = TableOfContents()
    toc.levelStyles = [styles["Heading2"]]
    elements = [Paragraph("📘 Annotazioni sui punti di frizione", styles["Heading1"]), toc, PageBreak()]

    color_map = {
        "descrizioni": colors.lightblue,
        "linguaggio complesso": colors.lightyellow,
        "dialoghi": colors.lightgreen,
        "riferimenti culturali": colors.pink
    }

    for i, r in enumerate(results):
        title = f"🧩 Chunk {i+1}"
        elements.append(Paragraph(title, styles["Heading2"]))
        elements[-1].__dict__["bookmark"] = f"chunk_{i+1}"
        estratti = estrai_estratti(r["result"])
        tipi = estrai_tipo_frizione(r["result"])
        correzioni = r["corrections"].split("\n\n")

        for j, (estratto, correzione) in enumerate(zip(estratti, correzioni)):
            tipo = tipi[j] if j < len(tipi) else "generico"
            colore = next((v for k, v in color_map.items() if k in tipo), colors.whitesmoke)
            data = [
                ["🔍 Estratto", estratto.strip()],
                ["✏️ Correzione", correzione.strip()],
                ["🎨 Tipo", tipo.title()]
            ]
            table = Table(data, colWidths=[140, 380])
            table.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colore)]))
            elements.append(table)
            elements.append(Spacer(1, 10))
        elements.append(PageBreak())
    doc.build(elements)

# 12. HTML
def genera_html_annotato(results, output_path="annotazioni_frizioni.html"):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("<html><body><h1>📘 Annotazioni Frizioni</h1><ul>")
        for i in range(len(results)):
            f.write(f"<li><a href='#chunk{i+1}'>Chunk {i+1}</a></li>")
        f.write("</ul><hr>")
        for i, r in enumerate(results):
            f.write(f"<h2 id='chunk{i+1}'>🧩 Chunk {i+1}</h2>")
            estratti = estrai_estratti(r["result"])
            correzioni = r["corrections"].split("\n\n")
            for j, (estratto, correzione) in enumerate(zip(estratti, correzioni)):
                f.write(f"<div><b>🔍 Estratto:</b><pre>{estratto.strip()}</pre></div>")
                f.write(f"<div><b>✏️ Correzione:</b><pre>{correzione.strip()}</pre></div><hr>")
        f.write("</body></html>")

# 13. Main
if __name__ == "__main__":
    pdf_path = "tolstoj_anna_karenina.pdf"  # 🔁 Inserisci qui il nome del tuo file PDF
    text = extract_text_from_pdf(pdf_path)
    chunks = split_text_into_chunks(text)
    retriever = embed_chunks(chunks)
    docs = retriever.get_relevant_documents("analizza il testo alla ricerca di punti di frizione secondo i criteri dati")

    results = []
    for i, doc in enumerate(docs):
        print(f"🔍 Analisi Chunk {i+1}")
        result = graph.invoke({"chunk": doc.page_content})
        results.append(result)

    salva_markdown(results)
    genera_pdf_annotato(text, results)
    genera_html_annotato(results)

    print("✅ Report generati: report_frizioni.md, annotazioni_frizioni.pdf, annotazioni_frizioni.html")
