import fitz
import numpy as np
import uuid
from typing import TypedDict
from textblob import TextBlob
import language_tool_python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

# Load environment variables (OpenRouter API key)
load_dotenv()
qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")
# === MODELLI ===
llm = ChatOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model="nvidia/llama-3.1-nemotron-ultra-253b-v1:free",
    temperature=0.1
)

embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
tool = language_tool_python.LanguageTool('it')

# === Qdrant Config ===
QDRANT_API_KEY = qdrant_api_key
QDRANT_URL = qdrant_url
COLLECTION_NAME = "manuscript-trends-" + str(uuid.uuid4())[:8]

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
client.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

# === Tipi ===
class ManuscriptState(TypedDict):
    manuscript: str
    grammar_score: float
    genre: str
    is_accepted_by_genre: bool
    sentiment: str
    sentiment_match: bool
    sentiment_score: float
    market_match: bool
    accepted_genres: list[str]
    trend_keywords: list[str]
    market_score: float

# === Estrazione ===
def extract_text_from_pdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    return "".join(page.get_text("text") for page in doc).strip()

# === Analisi ===
def grammar_check(state: ManuscriptState) -> ManuscriptState:
    matches = tool.check(state["manuscript"])
    num_errors = len(matches)
    total_words = len(state["manuscript"].split())
    score = max(0, 100 - (num_errors / total_words * 100))
    state["grammar_score"] = round(score, 2)
    return state

def classify_genre(state: ManuscriptState) -> ManuscriptState:
    prompt = "Classifica il seguente testo come poesia, narrativa, saggio, o altro:\n" + state["manuscript"][:1000]
    response = llm.invoke(prompt)
    state["genre"] = str(response).strip().lower()
    return state

def filter_by_editor(state: ManuscriptState) -> ManuscriptState:
    state["is_accepted_by_genre"] = state["genre"] in state.get("accepted_genres", [])
    return state

def analyze_sentiment(state: ManuscriptState) -> ManuscriptState:
    blob = TextBlob(state["manuscript"])
    polarity = float(blob.sentiment.polarity)
    state["sentiment"] = "positivo" if polarity > 0 else "negativo" if polarity < 0 else "neutro"
    state["sentiment_score"] = round(polarity, 4)
    state["sentiment_match"] = state["sentiment"] == "positivo"
    return state

def match_with_trends(state: ManuscriptState, retriever=None) -> ManuscriptState:
    if not retriever:
        raise ValueError("Retriever non fornito")

    query_texts = state.get("trend_keywords", [])
    query_embeddings = embed_model.embed_documents(query_texts)
    docs = retriever.invoke(" ".join(query_texts))
    chunk_texts = [doc.page_content for doc in docs]
    chunk_embeddings = embed_model.embed_documents(chunk_texts)

    similarities = [
        np.dot(q_emb, c_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(c_emb))
        for q_emb in query_embeddings for c_emb in chunk_embeddings
    ]

    avg_similarity = float(np.mean(similarities)) if similarities else 0.0
    state["market_score"] = round(avg_similarity, 4)
    state["market_match"] = avg_similarity >= 0.7
    return state

def match_with_trends_factory(retriever):
    def wrapped(state: ManuscriptState) -> ManuscriptState:
        return match_with_trends(state, retriever=retriever)
    return wrapped

# === Analisi completa ===
def run_analysis(pdf_path: str, accepted_genres: list[str], trend_keywords: list[str]) -> dict:
    full_text = extract_text_from_pdf(pdf_path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(full_text)

    db = Qdrant.from_texts(
        texts=chunks,
        embedding=embed_model,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name=COLLECTION_NAME,
    )

    retriever = db.as_retriever()

    builder = StateGraph(ManuscriptState)
    builder.add_node("check_grammar", grammar_check)
    builder.add_node("classify_genre", classify_genre)
    builder.add_node("filter_by_genre", filter_by_editor)
    builder.add_node("analyze_sentiment", analyze_sentiment)
    builder.add_node("match_with_trends", match_with_trends_factory(retriever))

    builder.set_entry_point("check_grammar")
    builder.add_edge("check_grammar", "classify_genre")
    builder.add_edge("classify_genre", "filter_by_genre")
    builder.add_edge("filter_by_genre", "analyze_sentiment")
    builder.add_edge("analyze_sentiment", "match_with_trends")
    builder.add_edge("match_with_trends", END)

    graph = builder.compile()
    result = graph.invoke({
        "manuscript": full_text[:3000],
        "accepted_genres": accepted_genres,
        "trend_keywords": trend_keywords
    })

    return {k: v for k, v in result.items() if k != "manuscript"}
