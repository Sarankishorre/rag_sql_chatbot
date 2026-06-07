from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import sqlite3
import requests
import re
from sentence_transformers import SentenceTransformer
import chromadb
from groq import Groq
from dotenv import load_dotenv
import os

app = FastAPI(title="Titanic SQL Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── startup: load data, build schema, populate vectorDB ──────────────────────
print("Loading Titanic dataset...")
df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
df = df.head(500)

conn = sqlite3.connect("titanic.db", check_same_thread=False)
df.to_sql("titanic", conn, if_exists="replace", index=False)
conn.commit()
print("SQLite DB ready.")

def build_schema(df, col):
    col_values = df[col]
    dtypee = str(col_values.dtype)
    clean = col_values.dropna()
    if clean.nunique() <= 12:
        value = sorted(clean.unique().tolist())
        value_info = f"unique values: {value}"
    elif dtypee in ["int64", "float64"]:
        col_min = float(clean.min()) if len(clean) else 0
        col_max = float(clean.max()) if len(clean) else 0
        col_mean = round(float(clean.mean()), 2) if len(clean) else 0
        value_info = f"range: {col_min} to {col_max}, mean: {col_mean}"
    else:
        sample = clean.sample(min(5, len(clean)), random_state=42).tolist()
        value_info = f"sample texts: {sample}"
    return f"column: {col}\n dtype: {dtypee}\n{value_info}\n"

schema = {col: build_schema(df, col) for col in df.columns}
print(f"Schema built for {len(schema)} columns.")

print("Loading sentence transformer...")
model = SentenceTransformer("all-MiniLM-L6-v2")

chroma_client = chromadb.PersistentClient(path="./chroma_db")
try:
    chroma_client.delete_collection("titanic_collections")
except:
    pass
collection = chroma_client.create_collection("titanic_collections", metadata={"hnsw:space": "cosine"})

for col, desc in schema.items():
    vector = model.encode(desc).tolist()
    collection.add(ids=[col], documents=[desc], embeddings=[vector])
print("ChromaDB ready.")

# ── helpers ───────────────────────────────────────────────────────────────────
def retrieve_schema(query: str, top_k: int = 5) -> str:
    vector = model.encode(query).tolist()
    result = collection.query(query_embeddings=[vector], n_results=top_k)
    return "\n\n".join(result["documents"][0])

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise RuntimeError("GROQ_API_KEY is not set in backend/.env or environment")

groq_model = os.getenv("GROQ_MODEL")
if not groq_model:
    raise RuntimeError("GROQ_MODEL must be set in backend/.env or the environment.")
groq_client = Groq(api_key=groq_api_key)
def generate_sql(query: str, schema_text: str) -> str:
    prompt = f"""You are a SQL expert. Table name is 'titanic'.
Write ONE valid SQLite SQL query to answer the question.
Return ONLY the SQL query. No explanation. No markdown. No backticks.

Column descriptions:
{schema_text}
Question: {query}
SQL:"""
    response = groq_client.chat.completions.create(
        model=groq_model,
        messages=[{"role": "user", "content": prompt}]
    )
    sql = response.choices[0].message.content.strip()
    sql = re.sub(r"```sql|```", "", sql).strip()
    return sql

import math

def run_sql(sql: str):
    result = pd.read_sql(sql, conn)
    # Replace NaN/inf with None so JSON can serialize it
    result = result.where(pd.notnull(result), other=None)
    rows = []
    for record in result.to_dict(orient="records"):
        clean = {}
        for k, v in record.items():
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                clean[k] = None
            else:
                clean[k] = v
        rows.append(clean)
    return rows, list(result.columns)

# ── API routes ────────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/query")
def query(req: QueryRequest):
    try:
        schema_text = retrieve_schema(req.question)
        sql = generate_sql(req.question, schema_text)
        rows, columns = run_sql(sql)
        return {
            "success": True,
            "sql": sql,
            "columns": columns,
            "rows": rows,
            "count": len(rows)
        }
    except Exception as e:
        return {"success": False, "error": str(e), "sql": "", "columns": [], "rows": [], "count": 0}

@app.get("/schema")
def get_schema():
    safe_schema = {k: str(v) for k, v in schema.items()}
    return {"schema": safe_schema}