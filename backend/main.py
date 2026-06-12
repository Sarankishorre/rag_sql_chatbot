from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import sqlite3
import requests
import re
import math
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

# ── startup ──────────────────────────────────────────────────────────────────
print("Loading Titanic dataset...")
df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
df = df.head(500)

conn = sqlite3.connect("titanic.db", check_same_thread=False)
df.to_sql("titanic", conn, if_exists="replace", index=False)
conn.commit()
print("SQLite DB ready.")

# BUILD SCHEMA
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

#question and answer pairs( few shot examples)


SQL_KEYWORDS = [
    "how many", "count", "total", "sum", "average", "avg",
    "rate", "percentage", "percent", "ratio",
    "highest", "lowest", "maximum", "minimum", "max", "min",
    "each", "per", "every", "by class", "by gender", "by sex",
    "compare", "vs", "versus", "difference",
    "show me", "list", "find", "give me",
    "survived", "died", "death", "survival",
    "fare", "embarked", "pclass", "breakdown", "distribution"
]

# These mean the user wants an explanation — NO SQL needed
GENERAL_KEYWORDS = [
    "what does", "what is a", "what are",
    "explain", "define", "meaning",
    "tell me about", "describe",
    "how does", "why does", "what happened",
    "what was the titanic"
]
def classify(question: str) -> str:
    q = question.lower().strip()
    for kw in GENERAL_KEYWORDS:
        if kw in q:
            return "general"
    for kw in SQL_KEYWORDS:
        if kw in q:
            return "sql"
    return "general"
print("Loading sentence transformer...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# CHROMADB SETUP
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

# RETRIEVE RELEVANT COLUMNS USING VECTOR SEARCH
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

# GENERATE SQL FROM QUESTION + HISTORY
from few_shot_examples import FEW_SHOT_EXAMPLES
def generate_sql(query: str, schema_text: str, history: list) -> str:
    system_prompt = f"""You are a SQL expert. Table name is 'titanic'.
    Write ONE valid SQLite SQL query to answer the question.
    Return ONLY the SQL query. No explanation. No markdown. No backticks.
    STRICT RULES — NEVER BREAK THESE:
    - Always use SQLite syntax (LIMIT not TOP, use IS NULL not ISNULL)
    - Always SELECT * or include Name column when user asks for details
    - Only use columns: PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked
    - For UNION queries always wrap each SELECT in subquery
    - NEVER add LIMIT unless user explicitly asks for top N results
    - ALWAYS use GROUP BY when question says "each", "per", "by class", "by gender"
    - ALWAYS multiply rates by 100.0 and use ROUND(..., 2) for percentages
    - ALWAYS use clean column aliases with AS — example: AS death_rate_pct
    - NEVER add filters the user did not ask for
    - When breaking down by TWO categories use GROUP BY col1, col2 — never subqueries

    EXAMPLES OF CORRECT SQL FOR THIS DATABASE:
    {FEW_SHOT_EXAMPLES}

    Column descriptions:
    {schema_text}"""

    messages = [{"role": "system", "content": system_prompt}]

    for msg in history:
        messages.append({"role": msg.role, "content": msg.content})

    messages.append({"role": "user", "content": f"Question: {query}\nSQL:"})

    response = groq_client.chat.completions.create(
        model=groq_model,
        messages=messages
    )
    sql = response.choices[0].message.content.strip()
    sql = re.sub(r"```sql|```", "", sql).strip()
    return sql

# AUTO-FIX SQL IF IT FAILS
def validate_and_fix_sql(sql: str) -> str:
    # fix common mistakes automatically
    sql = re.sub(r'\bSELECT\s+TOP\s+(\d+)\b', r'SELECT', sql, flags=re.IGNORECASE)
    sql = re.sub(r'\bISNULL\s*\(([^,]+),([^)]+)\)', r'COALESCE(\1,\2)', sql, flags=re.IGNORECASE)

    # test run silently
    try:
        pd.read_sql(sql, conn)
        return sql  # works fine
    except Exception as e:
        error_msg = str(e)
        # ask groq to fix it
        fix_prompt = f"""This SQLite query failed with error: {error_msg}

Failed SQL: {sql}

Fix the SQL query for SQLite. Rules:
- Use LIMIT not TOP
- For UNION with ORDER BY, wrap each SELECT in subquery: SELECT * FROM (SELECT ... ORDER BY ... LIMIT n)
- Use IS NULL not ISNULL
- Use COALESCE not ISNULL
- Only use columns: PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked
Return ONLY the fixed SQL. No explanation. No markdown. No backticks."""

        fix_response = groq_client.chat.completions.create(
            model=groq_model,
            messages=[{"role": "user", "content": fix_prompt}]
        )
        fixed_sql = fix_response.choices[0].message.content.strip()
        fixed_sql = re.sub(r"```sql|```", "", fixed_sql).strip()
        return fixed_sql

# RUN SQL AND CLEAN OUTPUT
def run_sql(sql: str):
    result = pd.read_sql(sql, conn)
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

# ── API models ────────────────────────────────────────────────────────────────
class Message(BaseModel):
    role: str
    content: str

class QueryRequest(BaseModel):
    question: str
    history: list[Message] = []

# ── API routes ────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/query")
def query(req: QueryRequest):
    try:
        question_type = classify(req.question)
        if question_type=="general":
            direct_response = groq_client.chat.completions.create(
                model=groq_model,
                messages=[{
                    "role": "user",
                    "content": f"""You are a helpful assistant for a Titanic passenger dataset.
                    Answer this question directly in 2-3 sentences. Do not write any SQL.
                    Question: {req.question}"""
                }])
            description = direct_response.choices[0].message.content.strip()
            return {
                "success": True,
                "sql": "",
                "columns": [],
                "rows": [],
                "count": 0,
                "description": description
            }
        schema_text = retrieve_schema(req.question)
        sql = generate_sql(req.question, schema_text, req.history)

        # auto-fix sql if it fails
        sql = validate_and_fix_sql(sql)

        rows, columns = run_sql(sql)

        # generate friendly description
        summary_prompt = f"""You are a helpful data analyst.
A user asked: "{req.question}"
The SQL query returned {len(rows)} row(s): {rows[:5]}
Write a SHORT 1-2 sentence friendly summary in plain English.
No SQL. No technical terms."""

        summary_response = groq_client.chat.completions.create(
            model=groq_model,
            messages=[{"role": "user", "content": summary_prompt}]
        )
        description = summary_response.choices[0].message.content.strip()

        return {
            "success": True,
            "sql": sql,
            "columns": columns,
            "rows": rows,
            "count": len(rows),
            "description": description
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "sql": "",
            "columns": [],
            "rows": [],
            "count": 0,
            "description": ""
        }

@app.get("/schema")
def get_schema():
    safe_schema = {k: str(v) for k, v in schema.items()}
    return {"schema": safe_schema}