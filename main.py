import os
import json
import pyodbc
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_KEY"))
LLM_NAME = os.getenv("LLM_NAME")
DB_CONNECTION_STRING = os.getenv("DB_CONNECTION_STRING")

app = FastAPI(title="French Tutor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "https://icy-ocean-093948e10.7.azurestaticapps.net"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# schema definitions
class SentenceRequest(BaseModel):
    sentence: str
    user_id: str

class SentenceResponse(BaseModel):
    translation: str
    tense: str
    grammar_points: str | None
    idiomatic_expressions: str | None

class WordRequest(BaseModel):
    word: str
    user_id: str

class WordResponse(BaseModel):
    translation: str
    conjugations: list[str] | None
    synonyms: list[str] | None
    common_phrases: list[str] | None
    example_sentence: str

class QuestionRequest(BaseModel):
    question: str

class QuestionResponse(BaseModel):
    answer: str

class DashboardRequest(BaseModel):
    user_id: str

class DashboardResponse(BaseModel):
    word_seeds: int
    sentence_seeds: int
    total_seeds: int
    word_water: int
    sentence_water: int
    total_water: int
    word_flowers: int
    sentence_flowers: int
    total_flowers: int
    oldest_seed_days: int | None

class ReviewRequest(BaseModel):
    user_id: str
    mode: str  # "random" = pick a random unmastered item; "oldest" = pick the unmastered item with the earliest insertion time

class ReviewResponse(BaseModel):
    id: int
    content: str
    is_word: bool
    is_sentence: bool
    review_count: int
    age: int  # days since the record was inserted

class MarkRequest(BaseModel):
    user_id: str
    id: int
    status: str  # "done" = set mastered=true; "not_done" = no-op (keep for future retry)


# system prompts
SENTENCE_SYSTEM_PROMPT = """You are a French language tutor. Given a French sentence, respond with a JSON object containing:
- "translation": English translation of the sentence.
- "tense": brief explanation of the grammatical tense (French name).
- "grammar_points": brief explanation of any complex grammar points in English; use null if none.
- "idiomatic_expressions": brief explanation of any idiomatic expressions and their meanings in English; use null if none.

Respond only with valid JSON. No extra text. Within each value (the string), wrap bold content in **double asterisks** if there is any. Whenever you quote French within English content, make the French bold."""

QA_SYSTEM_PROMPT = """You are a French language tutor. Answer the student's question thoroughly and clearly. No need to ask follow-up questions or ask for clarification unless necessary.
Respond with a JSON object containing a single key:
- "answer": your full answer as an HTML string. Use <p>, <ul>, <li>, <strong>, <em>, and <code> tags as appropriate to structure and format the response for display on a webpage. Do not include <html>, <head>, or <body> tags.

Respond only with valid JSON. No extra text."""

WORD_SYSTEM_PROMPT = """You are a French language tutor. Given a French word, respond with a JSON object containing:
- "translation": English translation of the word.
- "conjugations": if it's a verb, an array of its conjugations in present tense (e.g. ["je parle", "tu parles", ...]); if it's a noun, an array of its singular and plural forms (e.g. ["le chat", "les chats"]); use null otherwise.
- "synonyms": an array of French synonyms; use null if none.
- "common_phrases": an array of common phrases using the word, each with English translation after a dash (e.g. ["avoir faim — to be hungry"]); use null if none. French parts should be in **bold**.
- "example_sentence": one example French sentence using the word, followed by its English translation after a dash. French part should be in **bold**.

Respond only with valid JSON. No extra text. Within each value (the string), wrap bold content in **double asterisks** if there is any."""


def log_to_db(user_id: str, content: str, is_word: bool) -> None:
    try:
        conn = pyodbc.connect(DB_CONNECTION_STRING)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO users (user_id, content, is_word, is_sentence, mastered, review_count) VALUES (?, ?, ?, ?, ?, ?)",
            (user_id, content, int(is_word), int(not is_word), 0, 0)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")


def call_llm(system_prompt: str, user_message: str) -> dict:
    try:
        response = client.chat.completions.create(
            model=LLM_NAME,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/sentence", response_model=SentenceResponse)
def analyze_sentence(request: SentenceRequest) -> SentenceResponse:
    data = call_llm(SENTENCE_SYSTEM_PROMPT, request.sentence)
    log_to_db(request.user_id, request.sentence, is_word=False)
    return SentenceResponse(**data)

@app.post("/word", response_model=WordResponse)
def analyze_word(request: WordRequest) -> WordResponse:
    data = call_llm(WORD_SYSTEM_PROMPT, request.word)
    log_to_db(request.user_id, request.word, is_word=True)
    return WordResponse(**data)

@app.post("/qa", response_model=QuestionResponse)
def answer_question(request: QuestionRequest) -> QuestionResponse:
    data = call_llm(QA_SYSTEM_PROMPT, request.question)
    return QuestionResponse(**data)

@app.post("/review_item", response_model=ReviewResponse)
def review_item(request: ReviewRequest) -> ReviewResponse:
    try:
        conn = pyodbc.connect(DB_CONNECTION_STRING)
        cursor = conn.cursor()

        # Select the target item based on mode before mutating
        if request.mode == "oldest":
            cursor.execute(
                "SELECT TOP 1 id FROM users WHERE user_id=? AND mastered=0 ORDER BY time ASC",
                request.user_id
            )
        else:  # "random"
            cursor.execute(
                "SELECT TOP 1 id FROM users WHERE user_id=? AND mastered=0 ORDER BY NEWID()",
                request.user_id
            )

        row = cursor.fetchone()
        if row is None:
            conn.close()
            raise HTTPException(status_code=404, detail="No unmastered items found for this user.")

        item_id = row[0]

        # Increment review_count before returning
        cursor.execute("UPDATE users SET review_count = review_count + 1 WHERE id=?", item_id)
        conn.commit()

        cursor.execute(
            "SELECT id, content, is_word, is_sentence, review_count, DATEDIFF(day, time, SYSUTCDATETIME()) FROM users WHERE id=?",
            item_id
        )
        item = cursor.fetchone()
        conn.close()

        return ReviewResponse(
            id=item[0],
            content=item[1],
            is_word=bool(item[2]),
            is_sentence=bool(item[3]),
            review_count=item[4],
            age=item[5],
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")

@app.post("/mark_item", status_code=204)
def mark_item(request: MarkRequest) -> None:
    # status="done"     → sets mastered=true on the item
    # status="not_done" → no-op; item remains in the review pool
    if request.status != "done":
        return

    try:
        conn = pyodbc.connect(DB_CONNECTION_STRING)
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE users SET mastered=1, mastered_time=SYSUTCDATETIME() WHERE id=? AND user_id=?",
            (request.id, request.user_id)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")

@app.post("/dashboard", response_model=DashboardResponse)
def get_dashboard(request: DashboardRequest) -> DashboardResponse:
    try:
        conn = pyodbc.connect(DB_CONNECTION_STRING)
        cursor = conn.cursor()
        uid = request.user_id

        cursor.execute("SELECT COUNT(*) FROM users WHERE user_id=? AND is_word=1 AND mastered=0", uid)
        word_seeds = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM users WHERE user_id=? AND is_sentence=1 AND mastered=0", uid)
        sentence_seeds = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM users WHERE user_id=? AND mastered=0", uid)
        total_seeds = cursor.fetchone()[0]

        cursor.execute("SELECT COALESCE(SUM(review_count), 0) FROM users WHERE user_id=? AND is_word=1", uid)
        word_water = cursor.fetchone()[0]

        cursor.execute("SELECT COALESCE(SUM(review_count), 0) FROM users WHERE user_id=? AND is_sentence=1", uid)
        sentence_water = cursor.fetchone()[0]

        cursor.execute("SELECT COALESCE(SUM(review_count), 0) FROM users WHERE user_id=?", uid)
        total_water = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM users WHERE user_id=? AND is_word=1 AND mastered=1", uid)
        word_flowers = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM users WHERE user_id=? AND is_sentence=1 AND mastered=1", uid)
        sentence_flowers = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM users WHERE user_id=? AND mastered=1", uid)
        total_flowers = cursor.fetchone()[0]

        cursor.execute(
            "SELECT DATEDIFF(day, MIN(time), SYSUTCDATETIME()) FROM users WHERE user_id=? AND mastered=0",
            uid
        )
        oldest_seed_days = cursor.fetchone()[0]

        conn.close()
        return DashboardResponse(
            word_seeds=word_seeds,
            sentence_seeds=sentence_seeds,
            total_seeds=total_seeds,
            word_water=word_water,
            sentence_water=sentence_water,
            total_water=total_water,
            word_flowers=word_flowers,
            sentence_flowers=sentence_flowers,
            total_flowers=total_flowers,
            oldest_seed_days=oldest_seed_days,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")
