import os
import json
import base64
import pyodbc
import httpx
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_KEY"))
LLM_NAME = os.getenv("LLM_NAME")
DB_CONNECTION_STRING = os.getenv("DB_CONNECTION_STRING")
AZURE_TRANSLATOR_KEY = os.getenv("AZURE_TRANSLATOR_KEY")

app = FastAPI(title="French Tutor API")


# schema definitions
class SentenceRequest(BaseModel):
    sentence: str

class SentenceResponse(BaseModel):
    translation: str
    tense: str | None
    grammar_points: str | None
    idiomatic_expressions: str | None

class WordRequest(BaseModel):
    word: str

class WordResponse(BaseModel):
    valid: bool
    translation: str | None
    conjugations: list[str] | None
    synonyms: list[str] | None
    common_phrases: list[str] | None
    example_sentence: str | None

class QuestionRequest(BaseModel):
    question: str

class QuestionResponse(BaseModel):
    answer: str

class DashboardRequest(BaseModel):
    pass

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
    mode: str  # "random" = pick a random unmastered item; "oldest" = pick the unmastered item with the earliest insertion time

class ReviewResponse(BaseModel):
    id: int
    content: str
    is_word: bool
    is_sentence: bool
    review_count: int
    age: int  # days since the record was inserted

class MarkRequest(BaseModel):
    id: int
    status: str  # "done" = set mastered=true; "not_done" = no-op (keep for future retry)

class ActivityRequest(BaseModel):
    pass

class ActivityResponse(BaseModel):
    created: list[int]   # count of new items per day, index 0 = 30 days ago, index 29 = today
    mastered: list[int]  # count of mastered items per day, same indexing

class InspectRequest(BaseModel):
    pass

class InspectItem(BaseModel):
    id: int
    content: str
    mastered_time: str

class InspectResponse(BaseModel):
    items: list[InspectItem]

class ReplantRequest(BaseModel):
    id: int

class TranslateRequest(BaseModel):
    text: str

class TranslateResponse(BaseModel):
    translation: str


# system prompts
SENTENCE_SYSTEM_PROMPT = """You are a French language tutor. Given a French sentence, respond with a JSON object containing:
- "translation": English translation of the sentence.
- "tense": brief explanation of the grammatical tense (French name); use null if none.
- "grammar_points": brief explanation of any complex grammar points in English; use null if none.
- "idiomatic_expressions": brief explanation of any idiomatic expressions and their meanings in English; use null if none.

Respond only with valid JSON. No extra text. Within each value (the string), wrap bold content in **double asterisks** if there is any. Whenever you quote French within English content, make the French bold."""

QA_SYSTEM_PROMPT = """You are a French language tutor. Answer the student's question thoroughly and clearly. No need to ask follow-up questions or ask for clarification unless necessary.
Respond with a JSON object containing a single key:
- "answer": your full answer as an HTML string. Use <p>, <ul>, <li>, <strong>, <em>, and <code> tags as appropriate to structure and format the response for display on a webpage. Do not include <html>, <head>, or <body> tags.

Respond only with valid JSON. No extra text."""

WORD_SYSTEM_PROMPT = """You are a French language tutor. Given a French word, respond with a JSON object containing:
- "valid": true if the input is a real French word or a recognisable inflected form of one (e.g. a conjugated verb, a plural noun, a feminine adjective) — even if it is not the base/dictionary form. Set to false if the input is a typo, gibberish, or otherwise not a real French word.
- "translation": English translation of the word. Provide the part of speech in parentheses after the translation (e.g. "cat (noun)", "to speak (verb)"). Use null if valid is false.
- "conjugations": if it's a verb, an array of its conjugations in present tense (e.g. ["je parle", "tu parles", ...]); if it's a noun, an array of its singular and plural forms (e.g. ["le chat", "les chats"]); use null otherwise.
- "synonyms": an array of French synonyms; use null if none.
- "common_phrases": an array of common phrases using the word, each with English translation after a dash (e.g. ["avoir faim — to be hungry"]); use null if none. French parts should be in **bold**.
- "example_sentence": one example French sentence using the word, followed by its English translation after a dash. French part should be in **bold**. Use null if valid is false.

Respond only with valid JSON. No extra text. Within each value (the string), wrap bold content in **double asterisks** if there is any."""


def get_user_id(request: Request) -> str:
    user_id = request.headers.get("X-MS-CLIENT-PRINCIPAL-ID")
    if not user_id:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user_id


def log_to_db(user_id: str, content: str, is_word: bool) -> None:
    content = content.strip().lower()
    try:
        conn = pyodbc.connect(DB_CONNECTION_STRING)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO users (user_id, content, is_word, is_sentence, mastered, review_count)
            SELECT ?, ?, ?, ?, 0, 0
            WHERE NOT EXISTS (SELECT 1 FROM users WHERE user_id=? AND content=?)
            """,
            (user_id, content, int(is_word), int(not is_word), user_id, content)
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


_AZURE_HEADERS = {
    "Ocp-Apim-Subscription-Key": AZURE_TRANSLATOR_KEY,
    "Ocp-Apim-Subscription-Region": "centralus",
    "Content-Type": "application/json",
}

def detect_french(text: str) -> bool:
    try:
        resp = httpx.post(
            "https://api.cognitive.microsofttranslator.com/detect",
            params={"api-version": "3.0"},
            headers=_AZURE_HEADERS,
            json=[{"text": text}],
            timeout=10.0,
        )
        resp.raise_for_status()
        detected = resp.json()[0]
        return detected.get("language") == "fr" and detected.get("score", 0) >= 0.5
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def call_translator_word(text: str) -> dict:
    try:
        detect_resp = httpx.post(
            "https://api.cognitive.microsofttranslator.com/detect",
            params={"api-version": "3.0"},
            headers=_AZURE_HEADERS,
            json=[{"text": text}],
            timeout=10.0,
        )
        detect_resp.raise_for_status()
        detected = detect_resp.json()[0]
        valid = detected.get("language") == "fr" and detected.get("score", 0) >= 0.5

        if not valid:
            return {
                "valid": False,
                "translation": None,
                "conjugations": None,
                "synonyms": None,
                "common_phrases": None,
                "example_sentence": None,
            }

        translate_resp = httpx.post(
            "https://api.cognitive.microsofttranslator.com/translate",
            params={"api-version": "3.0", "from": "fr", "to": "en"},
            headers=_AZURE_HEADERS,
            json=[{"text": text}],
            timeout=10.0,
        )
        translate_resp.raise_for_status()
        translation = translate_resp.json()[0]["translations"][0]["text"]

        return {
            "valid": True,
            "translation": translation,
            "conjugations": None,
            "synonyms": None,
            "common_phrases": None,
            "example_sentence": None,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def detect_english(text: str) -> bool:
    try:
        resp = httpx.post(
            "https://api.cognitive.microsofttranslator.com/detect",
            params={"api-version": "3.0"},
            headers=_AZURE_HEADERS,
            json=[{"text": text}],
            timeout=10.0,
        )
        resp.raise_for_status()
        detected = resp.json()[0]
        return detected.get("language") == "en" and detected.get("score", 0) >= 0.85
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def call_translator_word_en(text: str) -> dict:
    try:
        if not detect_english(text):
            return {
                "valid": False,
                "translation": None,
                "conjugations": None,
                "synonyms": None,
                "common_phrases": None,
                "example_sentence": None,
            }

        translate_resp = httpx.post(
            "https://api.cognitive.microsofttranslator.com/translate",
            params={"api-version": "3.0", "from": "en", "to": "fr"},
            headers=_AZURE_HEADERS,
            json=[{"text": text}],
            timeout=10.0,
        )
        translate_resp.raise_for_status()
        translation = translate_resp.json()[0]["translations"][0]["text"]

        return {
            "valid": True,
            "translation": translation,
            "conjugations": None,
            "synonyms": None,
            "common_phrases": None,
            "example_sentence": None,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def call_translator_sentence_en(text: str) -> dict | None:
    try:
        if not detect_english(text):
            return None

        resp = httpx.post(
            "https://api.cognitive.microsofttranslator.com/translate",
            params={"api-version": "3.0", "from": "en", "to": "fr"},
            headers=_AZURE_HEADERS,
            json=[{"text": text}],
            timeout=10.0,
        )
        resp.raise_for_status()
        translation = resp.json()[0]["translations"][0]["text"]

        return {
            "translation": translation,
            "tense": None,
            "grammar_points": None,
            "idiomatic_expressions": None,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def call_translator_sentence(text: str) -> dict:
    try:
        resp = httpx.post(
            "https://api.cognitive.microsofttranslator.com/translate",
            params={"api-version": "3.0", "from": "fr", "to": "en"},
            headers=_AZURE_HEADERS,
            json=[{"text": text}],
            timeout=10.0,
        )
        resp.raise_for_status()
        translation = resp.json()[0]["translations"][0]["text"]

        return {
            "translation": translation,
            "tense": None,
            "grammar_points": None,
            "idiomatic_expressions": None,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _get_claim(request: Request, claim_type: str) -> str | None:
    principal = request.headers.get("X-MS-CLIENT-PRINCIPAL")
    if not principal:
        return None
    try:
        payload = json.loads(base64.b64decode(principal))
        for claim in payload.get("claims", []):
            if claim.get("typ") == claim_type:
                return claim.get("val") or None
    except Exception:
        pass
    return None


@app.get("/me")
def me(request: Request):
    user_id = get_user_id(request)
    name = request.headers.get("X-MS-CLIENT-PRINCIPAL-NAME", "")
    given_name = _get_claim(request, "given_name")
    family_name = _get_claim(request, "family_name")
    preferred_username = _get_claim(request, "email")
    claim_display_name = _get_claim(request, "name")

    display_name = claim_display_name
    try:
        conn = pyodbc.connect(DB_CONNECTION_STRING)
        cursor = conn.cursor()
        cursor.execute(
            """
            IF NOT EXISTS (SELECT 1 FROM user_registry WHERE user_id = ?)
                INSERT INTO user_registry
                    (user_id, name, given_name, family_name, preferred_username, display_name)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
            user_id,
            user_id, name, given_name, family_name, preferred_username, claim_display_name,
        )
        conn.commit()
        cursor.execute(
            "SELECT display_name FROM user_registry WHERE user_id = ?",
            user_id,
        )
        row = cursor.fetchone()
        if row and row[0]:
            display_name = row[0]
        conn.close()
    except Exception:
        pass

    return {"user_id": user_id, "name": name, "given_name": given_name, "display_name": display_name, "preferred_username": preferred_username}


class UpdateDisplayNameRequest(BaseModel):
    display_name: str


@app.post("/update_display_name")
def update_display_name(http_request: Request, request: UpdateDisplayNameRequest):
    user_id = get_user_id(http_request)
    name = request.display_name.strip()
    if not name or len(name) > 25:
        raise HTTPException(status_code=400, detail="display_name must be 1–25 characters")
    try:
        conn = pyodbc.connect(DB_CONNECTION_STRING)
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE user_registry SET display_name = ? WHERE user_id = ?",
            name, user_id,
        )
        conn.commit()
        conn.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")
    return {"display_name": name}


@app.post("/sentence", response_model=SentenceResponse)
def analyze_sentence(http_request: Request, request: SentenceRequest) -> SentenceResponse:
    user_id = get_user_id(http_request)
    if not detect_french(request.sentence):
        raise HTTPException(status_code=400, detail="That doesn't look like a French sentence — check your input and try again.")
    data = call_llm(SENTENCE_SYSTEM_PROMPT, request.sentence)
    log_to_db(user_id, request.sentence, is_word=False)
    return SentenceResponse(**data)

@app.post("/word", response_model=WordResponse)
def analyze_word(http_request: Request, request: WordRequest) -> WordResponse:
    user_id = get_user_id(http_request)
    data = call_llm(WORD_SYSTEM_PROMPT, request.word)
    if not data.get("valid"):
        raise HTTPException(status_code=400, detail="That doesn't look like a French word — check your spelling and try again.")
    log_to_db(user_id, request.word, is_word=True)
    return WordResponse(**data)

@app.post("/word_quick", response_model=WordResponse)
def analyze_word_quick(http_request: Request, request: WordRequest) -> WordResponse:
    user_id = get_user_id(http_request)
    data = call_translator_word(request.word)
    if not data.get("valid"):
        raise HTTPException(status_code=400, detail="That doesn't look like a French word — check your spelling and try again.")
    log_to_db(user_id, request.word, is_word=True)
    return WordResponse(**data)

@app.post("/sentence_quick", response_model=SentenceResponse)
def analyze_sentence_quick(http_request: Request, request: SentenceRequest) -> SentenceResponse:
    user_id = get_user_id(http_request)
    if not detect_french(request.sentence):
        raise HTTPException(status_code=400, detail="That doesn't look like a French sentence — check your input and try again.")
    data = call_translator_sentence(request.sentence)
    log_to_db(user_id, request.sentence, is_word=False)
    return SentenceResponse(**data)

@app.post("/word_en_quick", response_model=WordResponse)
def analyze_word_en_quick(http_request: Request, request: WordRequest) -> WordResponse:
    user_id = get_user_id(http_request)
    data = call_translator_word_en(request.word)
    if not data.get("valid"):
        raise HTTPException(status_code=400, detail="That doesn't look like an English word — check your spelling and try again.")
    log_to_db(user_id, data["translation"], is_word=True)
    return WordResponse(**data)

@app.post("/sentence_en_quick", response_model=SentenceResponse)
def analyze_sentence_en_quick(http_request: Request, request: SentenceRequest) -> SentenceResponse:
    user_id = get_user_id(http_request)
    data = call_translator_sentence_en(request.sentence)
    if data is None:
        raise HTTPException(status_code=400, detail="That doesn't look like an English sentence — check your input and try again.")
    log_to_db(user_id, data["translation"], is_word=False)
    return SentenceResponse(**data)

@app.post("/qa", response_model=QuestionResponse)
def answer_question(request: QuestionRequest) -> QuestionResponse:
    data = call_llm(QA_SYSTEM_PROMPT, request.question)
    return QuestionResponse(**data)

@app.post("/translate", response_model=TranslateResponse)
def translate_text(http_request: Request, request: TranslateRequest) -> TranslateResponse:
    get_user_id(http_request)  # auth check only
    resp = httpx.post(
        "https://api.cognitive.microsofttranslator.com/translate",
        params={"api-version": "3.0", "from": "fr", "to": "en"},
        headers={
            "Ocp-Apim-Subscription-Key": AZURE_TRANSLATOR_KEY,
            "Ocp-Apim-Subscription-Region": "centralus",
            "Content-Type": "application/json",
        },
        json=[{"text": request.text}],
        timeout=10.0,
    )
    resp.raise_for_status()
    translation = resp.json()[0]["translations"][0]["text"]
    return TranslateResponse(translation=translation)

@app.post("/review_item", response_model=ReviewResponse)
def review_item(http_request: Request, request: ReviewRequest) -> ReviewResponse:
    user_id = get_user_id(http_request)
    try:
        conn = pyodbc.connect(DB_CONNECTION_STRING)
        cursor = conn.cursor()

        # Select the target item based on mode before mutating
        if request.mode == "oldest":
            cursor.execute(
                "SELECT TOP 1 id FROM users WHERE user_id=? AND mastered=0 ORDER BY time ASC",
                user_id
            )
        else:  # "random"
            cursor.execute(
                "SELECT TOP 1 id FROM users WHERE user_id=? AND mastered=0 ORDER BY NEWID()",
                user_id
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
def mark_item(http_request: Request, request: MarkRequest) -> None:
    # status="done"     → sets mastered=true on the item
    # status="not_done" → no-op; item remains in the review pool
    if request.status != "done":
        return

    user_id = get_user_id(http_request)
    try:
        conn = pyodbc.connect(DB_CONNECTION_STRING)
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE users SET mastered=1, mastered_time=SYSUTCDATETIME() WHERE id=? AND user_id=?",
            (request.id, user_id)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")

@app.post("/activity", response_model=ActivityResponse)
def get_activity(http_request: Request, request: ActivityRequest) -> ActivityResponse:
    user_id = get_user_id(http_request)
    try:
        conn = pyodbc.connect(DB_CONNECTION_STRING)
        cursor = conn.cursor()

        # Returns (days_ago, count) for items created in last 30 days
        cursor.execute(
            """
            SELECT DATEDIFF(day, time, SYSUTCDATETIME()) AS days_ago, COUNT(*) AS cnt
            FROM users
            WHERE user_id=? AND DATEDIFF(day, time, SYSUTCDATETIME()) BETWEEN 0 AND 29
            GROUP BY DATEDIFF(day, time, SYSUTCDATETIME())
            """,
            user_id
        )
        created_map = {row[0]: row[1] for row in cursor.fetchall()}

        # Returns (days_ago, count) for items mastered in last 30 days
        cursor.execute(
            """
            SELECT DATEDIFF(day, mastered_time, SYSUTCDATETIME()) AS days_ago, COUNT(*) AS cnt
            FROM users
            WHERE user_id=? AND mastered_time IS NOT NULL
              AND DATEDIFF(day, mastered_time, SYSUTCDATETIME()) BETWEEN 0 AND 29
            GROUP BY DATEDIFF(day, mastered_time, SYSUTCDATETIME())
            """,
            user_id
        )
        mastered_map = {row[0]: row[1] for row in cursor.fetchall()}

        conn.close()

        # index 0 = 30 days ago, index 29 = today  →  days_ago = 29 - i
        created  = [created_map.get(29 - i, 0) for i in range(30)]
        mastered = [mastered_map.get(29 - i, 0) for i in range(30)]

        return ActivityResponse(created=created, mastered=mastered)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")

@app.post("/inspect", response_model=InspectResponse)
def inspect_bloomed(http_request: Request, request: InspectRequest) -> InspectResponse:
    user_id = get_user_id(http_request)
    try:
        conn = pyodbc.connect(DB_CONNECTION_STRING)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, content, mastered_time FROM users WHERE user_id=? AND mastered=1 ORDER BY mastered_time DESC",
            (user_id,)
        )
        items = [
            InspectItem(id=row[0], content=row[1], mastered_time=row[2].isoformat())
            for row in cursor.fetchall()
        ]
        conn.close()
        return InspectResponse(items=items)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")

@app.post("/replant", status_code=204)
def replant(request: ReplantRequest) -> None:
    try:
        conn = pyodbc.connect(DB_CONNECTION_STRING)
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE users SET mastered=0, time=SYSUTCDATETIME(), mastered_time=NULL WHERE id=?",
            request.id
        )
        conn.commit()
        conn.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")

@app.post("/dashboard", response_model=DashboardResponse)
def get_dashboard(http_request: Request, request: DashboardRequest) -> DashboardResponse:
    user_id = get_user_id(http_request)
    try:
        conn = pyodbc.connect(DB_CONNECTION_STRING)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM users WHERE user_id=? AND is_word=1 AND mastered=0", user_id)
        word_seeds = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM users WHERE user_id=? AND is_sentence=1 AND mastered=0", user_id)
        sentence_seeds = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM users WHERE user_id=? AND mastered=0", user_id)
        total_seeds = cursor.fetchone()[0]

        cursor.execute("SELECT COALESCE(SUM(review_count), 0) FROM users WHERE user_id=? AND is_word=1", user_id)
        word_water = cursor.fetchone()[0]

        cursor.execute("SELECT COALESCE(SUM(review_count), 0) FROM users WHERE user_id=? AND is_sentence=1", user_id)
        sentence_water = cursor.fetchone()[0]

        cursor.execute("SELECT COALESCE(SUM(review_count), 0) FROM users WHERE user_id=?", user_id)
        total_water = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM users WHERE user_id=? AND is_word=1 AND mastered=1", user_id)
        word_flowers = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM users WHERE user_id=? AND is_sentence=1 AND mastered=1", user_id)
        sentence_flowers = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM users WHERE user_id=? AND mastered=1", user_id)
        total_flowers = cursor.fetchone()[0]

        cursor.execute(
            "SELECT DATEDIFF(day, MIN(time), SYSUTCDATETIME()) FROM users WHERE user_id=? AND mastered=0",
            user_id
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
