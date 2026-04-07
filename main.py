import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_KEY"))
LLM_NAME = os.getenv("LLM_NAME")

app = FastAPI(title="French Tutor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# schema definitions
class SentenceRequest(BaseModel):
    sentence: str

class SentenceResponse(BaseModel):
    translation: str
    tense: str
    grammar_points: str | None
    idiomatic_expressions: str | None

class WordRequest(BaseModel):
    word: str

class WordResponse(BaseModel):
    translation: str
    conjugations: list[str] | None
    synonyms: list[str] | None
    common_phrases: list[str] | None
    example_sentence: str


# system prompts
SENTENCE_SYSTEM_PROMPT = """You are a French language tutor. Given a French sentence, respond with a JSON object containing:
- "translation": English translation of the sentence.
- "tense": brief explanation of the grammatical tense (French name).
- "grammar_points": brief explanation of any complex grammar points in English; use null if none.
- "idiomatic_expressions": brief explanation of any idiomatic expressions and their meanings in English; use null if none.

Respond only with valid JSON. No markdown, no extra text."""

WORD_SYSTEM_PROMPT = """You are a French language tutor. Given a French word, respond with a JSON object containing:
- "translation": English translation of the word.
- "conjugations": if it's a verb, an array of its conjugations in present tense (e.g. ["je parle", "tu parles", ...]); if it's a noun, an array of its singular and plural forms (e.g. ["le chat", "les chats"]); use null otherwise.
- "synonyms": an array of French synonyms; use null if none.
- "common_phrases": an array of common phrases using the word, each with English translation after a dash (e.g. ["avoir faim — to be hungry"]); use null if none.
- "example_sentence": one example French sentence using the word, followed by its English translation after a dash.

Respond only with valid JSON. No markdown, no extra text."""


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
    return SentenceResponse(**data)

@app.post("/word", response_model=WordResponse)
def analyze_word(request: WordRequest) -> WordResponse:
    data = call_llm(WORD_SYSTEM_PROMPT, request.word)
    return WordResponse(**data)
