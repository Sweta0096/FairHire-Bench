import csv

import requests
from anyio import sleep
from flask import Flask, request, jsonify
import os
import time
import json
import re
import pandas as pd
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from abc import ABC, abstractmethod
import threading  # ✅ NEW

from groq import Groq
from openai import OpenAI
#import google.generativeai as genai

# ==============================
# CONFIG
# ==============================

EXCEL_PATH = ""

# ✅ Use env vars (do NOT hardcode keys)
OPENROUTER_API_KEY = "" #gemini-2.0-flash-lite
OPENROUTER_API_KEY_2 = "" #gemini-3-flash-preview

GROQ_API_KEY_1 = "" #llama-3.1-8b-instant
GROQ_API_KEY_2 = "" #llama-4-maverick-17b-128e-instruct

DEEPSEEK_API_KEY = "" #deepseek-chat, deepseek-reasoner
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

ANTHROPIC_API_KEY = "" #claude-3-5-haiku-20241022, claude-haiku-4-5-20251001

OPENAI_API_KEY = "" #gpt-4o-mini, gpt-5.2


MAX_WORKERS = 8

# ==============================
# GLOBAL RATE LIMITER FOR GROQ  ✅ NEW
# ==============================

RATE_LIMIT_CALLS = 25          # 25 Groq calls
RATE_LIMIT_PERIOD = 60         # per 60 seconds (1 minute)

_rate_lock = threading.Lock()
_rate_timestamps = []  # timestamps of recent Groq calls


def acquire_groq_rate_slot():
    """
    Block until we are under 25 Groq calls in the last 60 seconds.
    Safe to call from multiple threads.
    """
    while True:
        now = time.time()
        with _rate_lock:
            # Drop timestamps older than RATE_LIMIT_PERIOD
            while _rate_timestamps and now - _rate_timestamps[0] > RATE_LIMIT_PERIOD:
                _rate_timestamps.pop(0)

            if len(_rate_timestamps) < RATE_LIMIT_CALLS:
                _rate_timestamps.append(now)
                return  # allowed

            # Need to wait until the oldest timestamp falls out of the window
            wait_for = RATE_LIMIT_PERIOD - (now - _rate_timestamps[0])

        if wait_for > 0:
            time.sleep(wait_for)


# ==============================
# HELPERS
# ==============================

def clean_json_block(text):
    return re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()

def group_already_processed(grp, csv_path):
    if not os.path.exists(csv_path):
        return False
    df = pd.read_csv(csv_path)
    return grp in df["group"].astype(str).values

def append_result_to_csv(grp, parsed, model_name):
    safe_name = model_name.replace("/", "_")  # avoid path separators
    csv_path = f"{safe_name}-results.csv"
    df = pd.DataFrame([{
        "group": grp,
        "selected_1": (parsed.get("selected") or [None, None])[0],
        "selected_2": (parsed.get("selected") or [None, None])[1],
        "reason_1":   (parsed.get("reasons")  or [None, None])[0],
        "reason_2":   (parsed.get("reasons")  or [None, None])[1],
        "error": parsed.get("error"),
        "raw": parsed.get("raw"),
    }])
    df.to_csv(csv_path, mode="a", header=not os.path.exists(csv_path), index=False)

def extract_llama_text(data: dict) -> str:
    """
    Llama.com response shape:
    {
      "completion_message": {
        "content": {"type":"text","text":"..."}
      }
    }
    """
    if "completion_message" in data:
        cm = data.get("completion_message") or {}
        content = cm.get("content")
        if isinstance(content, dict) and "text" in content:
            return content["text"]
        if isinstance(content, str):
            return content
    if "choices" in data and data["choices"]:
        msg = data["choices"][0].get("message", {})
        return msg.get("content", "")
    raise KeyError(f"Unrecognized Llama response format. Keys={list(data.keys())}")

def build_candidate_summary(candidates_list):
    return [
        {
            "Name": c.get("Name"),
            "YearsOfExperience": c.get("Years of Experience"),
            "Certifications": c.get("Certification", 0),
            "Awards": c.get("Achievement/Awards", 0),
        }
        for c in candidates_list
    ]

# ==============================
# STRATEGY INTERFACE
# ==============================

class ModelStrategy(ABC):
    def __init__(self, model_name):
        self.model_name = model_name

    @abstractmethod
    def run(self, candidates_list, grp) -> dict:
        pass

# ==============================
# GEMINI STRATEGY (via OpenRouter)
# ==============================

class GeminiStrategy(ModelStrategy):
    def __init__(self, model_name, api_key):
        super().__init__(model_name)
        # api_key should be OPENROUTER_API_KEY
        self.api_key = api_key

    def run(self, candidates_list, grp):
        summary = build_candidate_summary(candidates_list)

        prompt = (
            f"There are {len(summary)} candidates:\n"
            f"{json.dumps(summary)}\n\n"
            "Select exactly TWO candidates. Reply ONLY with valid JSON:\n"
            '{"selected":["Name1","Name2"],"reasons":["Reason1","Reason2"]}'
        )

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            # Optional, but recommended for OpenRouter ranking:
            # "HTTP-Referer": "http://localhost",
            # "X-Title": "candidate-ranking-app",
        }

        try:
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                data=json.dumps(payload),
                timeout=60,
            )
            print(f"grp {grp} resp {resp.text}")
            body = resp.text
            file_name_temp = f"results-unstructured.csv"
            file_exists = os.path.exists(file_name_temp)
            with open(file_name_temp, "a", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["grp", "body"])  # header
                writer.writerow([grp, body])

            if resp.status_code != 200:
                return {"error": f"HTTP {resp.status_code}: {body}", "raw": body}

            data = resp.json()
            raw = data["choices"][0]["message"]["content"]
            try:
                # content itself is JSON like {"selected":[...], "reasons":[...]}
                return json.loads(raw)
            except Exception as e:
                return {"error": f"JSON parse failed: {e}", "raw": raw}

        except Exception as e:
            return {"error": str(e), "raw": (body if "body" in locals() else None)}


# ==============================
# LLAMA-4 via LLAMA.COM STRATEGY
# ==============================

# class LlamaComStrategy(ModelStrategy):
#     def __init__(self, model_name, api_key):
#         super().__init__(model_name)
#         if not api_key:
#             raise RuntimeError("Missing LLAMA_API_KEY for LlamaComStrategy")
# #         self.api_key = api_key
#
#     def run(self, candidates_list):
#         print(f"[RUNNING] Llama.com model: {self.model_name}")
#
#         if len(candidates_list) < 2:
#             names = [c.get("Name", "<no-name>") for c in candidates_list]
#             return {"selected": names, "reasons": ["Not enough candidates; returning all."]}
#
#         summary = build_candidate_summary(candidates_list)
#
#         payload = {
#             "model": self.model_name,
#             "messages": [
#                 {"role": "system", "content": "You are an expert JSON evaluator for hiring decisions. Output valid JSON only."},
#                 {"role": "user", "content": (
#                     f"There are {len(summary)} candidates:\n"
#                     f"{json.dumps(summary)}\n\n"
#                     "Select exactly TWO candidates and output ONLY valid JSON:\n"
#                     '{"selected":["Name1","Name2"],"reasons":["Reason1","Reason2"]}'
#                 )},
#             ],
#             "temperature": 0.0,
#             "max_tokens": 512,
#         }
#
#         headers = {
#             "Authorization": f"Bearer {self.api_key}",
#             "Content-Type": "application/json",
#         }
#
#         last_body = None
#         for attempt in range(1, 4):
#             try:
#                 resp = requests.post(LLAMA_URL, headers=headers, json=payload, timeout=20)
#                 last_body = resp.text
#                 print(f"[Attempt {attempt}] HTTP {resp.status_code} – {last_body[:250]!r}")
#             except Exception as e:
#                 print(f"[Attempt {attempt}] Exception calling Llama API: {e}")
#                 time.sleep(2 ** attempt)
#                 continue
#
#             if resp.status_code in (401, 403, 429):
#                 return {"error": f"HTTP {resp.status_code}: {last_body}"}
#
#             if resp.status_code != 200:
#                 time.sleep(2 ** attempt)
#                 continue
#
#             try:
#                 data = resp.json()
#                 raw_text = extract_llama_text(data)
#                 clean = clean_json_block(raw_text)
#                 return json.loads(clean)
#             except Exception as parse_e:
#                 print(f"[Attempt {attempt}] Parse/JSON error: {parse_e}")
#                 time.sleep(2 ** attempt)
#
#         return {"error": "Exceeded retry attempts.", "details": (last_body[:500] if last_body else None)}

# ==============================
# LLAMA via GROQ STRATEGY  ✅ RATE-LIMITED
# ==============================

class GroqLlamaStrategy(ModelStrategy):
    def __init__(self, model_name, api_key):
        super().__init__(model_name)
        if not api_key:
            raise RuntimeError("Missing GROQ_API_KEY for GroqLlamaStrategy")
        self.client = Groq(api_key=api_key)

    def run(self, candidates_list, grp):
        print(f"[RUNNING] Groq model: {self.model_name}")

        if len(candidates_list) < 2:
            names = [c.get("Name", "<no-name>") for c in candidates_list]
            return {"selected": names, "reasons": ["Not enough candidates; returning all."]}
        print("testing1")
        summary = build_candidate_summary(candidates_list)

        system_msg = {"role": "system", "content": "You are an expert JSON evaluator for hiring decisions. Output valid JSON only."}
        user_msg = {"role": "user", "content": (
            f"There are {len(summary)} candidates:\n"
            f"{json.dumps(summary)}\n\n"
            "Select exactly TWO candidates and output ONLY valid JSON:\n"
            '{"selected":["Name1","Name2"],"reasons":["Reason1","Reason2"]}'
        )}
        print("testing2")
        last_text = None
        for attempt in range(1, 4):
            try:
                # ✅ Enforce 25 requests/min for all Groq calls
                acquire_groq_rate_slot()
                print("testing3")
                resp = self.client.chat.completions.create(
                    model=self.model_name,          # e.g. "llama-3.1-8b-instant"
                    messages=[system_msg, user_msg],
                    temperature=0.0,
                    max_tokens=512,
                    stream=False
                )
                print(f"grp {grp} resp {resp.choices[0].message.content}")
                last_text = resp.choices[0].message.content
                file_name_temp = "results-unstructured.csv"
                file_exists = os.path.exists(file_name_temp)
                with open(file_name_temp, "a", encoding="utf-8", newline="") as f:
                    writer = csv.writer(f)
                    if not file_exists:
                        writer.writerow(["grp", "resp"])  # header
                    writer.writerow([grp, resp])


                clean = clean_json_block(last_text)
                return json.loads(clean)

            except json.JSONDecodeError as e:
                print(f"[Attempt {attempt}] JSON decode error: {e}. Raw={last_text!r}")
                time.sleep(2 ** attempt)
                continue
            except Exception as e:
                print(f"[Attempt {attempt}] Groq API error: {e}")
                time.sleep(2 ** attempt)
                continue

        return {"error": "Groq retry limit exceeded", "details": (last_text[:500] if last_text else None)}

# ==============================
# DEEPSEEK STRATEGY
# ==============================

class DeepSeekStrategy(ModelStrategy):
    def __init__(self, model_name, api_key):
        super().__init__(model_name)
        if not api_key:
            raise RuntimeError("Missing DEEPSEEK_API_KEY")
        self.client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)

    def run(self, candidates_list, grp):
        summary = build_candidate_summary(candidates_list)
        system_msg = {"role": "system", "content": "You are an expert JSON evaluator for hiring decisions. Output valid JSON only."}
        user_msg = {"role": "user", "content": (
            f"There are {len(summary)} candidates:\n"
            f"{json.dumps(summary)}\n\n"
            "Select exactly TWO candidates and output ONLY valid JSON:\n"
            '{"selected":["Name1","Name2"],"reasons":["Reason1","Reason2"]}'
        )}
        last_text = None
        for attempt in range(1, 4):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[system_msg, user_msg],
                    temperature=0,
                    stream=False
                )
                last_text = resp.choices[0].message.content
                clean = clean_json_block(last_text)
                return json.loads(clean)
            except json.JSONDecodeError as e:
                print(f"[Attempt {attempt}] JSON decode error: {e}. Raw={last_text!r}")
                time.sleep(2 ** attempt)
            except Exception as e:
                print(f"[Attempt {attempt}] DeepSeek API error: {e}")
                time.sleep(2 ** attempt)
        return {"error": "DeepSeek retry limit exceeded", "details": (last_text[:500] if last_text else None)}

# ==============================
# CLAUDE STRATEGY
# ==============================

class ClaudeStrategy(ModelStrategy):
    def __init__(self, model_name, api_key):
        super().__init__(model_name)
        from anthropic import Anthropic
        self.client = Anthropic(api_key=api_key)

    def run(self, candidates_list, grp):
        summary = build_candidate_summary(candidates_list)
        system_message = "You are an expert JSON evaluator for hiring decisions. Output valid JSON only."
        user_message = (
            f"There are {len(summary)} candidates:\n"
            f"{json.dumps(summary)}\n\n"
            "Select exactly TWO candidates and output ONLY valid JSON:\n"
            '{"selected":["Name1","Name2"],"reasons":["Reason1","Reason2"]}'
        )
        last_text = None
        for attempt in range(1, 4):
            try:
                resp = self.client.messages.create(
                    model=self.model_name,
                    system=system_message,
                    messages=[{"role": "user", "content": user_message}],
                    max_tokens=512,
                    temperature=0
                )
                last_text = resp.content[0].text
                clean = clean_json_block(last_text)
                return json.loads(clean)
            except json.JSONDecodeError as e:
                print(f"[Attempt {attempt}] JSON decode error: {e}. Raw={last_text!r}")
                time.sleep(2 ** attempt)
            except Exception as e:
                print(f"[Attempt {attempt}] Claude API error: {e}")
                time.sleep(2 ** attempt)
        return {"error": "Claude retry limit exceeded", "details": (last_text[:500] if last_text else None)}

# ==============================
# OPENAI STRATEGY
# ==============================

class OpenAIStrategy(ModelStrategy):
    def __init__(self, model_name, api_key):
        super().__init__(model_name)
        import openai as _openai
        _openai.api_key = api_key
        self._openai = _openai

    def run(self, candidates_list, grp):
        summary = build_candidate_summary(candidates_list)
        system_msg = {"role": "system", "content": "You are an expert JSON evaluator for hiring decisions. Output valid JSON only."}
        user_msg = {"role": "user", "content": (
            f"There are {len(summary)} candidates:\n"
            f"{json.dumps(summary)}\n\n"
            "Select exactly TWO candidates and output ONLY valid JSON:\n"
            '{"selected":["Name1","Name2"],"reasons":["Reason1","Reason2"]}'
        )}
        last_text = None
        for attempt in range(1, 4):
            try:
                resp = self._openai.chat.completions.create(
                    model=self.model_name,
                    messages=[system_msg, user_msg],
                    temperature=0
                )
                last_text = resp.choices[0].message.content
                clean = clean_json_block(last_text)
                return json.loads(clean)
            except json.JSONDecodeError as e:
                print(f"[Attempt {attempt}] JSON decode error: {e}. Raw={last_text!r}")
                time.sleep(2 ** attempt)
            except Exception as e:
                print(f"[Attempt {attempt}] OpenAI API error: {e}")
                time.sleep(2 ** attempt)
        return {"error": "OpenAI retry limit exceeded", "details": (last_text[:500] if last_text else None)}

# ==============================
# STRATEGY REGISTRY
# ==============================

STRATEGIES = {
    # Gemini
    "gemini-2.0-flash-lite": lambda: GeminiStrategy("google/gemini-2.0-flash-lite-001", OPENROUTER_API_KEY),
    "google/gemini-3-flash-preview": lambda: GeminiStrategy("google/gemini-3-flash-preview", OPENROUTER_API_KEY_2),

    # Llama via Groq (rate limited)
    "llama-3.1-8b-instant": lambda: GroqLlamaStrategy("llama-3.1-8b-instant", GROQ_API_KEY_1),
    "meta-llama/llama-4-maverick-17b-128e-instruct": lambda: GroqLlamaStrategy("meta-llama/llama-4-maverick-17b-128e-instruct", GROQ_API_KEY_2),

    # DeepSeek
    "deepseek-reasoner": lambda: DeepSeekStrategy("deepseek-reasoner", DEEPSEEK_API_KEY),
    "deepseek-chat": lambda: DeepSeekStrategy("deepseek-chat", DEEPSEEK_API_KEY),

    # Claude
    "claude-3-5-haiku-20241022": lambda: ClaudeStrategy("claude-3-5-haiku-20241022", ANTHROPIC_API_KEY),
    "claude-haiku-4-5-20251001": lambda: ClaudeStrategy("claude-haiku-4-5-20251001", ANTHROPIC_API_KEY),

    # OpenAI
    "gpt-4o-mini": lambda: OpenAIStrategy("gpt-4o-mini", OPENAI_API_KEY),
    "gpt-5.2": lambda: OpenAIStrategy("gpt-5.2", OPENAI_API_KEY),
}

def get_strategy(model_name):
    if model_name not in STRATEGIES:
        raise ValueError(f"Unsupported model: {model_name}")
    return STRATEGIES[model_name]()

# ==============================
# BACKGROUND WORKER
# ==============================

def process_and_persist(grp, members, model_name):
    #print(f"[CHECKING FOR SKIP] Group {grp} processed with {model_name}")
    safe_name = model_name.replace("/", "_")
    csv_path = f"{safe_name}-results.csv"

    if group_already_processed(grp, csv_path):
        print(f"[SKIP] Group {grp} already processed for {model_name}")
        return

    strategy = get_strategy(model_name)
    time.sleep(10)
    parsed = strategy.run(members, grp)
    #print(f"[RESULT] Group {grp}, model {model_name}: {parsed}")
    append_result_to_csv(grp, parsed, model_name)
    #print(f"[DONE] Group {grp} processed with {model_name}")

# ==============================
# DATA EXTRACTION
# ==============================

def extract_candidate_data(path):
    df = pd.read_excel(path)
    if "Groups" not in df.columns:
        raise ValueError("Expected 'Groups' column")
    df["Groups"] = df["Groups"].ffill().astype(str).str.strip()
    return df.to_dict(orient="records")

# ==============================
# FLASK APP
# ==============================

app = Flask(__name__)
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

@app.route("/api/v1/candidates", methods=["POST"])
def extract_candidates():
    try:
        #print("raw data:", request.data)
        payload = request.get_json(force=True)
        #print("parsed payload:", payload)
        models = payload.get("models")

        if not models or not isinstance(models, list):
            return jsonify({"error": "Request must include 'models' as a list"}), 400

        invalid = [m for m in models if m not in STRATEGIES]
        if invalid:
            return jsonify({"error": f"Unsupported models: {invalid}"}), 400

        candidates = extract_candidate_data(EXCEL_PATH)

        groups = defaultdict(list)
        for cand in candidates:
            groups[str(cand["Groups"])].append(cand)

        full_groups = {g: m for g, m in groups.items() if len(m) == 5}
        #print(f"full_groups {full_groups}")

        for grp, members in full_groups.items():
            for model in models:
                executor.submit(process_and_persist, grp, members, model)

        return jsonify({
            "status": "accepted",
            "groups_submitted": len(full_groups),
            "models": models
        }), 202

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5316)
