# -*- coding: utf-8 -*-
import uuid
import datetime
import base64
import hmac
import hashlib
import requests
import json
import os
import re
import argparse
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from collections import Counter, defaultdict


API_VERSION = os.environ.get("JUDGE_API_VERSION", "v2.03")
# NOTE: Configure via environment variables before running.
#   JUDGE_API_BASE_URL  - base URL of the GPT-4o judge service
#   JUDGE_APP_ID        - HMAC key id (a.k.a. SecretId)
#   JUDGE_APP_KEY       - HMAC secret (a.k.a. SecretKey)
#   JUDGE_MODEL_MARKER  - upstream model marker, defaults to gpt-4o
BASE_URL = os.environ.get("JUDGE_API_BASE_URL", "")
APP_ID = os.environ.get("JUDGE_APP_ID", "")
APP_KEY = os.environ.get("JUDGE_APP_KEY", "")
MODEL_MARKER = os.environ.get("JUDGE_MODEL_MARKER", "api_openai_gpt-4o")
# Set JUDGE_API_KEY to use a plain OpenAI-compatible endpoint at JUDGE_API_BASE_URL.
JUDGE_API_KEY = os.environ.get("JUDGE_API_KEY", "")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gpt-4o")


# ======================== Judge Prompts ========================

JUDGE_PROMPT = (
    "You are an impartial judge evaluating whether a deep research report contains the correct answer.\n\n"
    "[Question]\n{question}\n\n"
    "[Correct Answer]\n{correct_answer}\n\n"
    "[Deep Research Report]\n{response}\n\n"
    "Task: Determine if the deep research report contains the correct answer anywhere in its content.\n\n"
    "Instructions:\n"
    "1. Read through the entire research report carefully\n"
    "2. Look for the correct answer anywhere in the report (it may be embedded in paragraphs, tables, or sections)\n"
    "3. Check if the information in the report is consistent with the correct answer\n"
    '4. The answer does NOT need to be in a specific format or labeled as "final answer"\n'
    "5. Provide your reasoning\n"
    '6. Answer with "yes" if the report contains the correct answer, "no" if it doesn\'t or contradicts it\n\n'
    "Output format:\n"
    "correct: [yes/no]\n"
    "reasoning: [your explanation]"
)


# ======================== GPT-4o API Wrapper ========================

def get_simple_auth(source: str, secret_id: str, secret_key: str):
    date_time = datetime.datetime.utcnow().strftime("%a, %d %b %Y %H:%M:%S GMT")
    auth = 'hmac id="' + secret_id + '", algorithm="hmac-sha1", headers="date source", signature="'
    sign_str = "date: " + date_time + "\n" + "source: " + source
    sign = hmac.new(secret_key.encode(), sign_str.encode(), hashlib.sha1).digest()
    sign = base64.b64encode(sign).decode()
    return auth + sign + '"', date_time


def _call_openai_compatible(prompt: str, max_retries: int, timeout: int) -> str:
    """Judge via a plain OpenAI-compatible endpoint (JUDGE_API_KEY set)."""
    body = {
        "model": JUDGE_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 1024,
        **json.loads(os.environ.get("LLM_EXTRA_BODY_JSON", "") or "{}"),
    }
    for attempt in range(max_retries):
        try:
            r = requests.post(
                f"{BASE_URL.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {JUDGE_API_KEY}"},
                json=body,
                timeout=timeout,
            )
            if r.status_code == 200:
                return r.json()["choices"][0]["message"].get("content") or ""
            print(f"  [Retry {attempt+1}] HTTP {r.status_code}: {r.text[:200]}")
        except Exception as e:
            print(f"  [Retry {attempt+1}] Error: {e}")
        time.sleep(2 ** attempt)
    return ""


def call_gpt4o(prompt: str, max_retries: int = 5, timeout: int = 120) -> str:
    if BASE_URL and JUDGE_API_KEY:
        return _call_openai_compatible(prompt, max_retries, timeout)
    if not (BASE_URL and APP_ID and APP_KEY):
        raise RuntimeError(
            "Judge API is not configured. Please set JUDGE_API_BASE_URL, "
            "JUDGE_APP_ID and JUDGE_APP_KEY environment variables."
        )
    for attempt in range(max_retries):
        try:
            sign, date_time = get_simple_auth("gpt-54-eval", APP_ID, APP_KEY)
            headers = {
                "Apiversion": API_VERSION,
                "Authorization": sign,
                "Date": date_time,
                "Source": "gpt-54-eval",
                "Content-Type": "application/json",
            }
            body = {
                "request_id": str(uuid.uuid4()),
                "model_marker": MODEL_MARKER,
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "value": prompt}],
                    }
                ],
                "system": "",
                "params": {"stream": False, "temperature": 0.0},
                "timeout": timeout,
            }
            r = requests.post(
                f"{BASE_URL}/api/v1/data_eval",
                headers=headers,
                json=body,
                timeout=timeout + 30,
            )
            if r.status_code == 200:
                resp = r.json()
                # GPT-54 format: {"code":0, "answer":[{"type":"text","value":"..."}], ...}
                if "answer" in resp and isinstance(resp["answer"], list) and len(resp["answer"]) > 0:
                    return resp["answer"][0].get("value", "")
                # OpenAI-compatible format
                if "choices" in resp and len(resp["choices"]) > 0:
                    return resp["choices"][0].get("message", {}).get("content", "")
                if "data" in resp and "choices" in resp["data"]:
                    choices = resp["data"]["choices"]
                    if len(choices) > 0:
                        msg = choices[0].get("message", {})
                        return msg.get("content", "")
                return json.dumps(resp)
            else:
                print(f"  [Retry {attempt+1}] HTTP {r.status_code}: {r.text[:200]}")
                time.sleep(2 ** attempt)
        except Exception as e:
            print(f"  [Retry {attempt+1}] Error: {e}")
            time.sleep(2 ** attempt)
    return ""


# ======================== Answer Extraction ========================

def extract_boxed_content(text: str) -> str:
    """Extract content from \\boxed{...} handling nested braces."""
    idx = text.find("\\boxed{")
    if idx == -1:
        return None
    start = idx + len("\\boxed{")
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    content = text[start:i-1].strip()
    content = re.sub(r'\\text\{([^}]*)\}', r'\1', content)
    content = content.replace("\\#", "#").replace("\\%", "%")
    return content


def extract_final_answer(text: str) -> str:
    """Extract the final answer from trajectory text."""
    if not text:
        return ""

    boxed = extract_boxed_content(text)
    if boxed:
        return boxed

    answer_tag = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if answer_tag:
        return answer_tag.group(1).strip()

    response_tag = re.search(r'<response>(.*?)</response>', text, re.DOTALL)
    if response_tag:
        return response_tag.group(1).strip()

    parts = text.split("</think>")
    if len(parts) > 1:
        last_part = parts[-1].strip()
        last_part = re.sub(r'<tool_call>.*?</tool_call>', '', last_part, flags=re.DOTALL).strip()
        if last_part:
            return last_part[:2000]

    return text[-2000:] if len(text) > 2000 else text


# ======================== Judge Functions ========================

def parse_judge_response(raw: str) -> dict:
    """Parse 'correct: [yes/no]' and 'reasoning: ...' from judge output."""
    acc = 0
    reasoning = ""
    correct_match = re.search(r'correct:\s*(yes|no)', raw, re.IGNORECASE)
    if correct_match:
        acc = 1 if correct_match.group(1).lower() == "yes" else 0
    reasoning_match = re.search(r'reasoning:\s*(.+)', raw, re.IGNORECASE | re.DOTALL)
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()
    return {"acc": acc, "reasoning": reasoning, "raw_judge": raw}


def judge_common(question: str, correct_answer: str, response: str) -> dict:
    """Unified judge using Vision-DeepResearch style prompt."""
    prompt = JUDGE_PROMPT.format(
        question=question,
        correct_answer=correct_answer,
        response=response[:8000],
    )
    raw = call_gpt4o(prompt)
    parsed = parse_judge_response(raw)
    # An unanswered judge call must not silently count as a wrong answer.
    parsed["judge_failed"] = not re.search(r"correct:\s*(yes|no)", raw or "", re.IGNORECASE)
    return parsed


# ======================== Load & Process ========================

def load_trajectory(filepath: str) -> dict:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


_answer_map: dict = {}


def load_answer_map(parquet_path: str) -> dict:
    """Load id->answer mapping from a parquet file (for VDR etc.)."""
    import pandas as pd
    df = pd.read_parquet(parquet_path)
    mapping = {}
    for _, row in df.iterrows():
        mapping[str(row.get("id", ""))] = str(row.get("answer", ""))
    print(f"Loaded {len(mapping)} answers from {parquet_path}")
    return mapping


def process_single(traj_path: str, benchmark: str) -> dict:
    """Process a single trajectory file and return the judge result."""
    traj = load_trajectory(traj_path)
    case_id = traj.get("case_id", Path(traj_path).stem)
    original = traj.get("original_data", {})
    final_text = traj.get("final_response_text", "")

    prompt_msgs = traj.get("prompt", [])
    question = ""
    for msg in prompt_msgs:
        if msg.get("role") == "user":
            question = msg.get("content", "")
            break
    if not question:
        question = original.get("question", "")

    answers = original.get("answers", [])
    if answers:
        if benchmark == "hle":
            if isinstance(answers[0], list):
                correct_answer = answers[0][0] if answers[0] else ""
            else:
                correct_answer = answers[0]
        else:
            correct_answer = ", ".join(answers) if isinstance(answers, list) else str(answers)
    elif _answer_map:
        correct_answer = _answer_map.get(case_id, "")
    else:
        correct_answer = ""

    response = extract_final_answer(final_text)

    judge_result = judge_common(question, correct_answer, response)

    return {
        "case_id": case_id,
        "question": question[:500],
        "correct_answer": correct_answer,
        "model_response": response[:500],
        "benchmark": benchmark,
        **judge_result,
    }


def run_eval(traj_dir: str, benchmark: str, max_workers: int = 4,
             output_path: str = None, limit: int = 0,
             answer_file: str = None):
    global _answer_map
    if answer_file:
        _answer_map = load_answer_map(answer_file)

    traj_files = sorted(Path(traj_dir).glob("*_trajectory.json"))
    if not traj_files:
        print(f"[ERROR] No trajectory files found in {traj_dir}")
        return

    if limit > 0:
        traj_files = traj_files[:limit]

    print(f"{'='*80}")
    print(f"Benchmark: {benchmark}")
    print(f"Trajectory dir: {traj_dir}")
    print(f"Files to evaluate: {len(traj_files)}")
    print(f"Max workers: {max_workers}")
    print(f"{'='*80}\n")

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(process_single, str(f), benchmark): f
            for f in traj_files
        }
        for future in tqdm(as_completed(future_map), total=len(traj_files), desc=f"Eval {benchmark}"):
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as e:
                fname = future_map[future]
                print(f"  [ERROR] {fname}: {e}")

    acc_list = [r["acc"] for r in results]
    total = len(results)
    correct = sum(acc_list)
    accuracy = correct / total * 100 if total > 0 else 0

    report = {
        "benchmark": benchmark,
        "traj_dir": traj_dir,
        "total": total,
        "correct": correct,
        "accuracy": f"{accuracy:.2f}%",
        "judge_model": JUDGE_MODEL if JUDGE_API_KEY else "gpt-4o",
        "judge_failed": sum(1 for r in results if r.get("judge_failed")),
    }

    print(f"\n{'='*80}")
    print(f"Results for {benchmark}:")
    print(f"  Total: {total}")
    print(f"  Correct: {correct}")
    print(f"  Accuracy: {accuracy:.2f}%")
    print(f"{'='*80}\n")

    if output_path is None:
        output_path = os.path.join(traj_dir, f"eval_report_{benchmark}.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Report saved to: {output_path}")

    details_path = output_path.replace(".json", "_details.jsonl")
    with open(details_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Details saved to: {details_path}")

    return report


# ======================== Main ========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trajectories with GPT-4o judge")
    parser.add_argument("--traj_dir", type=str, required=True,
                        help="Directory containing trajectory JSON files")
    parser.add_argument("--benchmark", type=str, required=True,
                        choices=["hle", "bc_vl", "vdr"],
                        help="Benchmark type: hle, bc_vl, or vdr")
    parser.add_argument("--max_workers", type=int, default=20,
                        help="Number of parallel workers")
    parser.add_argument("--output", type=str, default=None,
                        help="Output report path (default: <traj_dir>/eval_report_<benchmark>.json)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit number of files to evaluate (0 = all)")
    parser.add_argument("--answer_file", type=str, default=None,
                        help="Parquet file with id/answer columns (for VDR etc.)")
    args = parser.parse_args()

    run_eval(
        traj_dir=args.traj_dir,
        benchmark=args.benchmark,
        max_workers=args.max_workers,
        output_path=args.output,
        limit=args.limit,
        answer_file=args.answer_file,
    )
