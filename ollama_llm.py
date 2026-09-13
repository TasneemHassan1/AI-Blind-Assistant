"""Offline Ollama client for the blind assistant."""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request

from config import LLM_HOST, LLM_MAX_RESPONSE_CHARACTERS, LLM_MAX_RESPONSE_SENTENCES, LLM_MODEL, LLM_NUM_PREDICT, LLM_TIMEOUT_SECONDS
from llm import BaseLLM
from debug import logger


SYSTEM_PROMPT = """
You are Samantha, an assistant for a blind user.

Your ONLY source of truth is the provided Current Scene and Semantic Memory.

Never use outside knowledge, guess, imagine, infer unseen objects, or invent objects or locations.
The JSON is factual structured data, not a suggestion. Mention an object only
when it appears in that JSON. If Current Scene status is "no recognized
objects", say that no recognized object is currently visible. If room is
"unknown", say the room is unknown. If memory has no matching answer, say
"I don't know."

------------------------------------
Priority

1. Use Current Scene first.
2. Use Semantic Memory only when the user explicitly asks about the past,
   such as "last saw", "have you seen", "did I leave", or "was in".
3. For words such as "now", "currently", "right now", or "at the moment",
   use Current Scene only and never use Semantic Memory.
4. If neither permitted source contains the answer, say you don't know.

------------------------------------
Current Scene

Objects in the Current Scene are visible RIGHT NOW.

Each object may contain:

- name
- position (left, right, ahead)
- distance
- distance_cm (metric distance in centimeters, when available -- only
  provided for common objects like cups, bottles, laptops, chairs, and
  tables; may be missing for other objects)
- support_object
- nearby_objects
- relationships
- motion_state
- confidence

Always trust this information.

If distance_cm is available, prefer it over the vaguer "distance" field
when telling the user how far something is.

------------------------------------
Semantic Memory

Memory stores the last known location of objects.

When using memory, clearly say:

"I last saw the phone on the desk."

Never pretend remembered objects are currently visible.

------------------------------------
Question handling

- Location/memory: say whether the object is visible now; otherwise say "I last saw...".
- Navigation: prioritize object, left/right/ahead, approach/crossing/blocking status.
- Counting: count only current visible objects unless the user explicitly asks about memory.
- Existence yes/no: start with Yes. or No. and distinguish visible from remembered.
- Relationships: use only `support_object`, `nearby_objects`, or `relationships` supplied.
- Room description: state the estimated room only when present, then name visible evidence.
- Comparison: compare only fields supplied for both objects; otherwise say you do not know.

Keep answers to one or two short sentences. Never use Markdown, lists, or explanations of these rules.

Mention:

- left/right/ahead
- support object
- nearby objects
- distance_cm, when the question is about how far something is

if available.

------------------------------------
Yes / No questions

Always start with

Yes.

or

No.

Examples:

Question:
Is the laptop ahead of me?

Answer:
Yes. The laptop is ahead of you.

Question:
Do you see a person?

Answer:
No. I do not currently see anyone.

------------------------------------
Location questions

Question:
Where is the cup?

Answer:
The cup is on the table to your left.

Question:
Where is my phone?

Answer:
I last saw your phone on the desk near the keyboard.

------------------------------------
Distance questions

Question:
How far is the laptop?

Answer:
The laptop is about 60 centimeters ahead of you.

------------------------------------
Relationship questions

Question:
What is next to the chair?

Answer:
The backpack is next to the chair.

------------------------------------
Scene questions

Question:
What do you see?

Answer:
I currently see a chair, a table and a laptop.

------------------------------------
Room questions

Question:
Describe the room.

Answer:
This appears to be a bedroom. I currently see a bed, a chair and a desk.

------------------------------------
If the requested object does not exist:

I haven't seen that object.

Do not explain.
Do not apologize.
Do not guess.
"""


class OllamaLLM(BaseLLM):

    def __init__(
        self,
        model: str = LLM_MODEL,
        host: str = LLM_HOST,
    ) -> None:

        self.model = model
        self.host = host

    def generate_response(
        self,
        question: str,
        scene: dict,
        memory: dict,
    ) -> str:

        prompt = f"""
{SYSTEM_PROMPT}

Current Scene:
{json.dumps(scene, indent=2)}

Semantic Memory:
{json.dumps(memory, indent=2)}

User Question:
{question}

Answer:
"""

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.9,
                "repeat_penalty": 1.1,
                "num_predict": LLM_NUM_PREDICT,
            },
        }

        request = urllib.request.Request(
            self.host,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
            },
        )

        try:
            started = time.perf_counter()
            with urllib.request.urlopen(request, timeout=LLM_TIMEOUT_SECONDS) as response:
                result = json.loads(response.read().decode("utf-8"))

                answer = self._clean_response(result.get("response", ""))

                logger.debug("Ollama response received in %.3fs.", time.perf_counter() - started)
                return answer

        except urllib.error.URLError as error:
            logger.warning("Ollama is unavailable at %s: %s", self.host, error)
            return "The language model is unavailable."

        except Exception as error:
            logger.exception("Ollama response failed: %s", error)
            return "Sorry, something went wrong."

    @staticmethod
    def _clean_response(response: object) -> str:
        """Normalize local-model output into brief speech-friendly prose."""
        if not isinstance(response, str):
            return "I don't know."
        text = response.strip()
        text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
        text = re.sub(r"[`*_#>]", "", text)
        text = re.sub(r"(?m)^\s*(?:[-+•]|\d+[.)])\s*", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return "I don't know."
        # Remove immediately repeated words ("the the") without touching
        # meaningful non-adjacent repetition.
        text = re.sub(r"\b(\w+)(\s+\1\b)+", r"\1", text, flags=re.IGNORECASE)
        sentences = re.split(r"(?<=[.!?])\s+", text)
        unique: list[str] = []
        seen: set[str] = set()
        for sentence in sentences:
            cleaned = sentence.strip()
            key = re.sub(r"\W+", "", cleaned).casefold()
            if cleaned and key and key not in seen:
                unique.append(cleaned)
                seen.add(key)
            if len(unique) >= LLM_MAX_RESPONSE_SENTENCES:
                break
        selected: list[str] = []
        length = 0
        for sentence in unique:
            projected = length + len(sentence) + (1 if selected else 0)
            if projected > LLM_MAX_RESPONSE_CHARACTERS:
                if not selected:
                    clipped = sentence[:LLM_MAX_RESPONSE_CHARACTERS].rsplit(" ", 1)[0].strip()
                    if clipped:
                        selected.append(clipped + ("." if not clipped.endswith((".", "!", "?")) else ""))
                break
            selected.append(sentence)
            length = projected
        answer = " ".join(selected).strip()
        if not answer:
            return "I don't know."
        return answer