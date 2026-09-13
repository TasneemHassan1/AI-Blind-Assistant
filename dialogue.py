"""Grounded dialogue with deterministic answers for factual scene questions."""

from __future__ import annotations

import re
import time
from typing import Any

from config import LLM_MEMORY_CONTEXT_LIMIT, LLM_VISIBLE_CONTEXT_LIMIT
from debug import debug_throttled, logger
from llm import BaseLLM
from memory import SemanticMemory
from ollama_llm import OllamaLLM
from scene_graph import SceneGraph
from scene_understanding import SceneUnderstanding
from utils import format_distance_cm


class DialogueManager:
    """Answer factual questions locally; reserve Ollama for open-ended wording.

    The public constructor and :meth:`answer` API are unchanged.  Deterministic
    routing prevents an LLM from inventing live objects for common questions.
    """

    # Phrases that clearly ask about the past. A question containing one of
    # these must never be short-circuited by the live-perception check
    # (which used to fire on the word "see" alone and answer "I currently
    # don't see..." even for "did you see my laptop before"), and is instead
    # routed straight to memory: it's both correct and far faster than
    # waiting on the local LLM.
    _MEMORY_SIGNAL_PHRASES = (
        "did you see", "have you seen", "did i have", "was there",
        "before", "previously", "earlier", "used to",
    )

    def __init__(
        self,
        scene: SceneGraph,
        memory: SemanticMemory,
        understanding: SceneUnderstanding,
        llm: BaseLLM | None = None,
    ) -> None:
        self._scene = scene
        self._memory = memory
        self._understanding = understanding
        self._llm = llm or OllamaLLM()

    def answer(self, question: str | None) -> str:
        if not question:
            return "I didn't hear your question."

        # A short grace window over the live scene absorbs single-frame
        # tracking flicker (e.g. a voice question happening to be answered
        # on exactly the one frame where nothing was confirmed yet) without
        # masking a scene that is genuinely empty for more than an instant.
        scene_snapshot = self._scene.snapshot_for_dialogue()
        # Dialogue must not itself advance the temporal room smoother; only
        # processed camera frames are allowed to change the active room.
        room = self._understanding.current_room()
        scene_objects = [self._scene_context(obj) for obj in scene_snapshot.values()]
        scene_objects.sort(key=lambda obj: (not bool(obj["walking_path"]), obj["name"] or ""))
        scene_objects = scene_objects[:LLM_VISIBLE_CONTEXT_LIMIT]

        deterministic = self._answer_factually(question, scene_objects, room)
        if deterministic is not None:
            return deterministic

        # Temporal wording means the user explicitly asks about the current
        # scene. Never consult permanent memory for these questions.
        if self._asks_about_now(question):
            return self._answer_current_only(question, scene_objects)

        # When there is no live perception, an open-ended visual question must
        # never be delegated to a model that could invent a scene. Questions
        # that are actually about the past (see _MEMORY_SIGNAL_PHRASES) must
        # skip this and go to memory instead.
        if (
            not scene_objects
            and self._is_live_perception_question(question)
            and not self._is_memory_question(question)
        ):
            return "I currently don't see any recognized object."

        memory_objects = self._select_memory_context(question, room.get("estimated_room"), scene_objects)
        scene = {
            "status": "no recognized objects" if not scene_objects else "recognized objects listed below",
            "room": room.get("estimated_room") or "unknown",
            "room_confidence": room.get("confidence"),
            "objects": scene_objects,
        }
        memory = {"objects": memory_objects, "status": "no matching memory" if not memory_objects else "records listed below"}
        try:
            started = time.perf_counter()
            answer = self._llm.generate_response(question=question, scene=scene, memory=memory)
            debug_throttled(
                "dialogue_timing",
                seconds=round(time.perf_counter() - started, 3),
                scene_objects=len(scene_objects),
                memory_records=len(memory_objects),
            )
            return answer
        except Exception as error:
            logger.exception("Dialogue generation failed for question %r: %s", question, error)
            return "Sorry, I couldn't answer that."

    def _answer_factually(
        self, question: str, objects: list[dict[str, Any]], room: dict[str, Any]
    ) -> str | None:
        normalized = question.casefold().strip()
        # Explicit past-tense/memory questions are routed straight to memory,
        # before any live-scene pattern gets a chance to misfire on them
        # (e.g. "did you see my laptop before" contains "see", which used to
        # match the live-scene/existence patterns below and answer as if the
        # user were asking about right now).
        if self._is_memory_question(normalized):
            return self._answer_memory_location(normalized)
        if self._is_live_scene_question(normalized):
            return self._describe_visible(objects)
        if "describe" in normalized and "room" in normalized:
            return self._describe_room(objects, room)
        if any(phrase in normalized for phrase in ("where am i", "what room", "which room")):
            return self._where_am_i(room)
        if "how many" in normalized:
            return self._count_objects(normalized, objects)
        if any(phrase in normalized for phrase in ("do you see", "is there", "are there", "is anyone", "do i have")):
            return self._answer_existence(normalized, objects)
        if "next to" in normalized or "near " in normalized:
            return self._answer_relationship(normalized, objects)
        if any(phrase in normalized for phrase in ("what is ahead", "what's ahead", "what is in front", "what's in front")):
            return self._answer_navigation(objects)
        if "where did you last" in normalized or "last see" in normalized:
            return self._answer_memory_location(normalized)
        if normalized.startswith(("where is", "where's")):
            live_matches = self._matching_objects(normalized, objects)
            if live_matches:
                return self._location_sentence(live_matches[0], visible=True)
            if self._asks_about_now(normalized):
                return self._answer_current_location_with_memory(normalized)
            return self._answer_memory_location(normalized)
        return None

    @staticmethod
    def _is_memory_question(question: str) -> bool:
        normalized = question.casefold()
        return any(phrase in normalized for phrase in DialogueManager._MEMORY_SIGNAL_PHRASES)

    @staticmethod
    def _is_live_scene_question(question: str) -> bool:
        return any(phrase in question for phrase in (
            "what do you see", "what can you see", "what do i see", "what is visible",
            "what's visible", "what is around", "what's around", "look around",
            "describe the scene", "describe scene", "describe my surroundings",
            "describe surroundings", "what is around me", "what's around me",
            "what objects are visible",
        ))

    @staticmethod
    def _asks_about_now(question: str) -> bool:
        normalized = question.casefold()
        return (
            "right now" in normalized
            or "at the moment" in normalized
            or "currently" in normalized
            or re.search(r"\bnow\b", normalized) is not None
        )

    def _answer_current_only(self, question: str, objects: list[dict[str, Any]]) -> str:
        """Answer a temporal question from current objects, never memory."""
        matches = self._matching_objects(question.casefold(), objects)
        if matches:
            return self._location_sentence(matches[0], visible=True)
        if not objects:
            return "I cannot currently see any recognizable objects."
        target = self._question_object_name(question.casefold())
        if target:
            return f"I do not currently see a {target}."
        return self._describe_visible(objects)

    @staticmethod
    def _is_live_perception_question(question: str) -> bool:
        normalized = question.casefold()
        return any(word in normalized for word in ("see", "visible", "ahead", "front", "around", "room", "near", "next to"))

    @staticmethod
    def _describe_visible(objects: list[dict[str, Any]]) -> str:
        if not objects:
            return "I currently don't see any recognized object."
        names = []
        seen = set()
        for item in objects:
            name = str(item.get('name', 'object'))
            if name not in seen:
                names.append(name)
                seen.add(name)
        return f"I currently see {DialogueManager._human_list(names)}."

    def _describe_room(self, objects: list[dict[str, Any]], room: dict[str, Any]) -> str:
        estimated = room.get("estimated_room")
        
        # Room part
        if estimated:
            room_part = f"You appear to be in a {estimated}."
        else:
            room_part = "I can't confidently determine the room yet."

        # Objects part
        if not objects:
            return room_part + " I currently don't see any recognized object."

        names = []
        seen = set()
        for item in objects:
            name = str(item.get('name', 'object'))
            if name not in seen:
                names.append(name)
                seen.add(name)

        objects_part = f"I currently see {DialogueManager._human_list(names)}."
        return room_part + " " + objects_part

    @staticmethod
    def _where_am_i(room: dict[str, Any]) -> str:
        estimated = room.get("estimated_room")
        return f"You appear to be in a {estimated}." if estimated else "I can't confidently determine the room yet."

    def _count_objects(self, question: str, objects: list[dict[str, Any]]) -> str:
        matches = self._matching_objects(question, objects)
        target = self._question_object_name(question)
        count = len(matches) if target else len(objects)
        label = target or "recognized object"
        plural = label if count == 1 else (label if label.endswith("s") else label + "s")
        return f"I currently see {count} {plural}."

    def _answer_existence(self, question: str, objects: list[dict[str, Any]]) -> str:
        if "anyone" in question or "person" in question:
            matches = [item for item in objects if str(item.get("name", "")).casefold() == "person"]
            target = "person"
        else:
            matches = self._matching_objects(question, objects)
            target = self._question_object_name(question)
        if "ahead" in question or "front" in question:
            matches = [item for item in matches if item.get("position") == "ahead"]
        if matches:
            return f"Yes. I currently see {self._human_list([str(item['name']) for item in matches])}."
        if target:
            return f"No. I do not currently see a {target}."
        return "No. I do not currently see any recognized object."

    def _answer_relationship(self, question: str, objects: list[dict[str, Any]]) -> str:
        matches = self._matching_objects(question, objects)
        if not matches:
            return "I don't know."
        item = matches[0]
        nearby = item.get("nearby_objects") or []
        if nearby:
            return f"{self._human_list([str(value) for value in nearby])} is near the {item['name']}."
        return f"I don't see a recognized object next to the {item['name']}."

    @staticmethod
    def _answer_navigation(objects: list[dict[str, Any]]) -> str:
        blocking = [
            item for item in objects
            if item.get("position") == "ahead" or item.get("walking_path")
        ]
        if not blocking:
            return "I don't currently see a recognized object ahead."
        if len(blocking) == 1:
            return f"Ahead of you: {DialogueManager._location_phrase(blocking[0])}."
        names = [str(item["name"]) for item in blocking]
        return f"Ahead of you: {DialogueManager._human_list(names)}."
    
    def _answer_memory_location(self, question: str) -> str:
        """Describe where an object was last seen using stable anchors.

        A relative direction ("to your left/ahead") is only meaningful for
        where the user was standing and facing at that past moment — by the
        time they ask, they have very likely moved, so repeating it back is
        actively misleading. Room + support/nearby object are stable facts
        that stay true regardless of where the user is standing now.
        """
        target = self._question_object_name(question)
        if not target:
            return "I don't know."
        record = self._memory.find_closest(target)
        if not record:
            return "I don't know."
        name = str(record.get("name", target))
        room = record.get("estimated_room")
        support = record.get("support_object")
        nearby = record.get("nearby_objects") or []
        description = f"I last saw the {name}"
        if room and str(room).casefold() != "unknown":
            description += f" in the {room}"
        if support:
            description += f", on the {support}"
        elif nearby:
            description += f", near the {nearby[0]}"
        return description + "."

    def _answer_current_location_with_memory(self, question: str) -> str:
        """Report a missing live object and then its last known location."""
        target = self._question_object_name(question)
        if not target:
            return "I cannot currently see any recognizable objects."
        current = f"I do not currently see a {target}."
        last_known = self._answer_memory_location(question)
        return current if last_known == "I don't know." else f"{current} {last_known}"

    @staticmethod
    def _location_sentence(item: dict[str, Any], visible: bool) -> str:
        prefix = "I currently see" if visible else "I last saw"
        return f"{prefix} {DialogueManager._location_phrase(item)}."

    @staticmethod
    def _position_phrase(position: str) -> str:
        """Phrase a relative position naturally ('ahead of you', not 'to your ahead')."""
        if position == "ahead":
            return "ahead of you"
        if position in ("left", "right"):
            return f"to your {position}"
        return position

    @staticmethod
    def _location_phrase(item: dict[str, Any]) -> str:
        name = str(item.get("name", "object"))
        support = item.get("support_object")
        position = item.get("position")
        distance_phrase = format_distance_cm(item.get("distance_cm"))
        phrase = f"a {name}"
        if support:
            phrase += f" on the {support}"
        if position:
            phrase += f" {DialogueManager._position_phrase(position)}"
        if distance_phrase:
            phrase += f", {distance_phrase} away"
        return phrase

    def _matching_objects(self, question: str, objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
        words = self._keywords(question)
        matches = []
        for item in objects:
            name = str(item.get("name", "")).casefold()
            name_words = set(name.split())
            if name in question or name_words & words or name.rstrip("s") in words:
                matches.append(item)
        return matches

    @staticmethod
    def _question_object_name(question: str) -> str | None:
        ignored = {"where", "what", "when", "which", "with", "that", "this", "your", "have", "last", "seen", "does", "from", "about", "there", "any", "the", "and", "for", "are", "is", "my", "how", "many", "you", "i", "did", "do", "currently", "now", "right", "moment", "can", "near", "next", "to", "in", "on", "ahead", "front", "room", "before", "previously", "earlier", "used"}
        words = [word.rstrip("s") for word in re.findall(r"[\w']+", question.casefold())]
        candidates = [word for word in words if len(word) > 1 and word not in ignored]
        return candidates[-1] if candidates else None

    @staticmethod
    def _human_list(values: list[str]) -> str:
        unique = list(dict.fromkeys(values))
        if not unique:
            return "no recognized objects"
        if len(unique) == 1:
            return f"a {unique[0]}"
        if len(unique) == 2:
            return f"a {unique[0]} and a {unique[1]}"
        return ", ".join(f"a {value}" for value in unique[:-1]) + f", and a {unique[-1]}"

    @staticmethod
    def _join_phrases(values: list[str]) -> str:
        """Join phrases that already include their own articles."""
        if len(values) == 1:
            return values[0]
        if len(values) == 2:
            return f"{values[0]} and {values[1]}"
        return ", ".join(values[:-1]) + f", and {values[-1]}"

    @staticmethod
    def _scene_context(obj: dict[str, Any]) -> dict[str, Any]:
        return {
            "name": obj.get("name"), "position": obj.get("position"),
            "distance": obj.get("distance_meters"), "distance_cm": obj.get("distance_cm"),
            "depth": obj.get("depth"), "support_object": obj.get("support_object"),
            "nearby_objects": obj.get("near_objects", []), "relationships": obj.get("relationships", []),
            "motion_state": obj.get("motion_state"), "walking_path": obj.get("in_walking_path"),
            "crossing_path": obj.get("crossing_path"), "confidence": round(float(obj.get("confidence", 0.0)), 2),
        }

    def _select_memory_context(self, question: str, current_room: str | None, scene_objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
        keywords = self._keywords(question)
        visible_names = {str(item.get("name", "")).casefold() for item in scene_objects}
        ranked: list[tuple[int, str, dict[str, Any]]] = []
        for record in self._memory.snapshot().values():
            name = str(record.get("name", ""))
            aliases = [str(alias) for alias in record.get("aliases", [])]
            labels = {name.casefold(), *(alias.casefold() for alias in aliases)}
            score = (100 if labels & keywords else 0) + (60 if name.casefold() in visible_names else 0)
            score += 30 if current_room and record.get("estimated_room") == current_room else 0
            ranked.append((score, str(record.get("last_seen") or ""), record))
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [self._memory_context(record) for _, _, record in ranked[:LLM_MEMORY_CONTEXT_LIMIT]]

    @staticmethod
    def _keywords(question: str) -> set[str]:
        return {word.rstrip("s") for word in re.findall(r"[\w']+", question.casefold()) if len(word) > 1}

    @staticmethod
    def _memory_context(record: dict[str, Any]) -> dict[str, Any]:
        return {"name": record.get("name"), "aliases": record.get("aliases", []), "room": record.get("estimated_room"), "position": record.get("position"), "support_object": record.get("support_object"), "nearby_objects": record.get("nearby_objects", []), "last_seen": record.get("last_seen")}