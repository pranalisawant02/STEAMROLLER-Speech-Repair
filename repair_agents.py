import re
from dataclasses import dataclass
from typing import List, Optional


class BaseRepairAgent:
    def repair_text(self, text: str, prompt: Optional[str] = None) -> str:
        raise NotImplementedError


def normalize_sentence(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip(" ,")
    text = re.sub(r"\s+([,.!?])", r"\1", text)
    text = re.sub(r"([,.!?])(?=\S)", r"\1 ", text)
    text = re.sub(r"\bi\b", "I", text, flags=re.IGNORECASE)
    if text:
        text = text[0].upper() + text[1:]
    if text and not text.endswith((".", "!", "?")):
        text += "."
    return text


class MeaningPreservationAgent(BaseRepairAgent):
    def repair_text(self, text: str, prompt: Optional[str] = None) -> str:
        text = re.sub(r"\b(uh|um|ah|er|hmm|you know)\b[,.\s]*", " ", text or "", flags=re.IGNORECASE)
        return normalize_sentence(text)


class GrammarOptimizedAgent(BaseRepairAgent):
    def repair_text(self, text: str, prompt: Optional[str] = None) -> str:
        text = re.sub(r"\b(can not)\b", "cannot", text or "", flags=re.IGNORECASE)
        text = re.sub(r"\b(dont)\b", "don't", text, flags=re.IGNORECASE)
        text = re.sub(r"\b(cant)\b", "can't", text, flags=re.IGNORECASE)
        return normalize_sentence(text)


class ConciseFluentAgent(BaseRepairAgent):
    def repair_text(self, text: str, prompt: Optional[str] = None) -> str:
        text = text or ""
        text = re.sub(r"\b(uh|um|ah|er|hmm|you know|basically|actually|like)\b", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"\b(\w+)(?:\s+\1\b)+", r"\1", text, flags=re.IGNORECASE)
        text = re.sub(r"([a-zA-Z])\1{2,}", r"\1", text)
        return normalize_sentence(text)


class SequenceOptimizedAgent(BaseRepairAgent):
    def repair_text(self, text: str, prompt: Optional[str] = None) -> str:
        text = text or ""
        # Remove short restart fragments like "I-I-I" or "w-w-want" that may survive ASR.
        text = re.sub(r"\b([A-Za-z]{1,3}-)+(?=[A-Za-z])", "", text)
        text = re.sub(r"\b(\w+)(?:[,.\s]+\1\b)+", r"\1", text, flags=re.IGNORECASE)
        return normalize_sentence(text)


@dataclass
class RepairCandidate:
    agent_name: str
    repaired_text: str
    confidence: float
    explanation: str


@dataclass
class RepairVerdict:
    chosen_candidate: RepairCandidate
    all_candidates: List[RepairCandidate]
    reasoning: str
    final_text: str


def _score_candidate(original_text: str, repaired_text: str) -> float:
    if not repaired_text:
        return 0.0
    words = repaired_text.lower().split()
    repeated = sum(1 for i in range(1, len(words)) if words[i] == words[i - 1])
    punctuation = 1.0 if repaired_text.endswith((".", "!", "?")) else 0.7
    length_ratio = min(len(repaired_text) / max(len(original_text), 1), 1.0)
    return max(0.45, min(0.98, 0.55 + 0.25 * punctuation + 0.15 * length_ratio - 0.08 * repeated))


def run_multi_agent_repair(pre_cleaned_text: str, original_text: Optional[str] = None, context: Optional[str] = None) -> RepairVerdict:
    agents = [MeaningPreservationAgent(), GrammarOptimizedAgent(), ConciseFluentAgent(), SequenceOptimizedAgent()]
    candidates: List[RepairCandidate] = []
    for agent in agents:
        repaired = agent.repair_text(pre_cleaned_text, context)
        candidates.append(RepairCandidate(agent.__class__.__name__, repaired, _score_candidate(pre_cleaned_text, repaired), "Local fast rule-based repair."))
    chosen = max(candidates, key=lambda c: c.confidence)
    return RepairVerdict(chosen, candidates, f"Selected {chosen.agent_name} for the final fluent output.", chosen.repaired_text)
