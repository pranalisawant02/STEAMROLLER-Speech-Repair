"""
nlp_cleaner.py — Pre-processing of raw ASR output before multi-agent repair.
Handles filler word removal, repetition detection, and basic normalisation.
"""

import re
from dataclasses import dataclass
from typing import List, Tuple


# Common stutter/filler patterns
FILLER_WORDS = [
    r"\b(um+|uh+|er+|ah+|eh+|hmm+|mm+)\b",
    r"\b(like,?\s*like)\b",
    r"\b(you\s*know,?\s*you\s*know)\b",
    r"\b(so+|well+)\b(?=\s)",
]

# Detect repeated syllables  e.g. "b-b-because", "to-to-today"
SYLLABLE_REPETITION = re.compile(
    r"\b([a-z]{1,4})-\1(?:-\1)*(?=[a-z])", re.IGNORECASE
)

# Detect word-level repetitions  e.g. "I I I want"
WORD_REPETITION = re.compile(
    r"\b(\w+)(?:\s+\1){1,4}\b", re.IGNORECASE
)

# Detect phrase-level repetitions  e.g. "I want to I want to go"
PHRASE_REPETITION = re.compile(
    r"\b(.{5,40}?)\s+\1\b", re.IGNORECASE
)


@dataclass
class CleaningReport:
    original: str
    cleaned: str
    fillers_removed: int
    repetitions_fixed: int
    syllable_stutters_fixed: int
    issues: List[str]


def _remove_fillers(text: str) -> Tuple[str, int]:
    count = 0
    for pattern in FILLER_WORDS:
        new_text, n = re.subn(pattern, "", text, flags=re.IGNORECASE)
        count += n
        text = new_text
    # clean up extra spaces
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text, count


def _fix_syllable_stutters(text: str) -> Tuple[str, int]:
    """b-b-because → because"""
    count = [0]

    def replacer(m: re.Match) -> str:
        count[0] += 1
        # Return just the full word that follows
        return ""

    # More targeted: match  "b-b-b" at word start and drop it
    pattern = re.compile(r"\b([a-z]{1,4}-)+", re.IGNORECASE)
    result = pattern.sub(replacer, text)
    result = re.sub(r"\s{2,}", " ", result).strip()
    return result, count[0]


def _fix_word_repetitions(text: str) -> Tuple[str, int]:
    """I I I want → I want"""
    new_text, n = re.subn(WORD_REPETITION, r"\1", text)
    return new_text, n


def _fix_phrase_repetitions(text: str) -> Tuple[str, int]:
    """I want to go I want to go → I want to go"""
    new_text, n = re.subn(PHRASE_REPETITION, r"\1", text)
    return new_text, n


def clean_asr_output(raw_text: str) -> CleaningReport:
    """
    Full NLP cleaning pipeline on raw ASR transcription.
    Returns a CleaningReport with cleaned text and stats.
    """
    issues: List[str] = []
    text = raw_text.strip()

    # Step 1 — syllable stutters
    text, syl_fixed = _fix_syllable_stutters(text)
    if syl_fixed:
        issues.append(f"Syllable stutters: {syl_fixed}")

    # Step 2 — word repetitions
    text, word_reps = _fix_word_repetitions(text)
    if word_reps:
        issues.append(f"Word repetitions: {word_reps}")

    # Step 3 — phrase repetitions
    text, phrase_reps = _fix_phrase_repetitions(text)
    if phrase_reps:
        issues.append(f"Phrase repetitions: {phrase_reps}")

    # Step 4 — filler words
    text, fillers = _remove_fillers(text)
    if fillers:
        issues.append(f"Fillers removed: {fillers}")

    # Step 5 — capitalise first letter
    if text:
        text = text[0].upper() + text[1:]

    return CleaningReport(
        original=raw_text,
        cleaned=text,
        fillers_removed=fillers,
        repetitions_fixed=word_reps + phrase_reps,
        syllable_stutters_fixed=syl_fixed,
        issues=issues,
    )
