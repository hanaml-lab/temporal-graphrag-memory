"""A cue-based modality tagger: the CPU reference for the learned tagger in the paper.

The paper fine-tunes DeBERTa-v3-large on annotated assertions in context; this tagger reproduces its label set
with lexical cues, so that the toy setting and the tests run without model weights.
"""

from __future__ import annotations

import re

from .types import Modality

CUES: list[tuple[Modality, re.Pattern[str]]] = [
    (Modality.REPORTED, re.compile(r"\b(my|his|her|our|their) \w+ (says|said|thinks|reckons|told me|mentioned)\b|\b(he|she|they) (says|said)\b")),
    (Modality.RETRACTION, re.compile(r"\b(no longer|not any ?more|any ?more|quit|left (my|the|our)|moved out|never (lived|worked|was))\b")),
    (Modality.PROSPECTIVE, re.compile(r"\b(might|may|maybe|thinking (about|of)|planning|plan to|going to|gonna|will|would|considering|if i|hoping to|want to|'d)\b")),
    (Modality.RETROSPECTIVE, re.compile(r"\b(back when|used to|previously|formerly|when i was|years ago|at the time)\b")),
]


def tag_modality(text: str) -> Modality:
    lowered = text.lower()
    for modality, pattern in CUES:
        if pattern.search(lowered):
            return modality
    return Modality.CURRENT
