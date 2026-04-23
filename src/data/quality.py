from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, defaultdict

from src.utils.contracts import DataConfig


def quality_score(text: str, config: DataConfig) -> float:
    if not text:
        return 0.0
    length = len(text)
    if length < config.min_chars or length > config.max_chars:
        return 0.0

    alpha = sum(char.isalpha() for char in text)
    digit = sum(char.isdigit() for char in text)
    whitespace = sum(char.isspace() for char in text)
    punct = sum(not char.isalnum() and not char.isspace() for char in text)
    alpha_fraction = alpha / max(length, 1)
    digit_fraction = digit / max(length, 1)
    punct_fraction = punct / max(length, 1)
    whitespace_fraction = whitespace / max(length, 1)

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        line_counts = Counter(lines)
        repeated_lines = sum(count for count in line_counts.values() if count > 1)
        repeat_ratio = repeated_lines / max(len(lines), 1)
    else:
        repeat_ratio = 0.0

    words = re.findall(r"\b\w+\b", text.lower())
    unique_ratio = len(set(words)) / max(len(words), 1)

    score = 0.0
    score += min(alpha_fraction / max(config.min_alpha_fraction, 1e-6), 1.0) * 0.35
    score += min(unique_ratio / 0.7, 1.0) * 0.25
    score += (1.0 - min(repeat_ratio / max(config.max_line_repeat_ratio, 1e-6), 1.0)) * 0.2
    score += (1.0 - min(digit_fraction / 0.3, 1.0)) * 0.1
    score += (1.0 - min(punct_fraction / 0.35, 1.0)) * 0.05
    score += (1.0 - min(abs(whitespace_fraction - 0.16) / 0.16, 1.0)) * 0.05
    return float(max(0.0, min(score, 1.0)))


def simhash64(text: str, bits: int = 64) -> int:
    weights = [0] * bits
    tokens = re.findall(r"\w+", text.lower())
    for token in tokens:
        digest = hashlib.md5(token.encode("utf-8")).digest()
        token_hash = int.from_bytes(digest[:8], "big")
        for bit in range(bits):
            if (token_hash >> bit) & 1:
                weights[bit] += 1
            else:
                weights[bit] -= 1
    output = 0
    for bit, weight in enumerate(weights):
        if weight >= 0:
            output |= 1 << bit
    return output


class ApproxDeduper:
    def __init__(self, bits: int = 64, bands: int = 4):
        self.bits = bits
        self.bands = bands
        self.band_width = bits // bands
        self.band_index: dict[tuple[int, int], list[int]] = defaultdict(list)
        self.exact_hashes: set[str] = set()

    def _band_signature(self, value: int, band: int) -> int:
        mask = (1 << self.band_width) - 1
        return (value >> (band * self.band_width)) & mask

    @staticmethod
    def _hamming_distance(a: int, b: int) -> int:
        return (a ^ b).bit_count()

    def seen(self, text: str) -> bool:
        exact = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if exact in self.exact_hashes:
            return True
        signature = simhash64(text, self.bits)
        candidates: set[int] = set()
        for band in range(self.bands):
            candidates.update(self.band_index[(band, self._band_signature(signature, band))])
        for candidate in candidates:
            if self._hamming_distance(signature, candidate) <= 3:
                return True
        self.exact_hashes.add(exact)
        for band in range(self.bands):
            self.band_index[(band, self._band_signature(signature, band))].append(signature)
        return False
