from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from src.data.tokenizer import BPETokenizer


def _jsonl_offsets(path: str | Path) -> list[int]:
    offsets: list[int] = []
    with Path(path).open("rb") as handle:
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            if line.strip():
                offsets.append(offset)
    return offsets


def _read_offset_json(path: str | Path, offset: int) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        handle.seek(offset)
        return json.loads(handle.readline().decode("utf-8"))


def _causal_example(tokens: list[int], seq_len: int, pad_token_id: int) -> dict[str, torch.Tensor]:
    tokens = tokens[: seq_len + 1]
    if len(tokens) < seq_len + 1:
        tokens = tokens + [pad_token_id] * (seq_len + 1 - len(tokens))
    input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
    labels = torch.tensor(tokens[1:], dtype=torch.long)
    attention_mask = (input_ids != pad_token_id).long()
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


class JsonlPackedSequenceDataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        seq_len: int,
        pad_token_id: int,
        offsets: list[int] | None = None,
    ):
        self.path = str(path)
        self.seq_len = seq_len
        self.pad_token_id = pad_token_id
        self.offsets = offsets if offsets is not None else _jsonl_offsets(path)

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = _read_offset_json(self.path, self.offsets[index])
        return _causal_example(list(row["tokens"]), self.seq_len, self.pad_token_id)


class JsonlSFTDataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        tokenizer: BPETokenizer,
        seq_len: int,
        offsets: list[int] | None = None,
    ):
        self.path = str(path)
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.offsets = offsets if offsets is not None else _jsonl_offsets(path)
        self.pad_id = tokenizer.token_to_id[tokenizer.pad_token]

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = _read_offset_json(self.path, self.offsets[index])
        prompt = f"Instruction: {row['prompt']} Response:"
        response = row["response"]
        prompt_ids = self.tokenizer.encode(prompt, add_bos=True, add_eos=False)
        full_ids = prompt_ids + self.tokenizer.encode(response, add_bos=False, add_eos=True)
        full_ids = full_ids[: self.seq_len + 1]
        if len(full_ids) < self.seq_len + 1:
            full_ids = full_ids + [self.pad_id] * (self.seq_len + 1 - len(full_ids))
        input_ids = torch.tensor(full_ids[:-1], dtype=torch.long)
        labels = torch.tensor(full_ids[1:], dtype=torch.long)
        prompt_cutoff = max(0, len(prompt_ids) - 1)
        labels[:prompt_cutoff] = -100
        attention_mask = (input_ids != self.pad_id).long()
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


class JsonlPreferenceDataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        tokenizer: BPETokenizer,
        seq_len: int,
        offsets: list[int] | None = None,
    ):
        self.path = str(path)
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.offsets = offsets if offsets is not None else _jsonl_offsets(path)
        self.pad_id = tokenizer.token_to_id[tokenizer.pad_token]

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = _read_offset_json(self.path, self.offsets[index])
        prompt = f"Instruction: {row['prompt']} Response:"
        prompt_ids = self.tokenizer.encode(prompt, add_bos=True, add_eos=False)
        chosen_ids = (prompt_ids + self.tokenizer.encode(row["chosen"], add_bos=False, add_eos=True))[: self.seq_len + 1]
        rejected_ids = (prompt_ids + self.tokenizer.encode(row["rejected"], add_bos=False, add_eos=True))[: self.seq_len + 1]
        chosen_ids = chosen_ids + [self.pad_id] * (self.seq_len + 1 - len(chosen_ids))
        rejected_ids = rejected_ids + [self.pad_id] * (self.seq_len + 1 - len(rejected_ids))
        return {
            "chosen_input_ids": torch.tensor(chosen_ids[:-1], dtype=torch.long),
            "chosen_labels": torch.tensor(chosen_ids[1:], dtype=torch.long),
            "rejected_input_ids": torch.tensor(rejected_ids[:-1], dtype=torch.long),
            "rejected_labels": torch.tensor(rejected_ids[1:], dtype=torch.long),
        }


class PackedSequenceDataset(Dataset):
    def __init__(self, sequences: list[list[int]], seq_len: int, pad_token_id: int):
        self.seq_len = seq_len
        self.pad_token_id = pad_token_id
        self.examples: list[dict[str, torch.Tensor]] = []
        for tokens in sequences:
            tokens = tokens[: seq_len + 1]
            if len(tokens) < 2:
                continue
            if len(tokens) < seq_len + 1:
                tokens = tokens + [pad_token_id] * (seq_len + 1 - len(tokens))
            input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
            labels = torch.tensor(tokens[1:], dtype=torch.long)
            attention_mask = (input_ids != pad_token_id).long()
            self.examples.append(
                {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.examples[index]


class SFTDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], tokenizer: BPETokenizer, seq_len: int):
        self.examples: list[dict[str, torch.Tensor]] = []
        pad_id = tokenizer.token_to_id[tokenizer.pad_token]
        for row in rows:
            prompt = f"Instruction: {row['prompt']} Response:"
            response = row["response"]
            prompt_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
            full_ids = prompt_ids + tokenizer.encode(response, add_bos=False, add_eos=True)
            full_ids = full_ids[: seq_len + 1]
            if len(full_ids) < seq_len + 1:
                full_ids = full_ids + [pad_id] * (seq_len + 1 - len(full_ids))
            input_ids = torch.tensor(full_ids[:-1], dtype=torch.long)
            labels = torch.tensor(full_ids[1:], dtype=torch.long)
            prompt_cutoff = max(0, len(prompt_ids) - 1)
            labels[:prompt_cutoff] = -100
            attention_mask = (input_ids != pad_id).long()
            self.examples.append(
                {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.examples[index]


class PreferenceDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], tokenizer: BPETokenizer, seq_len: int):
        self.examples: list[dict[str, Any]] = []
        pad_id = tokenizer.token_to_id[tokenizer.pad_token]
        for row in rows:
            prompt = f"Instruction: {row['prompt']} Response:"
            prompt_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
            chosen_ids = (prompt_ids + tokenizer.encode(row["chosen"], add_bos=False, add_eos=True))[: seq_len + 1]
            rejected_ids = (prompt_ids + tokenizer.encode(row["rejected"], add_bos=False, add_eos=True))[: seq_len + 1]
            chosen_ids = chosen_ids + [pad_id] * (seq_len + 1 - len(chosen_ids))
            rejected_ids = rejected_ids + [pad_id] * (seq_len + 1 - len(rejected_ids))
            self.examples.append(
                {
                    "chosen_input_ids": torch.tensor(chosen_ids[:-1], dtype=torch.long),
                    "chosen_labels": torch.tensor(chosen_ids[1:], dtype=torch.long),
                    "rejected_input_ids": torch.tensor(rejected_ids[:-1], dtype=torch.long),
                    "rejected_labels": torch.tensor(rejected_ids[1:], dtype=torch.long),
                }
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.examples[index]
