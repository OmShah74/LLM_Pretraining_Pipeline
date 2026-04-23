from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.normalizers import NFKC, Lowercase, Sequence, StripAccents
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer

from src.utils.io import ensure_dir


SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]


@dataclass
class BPETokenizer:
    tokenizer: Tokenizer
    pad_token: str = "<pad>"
    bos_token: str = "<bos>"
    eos_token: str = "<eos>"
    unk_token: str = "<unk>"

    @classmethod
    def train(
        cls,
        texts: list[str],
        max_vocab_size: int,
        min_frequency: int,
        lowercase: bool = False,
    ) -> "BPETokenizer":
        tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        normalizers = [NFKC(), StripAccents()]
        if lowercase:
            normalizers.append(Lowercase())
        tokenizer.normalizer = Sequence(normalizers)
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        tokenizer.decoder = ByteLevelDecoder()
        trainer = BpeTrainer(
            vocab_size=max_vocab_size,
            min_frequency=min_frequency,
            special_tokens=SPECIAL_TOKENS,
            show_progress=False,
        )
        tokenizer.train_from_iterator(texts, trainer=trainer)
        bos_id = tokenizer.token_to_id("<bos>")
        eos_id = tokenizer.token_to_id("<eos>")
        tokenizer.post_processor = TemplateProcessing(
            single="<bos> $A <eos>",
            pair="<bos> $A <eos> $B:1 <eos>:1",
            special_tokens=[("<bos>", bos_id), ("<eos>", eos_id)],
        )
        return cls(tokenizer=tokenizer)

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = True) -> list[int]:
        encoding = self.tokenizer.encode(text)
        token_ids = list(encoding.ids)
        bos_id = self.token_to_id[self.bos_token]
        eos_id = self.token_to_id[self.eos_token]
        if not add_bos and token_ids and token_ids[0] == bos_id:
            token_ids = token_ids[1:]
        if not add_eos and token_ids and token_ids[-1] == eos_id:
            token_ids = token_ids[:-1]
        if add_bos and (not token_ids or token_ids[0] != bos_id):
            token_ids = [bos_id] + token_ids
        if add_eos and (not token_ids or token_ids[-1] != eos_id):
            token_ids = token_ids + [eos_id]
        return token_ids

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens).strip()

    @property
    def token_to_id(self) -> dict[str, int]:
        vocab = self.tokenizer.get_vocab()
        return {token: int(token_id) for token, token_id in vocab.items()}

    @property
    def id_to_token(self) -> dict[int, str]:
        return {token_id: token for token, token_id in self.tokenizer.get_vocab().items()}

    def save(self, path: str | Path) -> None:
        path_obj = Path(path)
        ensure_dir(path_obj.parent)
        self.tokenizer.save(str(path_obj))

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        return cls(tokenizer=Tokenizer.from_file(str(path)))
