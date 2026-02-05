import glob
import io
import itertools
import json
import os
import tarfile

import sentencepiece as spm
import torch
from datasets import IterableDataset as HFIterableDataset
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from torch.utils.data import DataLoader, IterableDataset


def require_zstandard():
    try:
        import zstandard  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "zstandard is required to read .jsonl.zst files. "
            "Install with: pip install zstandard"
        ) from exc


def iter_jsonl_zst(path):
    import zstandard as zstd

    with open(path, "rb") as f:
        reader = zstd.ZstdDecompressor().stream_reader(f)
        with io.TextIOWrapper(reader, encoding="utf-8") as text_stream:
            for line in text_stream:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def openwebtext2_generator(files):
    for path in files:
        for record in iter_jsonl_zst(path):
            yield record


def find_openwebtext2_files(cache_dir):
    search_roots = []
    if cache_dir:
        search_roots.append(cache_dir)
    search_roots.append(os.path.expanduser("~/.cache/huggingface/datasets"))

    for root in search_roots:
        pattern = os.path.join(root, "downloads", "extracted", "**", "*.jsonl.zst")
        files = glob.glob(pattern, recursive=True)
        if files:
            return sorted(files)
    return []


def extract_openwebtext2_tar(tar_path, extract_dir):
    os.makedirs(extract_dir, exist_ok=True)
    marker = os.path.join(extract_dir, ".extracted")
    if os.path.exists(marker):
        return
    with tarfile.open(tar_path, "r:*") as tar:
        tar.extractall(path=extract_dir)
    with open(marker, "w", encoding="utf-8") as f:
        f.write("ok")


def load_openwebtext2(args):
    require_zstandard()
    files = find_openwebtext2_files(args.cache_dir)
    if not files:
        tar_path = hf_hub_download(
            repo_id="segyges/OpenWebText2",
            filename="openwebtext2.jsonl.zst.tar",
            repo_type="dataset",
            cache_dir=args.cache_dir,
        )
        extract_dir = os.path.join(os.path.dirname(tar_path), "openwebtext2_extracted")
        extract_openwebtext2_tar(tar_path, extract_dir)
        files = sorted(glob.glob(os.path.join(extract_dir, "*.jsonl.zst")))

    if not files:
        raise RuntimeError("OpenWebText2 files not found after download/extract.")

    dataset = HFIterableDataset.from_generator(lambda: openwebtext2_generator(files))
    return dataset


def load_hf_dataset(args, split):
    if args.dataset == "segyges/OpenWebText2":
        return load_openwebtext2(args)
    return load_dataset(
        args.dataset,
        split=split,
        streaming=args.streaming,
        cache_dir=args.cache_dir,
    )


def format_conversations(conversations):
    parts = []
    for turn in conversations:
        role = turn.get("from") or turn.get("role") or ""
        if role == "human":
            role = "user"
        elif role == "gpt":
            role = "assistant"
        value = turn.get("value") or turn.get("content") or ""
        value = value.strip()
        if not value:
            continue
        parts.append(f"{role}: {value}")
    return "\n".join(parts).strip()


def extract_text(sample, text_field, conversations_field):
    text = None
    if text_field and text_field in sample:
        text = sample.get(text_field)
    if (not text) and conversations_field:
        conversations = sample.get(conversations_field)
        if conversations:
            text = format_conversations(conversations)
    if not text:
        for value in sample.values():
            if isinstance(value, str) and value.strip():
                text = value
                break
    return text


class PackedDataset(IterableDataset):
    def __init__(
        self,
        hf_dataset,
        sp_model,
        block_size,
        eos_id,
        text_field,
        conversations_field,
        shuffle,
        shuffle_buffer,
        seed,
        max_bytes,
    ):
        super().__init__()
        self.hf_dataset = hf_dataset
        self.sp_model = sp_model
        self.block_size = block_size
        self.eos_id = eos_id
        self.text_field = text_field
        self.conversations_field = conversations_field
        self.shuffle = shuffle
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        self.max_bytes = max_bytes
        self.epoch = 0

    def _maybe_shuffle(self, dataset):
        if not self.shuffle:
            return dataset
        seed = self.seed + self.epoch
        if hasattr(dataset, "shuffle"):
            return dataset.shuffle(buffer_size=self.shuffle_buffer, seed=seed)
        return dataset

    def __iter__(self):
        self.epoch += 1
        dataset = self._maybe_shuffle(self.hf_dataset)

        buffer = []
        bytes_seen = 0
        for sample in dataset:
            text = extract_text(sample, self.text_field, self.conversations_field)
            if not text:
                continue
            if self.max_bytes is not None:
                bytes_seen += len(text.encode("utf-8"))
                if bytes_seen > self.max_bytes:
                    break
            tokens = self.sp_model.encode(text, out_type=int)
            tokens.append(self.eos_id)
            buffer.extend(tokens)
            while len(buffer) >= self.block_size + 1:
                chunk = buffer[: self.block_size + 1]
                buffer = buffer[self.block_size + 1 :]
                yield torch.tensor(chunk, dtype=torch.long)


def infinite_loader(loader):
    while True:
        for batch in loader:
            yield batch


def ensure_tokenizer(args, is_main):
    require_zstandard()
    os.makedirs(args.tokenizer_dir, exist_ok=True)
    model_prefix = os.path.join(args.tokenizer_dir, args.tokenizer_prefix)
    model_path = model_prefix + ".model"

    if is_main and not os.path.exists(model_path):
        ds = load_hf_dataset(args, split="train")
        if args.streaming and hasattr(ds, "shuffle"):
            ds = ds.shuffle(buffer_size=args.shuffle_buffer, seed=args.seed)
        samples_path = os.path.join(args.tokenizer_dir, "tokenizer_samples.txt")
        with open(samples_path, "w", encoding="utf-8") as f:
            for sample in itertools.islice(ds, args.tokenizer_samples):
                text = extract_text(sample, args.text_field, args.conversations_field)
                if text:
                    f.write(text.replace("\n", " ") + "\n")

        spm.SentencePieceTrainer.train(
            input=samples_path,
            model_prefix=model_prefix,
            vocab_size=args.vocab_size,
            model_type="unigram",
            character_coverage=1.0,
            byte_fallback=True,
            hard_vocab_limit=False,
            bos_id=-1,
            eos_id=args.eos_id,
            pad_id=0,
            unk_id=1,
        )

    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    return sp


def build_loaders(args, sp, build_val):
    require_zstandard()
    max_bytes = None
    if args.max_data_gb and args.max_data_gb > 0:
        max_bytes = int(args.max_data_gb * 1024 * 1024 * 1024)

    train_raw = load_hf_dataset(args, split="train")

    val_raw = None
    if args.streaming:
        if hasattr(train_raw, "shuffle"):
            train_raw = train_raw.shuffle(buffer_size=args.shuffle_buffer, seed=args.seed)
        if args.val_examples > 0:
            val_raw = train_raw.take(args.val_examples)
            train_raw = train_raw.skip(args.val_examples)
    else:
        if args.val_ratio > 0:
            split = train_raw.train_test_split(test_size=args.val_ratio, seed=args.seed)
            train_raw = split["train"]
            val_raw = split["test"]

    train_shuffle = not (args.streaming and args.val_examples > 0)
    train_dataset = PackedDataset(
        train_raw,
        sp,
        args.block_size,
        args.eos_id,
        args.text_field,
        args.conversations_field,
        shuffle=train_shuffle,
        shuffle_buffer=args.shuffle_buffer,
        seed=args.seed,
        max_bytes=max_bytes,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=True,
    )

    val_loader = None
    if build_val and val_raw is not None:
        val_dataset = PackedDataset(
            val_raw,
            sp,
            args.block_size,
            args.eos_id,
            args.text_field,
            args.conversations_field,
            shuffle=False,
            shuffle_buffer=args.shuffle_buffer,
            seed=args.seed,
            max_bytes=max_bytes,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.eval_batch_size,
            num_workers=0,
            pin_memory=args.pin_memory,
            drop_last=True,
        )

    return train_loader, val_loader
