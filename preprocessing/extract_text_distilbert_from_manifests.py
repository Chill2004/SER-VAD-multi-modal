import os
import argparse
import json
import warnings
from pathlib import Path

ENV_DATA_ROOT = Path(os.getenv("SER_DATA_ROOT", "data"))
ENV_OUTPUT_ROOT = Path(os.getenv("SER_OUTPUT_ROOT", "outputs"))


warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel


# CLI
parser = argparse.ArgumentParser()
parser.add_argument("--setting", type=str, required=True)
args = parser.parse_args()

SETTING = args.setting


# Configuration
AUG_ROOT = Path(str(ENV_OUTPUT_ROOT / 'aug_runs_audio'))

# Use manifests after transcripts have been attached.
INPUT_FEATURE_DIR = "wav2vec2_features"
OUTPUT_FEATURE_DIR = "text_distilbert_features"

SETTING_DIR = AUG_ROOT / SETTING / INPUT_FEATURE_DIR
OUTPUT_DIR = AUG_ROOT / SETTING / OUTPUT_FEATURE_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TEXT_MODEL_NAME = "distilbert-base-uncased"
TEXT_COL = "transcript"

SPLITS = ["train", "val", "test"]
MAX_TEXT_LEN = 96
BATCH_SIZE = 32

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Helpers
def clean_text(x):
    if pd.isna(x):
        return ""
    x = str(x).strip()
    x = " ".join(x.split())
    return x


def safe_name(x):
    x = str(x)
    for bad in ["/", "\\", ":", "*", "?", "\"", "<", ">", "|", " "]:
        x = x.replace(bad, "_")
    return x


def get_row_id(row, idx):
    if "utterance_id" in row and pd.notna(row["utterance_id"]):
        return safe_name(row["utterance_id"])
    if "source_utt_id" in row and pd.notna(row["source_utt_id"]):
        return safe_name(row["source_utt_id"])
    return f"row_{idx:06d}"


def mean_pool(last_hidden_state, attention_mask):
    """
    last_hidden_state: (B, L, H)
    attention_mask:    (B, L)
    """
    mask = attention_mask.unsqueeze(-1).float()
    masked_hidden = last_hidden_state * mask
    summed = masked_hidden.sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-9)
    return summed / denom


def encode_texts(texts, tokenizer, model):
    all_embs = []

    for start in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[start:start + BATCH_SIZE]

        enc = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_TEXT_LEN,
            return_tensors="pt",
        )

        input_ids = enc["input_ids"].to(DEVICE)
        attention_mask = enc["attention_mask"].to(DEVICE)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            # DistilBERT has no pooler output, so use masked mean pooling.
            emb = mean_pool(outputs.last_hidden_state, attention_mask)

        emb = emb.detach().cpu().numpy().astype(np.float32)
        all_embs.append(emb)

        print(f"Encoded {min(start + BATCH_SIZE, len(texts))}/{len(texts)} texts")

    return np.concatenate(all_embs, axis=0)


def process_split(split, tokenizer, model):
    in_csv = SETTING_DIR / f"{split}_manifest_with_text.csv"
    if not in_csv.exists():
        raise FileNotFoundError(f"Missing manifest with text: {in_csv}")

    df = pd.read_csv(in_csv).copy()

    if TEXT_COL not in df.columns:
        raise ValueError(f"Missing column '{TEXT_COL}' in {in_csv}")

    df[TEXT_COL] = df[TEXT_COL].apply(clean_text)

    before = len(df)
    df = df[df[TEXT_COL].str.len() > 0].copy().reset_index(drop=True)
    after = len(df)

    print(f"\n===== {split.upper()} =====")
    print(f"Rows kept: {after}/{before}")

    texts = df[TEXT_COL].tolist()
    embs = encode_texts(texts, tokenizer, model)

    split_seq_dir = OUTPUT_DIR / "text_embeddings"
    split_seq_dir.mkdir(parents=True, exist_ok=True)

    text_feat_paths = []

    for i, row in df.iterrows():
        row_id = get_row_id(row, i)
        out_path = split_seq_dir / f"{row_id}_text_distilbert.npy"
        np.save(out_path, embs[i])
        text_feat_paths.append(str(out_path))

    df["text_model_name"] = TEXT_MODEL_NAME
    df["text_max_len"] = MAX_TEXT_LEN
    df["text_hidden_size"] = int(embs.shape[1])
    df["text_feat_path"] = text_feat_paths
    df["text_feature_type"] = "distilbert_mean_pool"

    out_csv = OUTPUT_DIR / f"{split}_manifest.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    meta = {
        "setting": SETTING,
        "split": split,
        "input_manifest": str(in_csv),
        "output_manifest": str(out_csv),
        "text_model_name": TEXT_MODEL_NAME,
        "text_feature_type": "distilbert_mean_pool",
        "max_text_len": MAX_TEXT_LEN,
        "hidden_size": int(embs.shape[1]),
        "rows_in": int(before),
        "rows_out": int(after),
    }

    with open(OUTPUT_DIR / f"{split}_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved: {out_csv}")


def main():
    print("SETTING:", SETTING)
    print("INPUT:", SETTING_DIR)
    print("OUTPUT:", OUTPUT_DIR)
    print("TEXT_MODEL_NAME:", TEXT_MODEL_NAME)
    print("DEVICE:", DEVICE)

    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
    model = AutoModel.from_pretrained(TEXT_MODEL_NAME)
    model.to(DEVICE)
    model.eval()

    for split in SPLITS:
        process_split(split, tokenizer, model)

    print("\nDone. Text embeddings saved to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()