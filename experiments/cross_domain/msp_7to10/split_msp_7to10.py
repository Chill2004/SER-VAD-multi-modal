import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ENV_DATA_ROOT = Path(os.getenv("SER_DATA_ROOT", "data"))
ENV_OUTPUT_ROOT = Path(os.getenv("SER_OUTPUT_ROOT", "outputs"))



DATA_DIR = Path(str(ENV_DATA_ROOT / 'abstracttts_podcast_7to10s'))
MANIFEST = DATA_DIR / "manifest_7to10s.csv"

df = pd.read_csv(MANIFEST).copy()

required_cols = [
    "utterance_id",
    "audio_path",
    "duration",
    "transcript",
    "valence_norm",
    "arousal_norm",
    "dominance_norm",
]

missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

df = df.dropna(
    subset=[
        "audio_path",
        "valence_norm",
        "arousal_norm",
        "dominance_norm",
    ]
).copy()

df["transcript"] = df["transcript"].fillna("").astype(str)

# Keep the original dataset ratio.
train_ratio = 7082 / 10039
val_ratio = 787 / 10039
test_ratio = 2170 / 10039

train_df, temp_df = train_test_split(
    df,
    train_size=train_ratio,
    random_state=42,
    shuffle=True,
)

relative_val_ratio = val_ratio / (val_ratio + test_ratio)

val_df, test_df = train_test_split(
    temp_df,
    train_size=relative_val_ratio,
    random_state=42,
    shuffle=True,
)

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

train_df.to_csv(DATA_DIR / "train_manifest.csv", index=False, encoding="utf-8-sig")
val_df.to_csv(DATA_DIR / "val_manifest.csv", index=False, encoding="utf-8-sig")
test_df.to_csv(DATA_DIR / "test_manifest.csv", index=False, encoding="utf-8-sig")

print("train:", len(train_df))
print("val  :", len(val_df))
print("test :", len(test_df))
print("total:", len(train_df) + len(val_df) + len(test_df))

print("\nDuration:")
print("train:", train_df["duration"].mean(), train_df["duration"].min(), train_df["duration"].max())
print("val  :", val_df["duration"].mean(), val_df["duration"].min(), val_df["duration"].max())
print("test :", test_df["duration"].mean(), test_df["duration"].min(), test_df["duration"].max())

print("\nTarget mean:")
for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
    print(name)
    print(split_df[["valence_norm", "arousal_norm", "dominance_norm"]].mean())