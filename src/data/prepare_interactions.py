from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_PATH = PROJECT_ROOT / "data" / "raw" / "Electronics.csv.gz"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "electronics_interactions"

CHUNK_SIZE = 500_000


def prepare_interactions():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            RAW_PATH,
            usecols=["user_id", "parent_asin", "rating", "timestamp"],
            chunksize=CHUNK_SIZE,
        )
    ):
        chunk["rating"] = chunk["rating"].astype("float32")
        chunk["timestamp"] = chunk["timestamp"].astype("int64")

        output_path = OUTPUT_DIR / f"part-{chunk_number:04d}.parquet"

        chunk.to_parquet(
            output_path,
            index=False,
            engine="pyarrow",
        )

        print(f"Saved {output_path.name}: {len(chunk):,} rows")


if __name__ == "__main__":
    prepare_interactions()
