from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[2]

METADATA_PATH = PROJECT_ROOT / "data" / "raw" / "meta_Electronics.jsonl.gz"
CATALOG_PATH = PROJECT_ROOT / "data" / "processed" / "product_catalog.parquet"
EMBEDDING_PATH = PROJECT_ROOT / "data" / "processed" / "product_embeddings.npy"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
BATCH_SIZE = 32


def load_model():
    model = SentenceTransformer(MODEL_NAME)

    actual_dim = model.get_embedding_dimension()

    if actual_dim != EMBEDDING_DIM:
        raise ValueError(
            f"Expected embedding dimension {EMBEDDING_DIM}, "
            f"but model returned {actual_dim}"
        )

    return model


# if __name__ == "__main__":
#     model = load_model()

#     print(f"Model: {MODEL_NAME}")
#     print(f"Embedding dimension: {model.get_embedding_dimension()}")
#     print(f"Device: {model.device}")

CHUNK_SIZE = 5_000


def load_catalog():
    return pd.read_parquet(CATALOG_PATH)


def load_metadata_chunk(con, asins):
    con.register("chunk_catalog", pd.DataFrame({"parent_asin": asins}))

    query = f"""
        SELECT
            c.parent_asin,
            m.title,
            m.description,
            m.features,
            m.categories
        FROM chunk_catalog c
        INNER JOIN read_json_auto('{METADATA_PATH}') m
            ON c.parent_asin = m.parent_asin
    """

    return con.sql(query).df()


# if __name__ == "__main__":
#     model = load_model()

#     catalog = load_catalog()

#     con = duckdb.connect()

#     test_asins = catalog["parent_asin"].iloc[:CHUNK_SIZE].tolist()

#     metadata = load_metadata_chunk(con, test_asins)

#     print(f"Requested products: {len(test_asins):,}")
#     print(f"Metadata rows returned: {len(metadata):,}")
#     print(metadata.head())

from src.features.product_text import build_truncated_product_text


def generate_test_embeddings(model, metadata, output_path):
    texts = [
        build_truncated_product_text(row, model.tokenizer)
        for _, row in metadata.iterrows()
    ]

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    print(f"Embedding shape: {embeddings.shape}")
    print(f"Embedding dtype: {embeddings.dtype}")

    mmap = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=embeddings.shape,
    )

    mmap[:] = embeddings.astype(np.float32)
    mmap.flush()

    print(f"Saved embeddings to: {output_path}")


test_embeddings = np.load(
    PROJECT_ROOT / "data" / "processed" / "test_embeddings.npy",
    mmap_mode="r",
)

print(test_embeddings.shape)
print(test_embeddings.dtype)
print(np.linalg.norm(test_embeddings[0]))

# if __name__ == "__main__":
#     model = load_model()

#     catalog = load_catalog()

#     con = duckdb.connect()

#     test_catalog = catalog.iloc[:CHUNK_SIZE]

#     metadata = load_metadata_chunk(
#         con,
#         test_catalog["parent_asin"].tolist(),
#     )

#     generate_test_embeddings(
#         model,
#         metadata,
#         PROJECT_ROOT / "data" / "processed" / "test_embeddings.npy",
#     )
