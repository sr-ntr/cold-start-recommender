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


# Using simple-writer


def generate_embeddings(model, catalog, con):
    num_products = len(catalog)

    embeddings = np.lib.format.open_memmap(
        EMBEDDING_PATH,
        mode="w+",
        dtype=np.float32,
        shape=(num_products, EMBEDDING_DIM),
    )

    for start in range(0, num_products, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, num_products)

        chunk = catalog.iloc[start:end]

        metadata = load_metadata_chunk(
            con,
            chunk["parent_asin"].tolist(),
        )

        metadata = (
            chunk[["item_idx", "parent_asin"]]
            .merge(
                metadata,
                on="parent_asin",
                how="left",
                validate="one_to_one",
            )
            .sort_values("item_idx")
        )

        texts = [
            build_truncated_product_text(
                row,
                model.tokenizer,
            )
            for _, row in metadata.iterrows()
        ]

        chunk_embeddings = model.encode(
            texts,
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        embeddings[start:end] = chunk_embeddings.astype(np.float32)

        print(f"Processed {end:,}/{num_products:,} ({end / num_products:.1%})")

    embeddings.flush()

    print(f"Saved embeddings to: {EMBEDDING_PATH}")


if __name__ == "__main__":
    model = load_model()
    catalog = load_catalog()
    con = duckdb.connect()

    generate_embeddings(model, catalog, con)
