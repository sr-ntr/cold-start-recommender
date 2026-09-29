from pathlib import Path

import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INTERACTIONS_DIR = PROJECT_ROOT / "data" / "processed" / "electronics_interactions"

MF_USERS_PATH = PROJECT_ROOT / "data" / "processed" / "mf_users.parquet"

MF_ITEMS_PATH = PROJECT_ROOT / "data" / "processed" / "mf_items.parquet"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "mf_training"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def transform_partition(input_path, output_path):
    con = duckdb.connect()

    query = f"""
        COPY (
            SELECT
                u.user_idx,
                i.mf_item_idx,
                CASE
                    WHEN r.rating = 1 THEN -1.0
                    WHEN r.rating = 2 THEN -0.5
                    WHEN r.rating = 3 THEN  0.0
                    WHEN r.rating = 4 THEN  0.5
                    WHEN r.rating = 5 THEN  1.0
                END::FLOAT AS target

            FROM read_parquet('{input_path}') AS r

            INNER JOIN read_parquet('{MF_USERS_PATH}') AS u
                ON r.user_id = u.user_id

            INNER JOIN read_parquet('{MF_ITEMS_PATH}') AS i
                ON r.parent_asin = i.parent_asin

            WHERE r.timestamp < 1640995200000
        )
        TO '{output_path}'
        (FORMAT PARQUET);
    """

    con.execute(query)
    con.close()


def main():
    input_files = sorted(INTERACTIONS_DIR.glob("part-*.parquet"))

    print(f"Found {len(input_files)} interaction partitions.")

    for partition_number, input_path in enumerate(input_files):
        output_path = OUTPUT_DIR / f"part-{partition_number:04d}.parquet"

        if output_path.exists():
            print(f"Skipping existing {output_path.name}")
            continue

        print(f"Processing {input_path.name}...")

        transform_partition(
            input_path=input_path,
            output_path=output_path,
        )

        print(f"Saved {output_path.name}")


if __name__ == "__main__":
    main()
