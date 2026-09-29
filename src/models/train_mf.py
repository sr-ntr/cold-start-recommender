from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAINING_DIR = PROJECT_ROOT / "data" / "processed" / "mf_training"

BATCH_SIZE = 1024
RANDOM_SEED = 42

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "mf_baseline.pt"
CONFIG_PATH = MODEL_DIR / "mf_baseline_config.json"

LATENT_DIM = 16


class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, latent_dim=LATENT_DIM):
        super().__init__()

        self.user_embeddings = nn.Embedding(
            num_users,
            latent_dim,
            sparse=True,
        )

        self.item_embeddings = nn.Embedding(
            num_items,
            latent_dim,
            sparse=True,
        )

        self.user_bias = nn.Embedding(
            num_users,
            1,
            sparse=True,
        )

        self.item_bias = nn.Embedding(
            num_items,
            1,
            sparse=True,
        )

        self.global_bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, user_idx, item_idx):
        user_vec = self.user_embeddings(user_idx)
        item_vec = self.item_embeddings(item_idx)

        interaction = (user_vec * item_vec).sum(dim=1)

        user_bias = self.user_bias(user_idx).squeeze(-1)
        item_bias = self.item_bias(item_idx).squeeze(-1)

        return interaction + user_bias + item_bias + self.global_bias


def load_partition(path, seed):
    df = pd.read_parquet(path)

    rng = np.random.default_rng(seed)

    order = rng.permutation(len(df))
    df = df.iloc[order]

    return df


def iter_batches(df, batch_size):
    for start in range(0, len(df), batch_size):
        batch = df.iloc[start : start + batch_size]

        yield (
            torch.tensor(
                batch["user_idx"].to_numpy(),
                dtype=torch.long,
            ),
            torch.tensor(
                batch["mf_item_idx"].to_numpy(),
                dtype=torch.long,
            ),
            torch.tensor(
                batch["target"].to_numpy(),
                dtype=torch.float32,
            ),
        )


if __name__ == "__main__":
    import time

    NUM_USERS = 16_058_864
    NUM_ITEMS = 1_395_130

    MAX_PARTITIONS = None

    model = MatrixFactorization(
        num_users=NUM_USERS,
        num_items=NUM_ITEMS,
        latent_dim=LATENT_DIM,
    )

    model.train()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.01,
    )

    partition_paths = sorted(TRAINING_DIR.glob("part-*.parquet"))

    if MAX_PARTITIONS is not None:
        partition_paths = partition_paths[:MAX_PARTITIONS]

    total_examples = 0
    total_loss = 0.0

    start_time = time.perf_counter()

    for partition_number, partition_path in enumerate(partition_paths):
        partition_start = time.perf_counter()

        df = load_partition(
            partition_path,
            seed=RANDOM_SEED + partition_number,
        )

        partition_loss = 0.0
        partition_examples = 0

        for batch_users, batch_items, batch_targets in iter_batches(
            df,
            BATCH_SIZE,
        ):
            optimizer.zero_grad()

            predictions = model(
                batch_users,
                batch_items,
            )

            loss = nn.functional.mse_loss(
                predictions,
                batch_targets,
            )

            loss.backward()
            optimizer.step()

            batch_size = len(batch_targets)

            partition_loss += loss.item() * batch_size
            partition_examples += batch_size

        total_loss += partition_loss
        total_examples += partition_examples

        elapsed = time.perf_counter() - partition_start

        print(
            f"Partition {partition_number + 1}/"
            f"{len(partition_paths)} "
            f"| examples = {partition_examples:,} "
            f"| loss = "
            f"{partition_loss / partition_examples:.4f} "
            f"| time = {elapsed:.2f}s"
        )

        del df

    total_elapsed = time.perf_counter() - start_time

    print("\nTraining smoke test complete.")
    print(f"Total examples: {total_examples:,}")
    print(f"Average loss: {total_loss / total_examples:.4f}")
    print(f"Total time: {total_elapsed:.2f}s")

    # Save the trained model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), MODEL_PATH)

    config = {
        "num_users": NUM_USERS,
        "num_items": NUM_ITEMS,
        "latent_dim": 16,
        "batch_size": BATCH_SIZE,
        "learning_rate": 0.01,
        "epochs": 1,
        "random_seed": RANDOM_SEED,
        "rating_mapping": {
            "1": -1.0,
            "2": -0.5,
            "3": 0.0,
            "4": 0.5,
            "5": 1.0,
        },
    }

    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Model saved to: {MODEL_PATH}")
    print(f"Config saved to: {CONFIG_PATH}")
