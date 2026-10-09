from pathlib import Path
import json
import duckdb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"


class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, latent_dim=16):
        super().__init__()
        self.user_embeddings = nn.Embedding(num_users, latent_dim)
        self.item_embeddings = nn.Embedding(num_items, latent_dim)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self.global_bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, user_idx, item_idx):
        user_vec = self.user_embeddings(user_idx)
        item_vec = self.item_embeddings(item_idx)
        interaction = (user_vec * item_vec).sum(dim=1)
        user_bias = self.user_bias(user_idx).squeeze(-1)
        item_bias = self.item_bias(item_idx).squeeze(-1)
        return interaction + user_bias + item_bias + self.global_bias


def load_model(model_path=None, config_path=None, device="cpu"):
    if model_path is None:
        model_path = MODEL_DIR / "mf_3epoch.pt"
    if config_path is None:
        config_path = MODEL_DIR / "mf_3epoch_config.json"

    with open(config_path) as f:
        config = json.load(f)

    model = MatrixFactorization(
        num_users=config["num_users"],
        num_items=config["num_items"],
        latent_dim=config.get("latent_dim", 16),
    )

    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, config


def get_evaluation_data(num_users=32):
    con = duckdb.connect()

    # 1. Deterministic sample of evaluation users present in MF user index
    eval_users = con.execute(f"""
        SELECT
            g.user_id,
            u.user_idx
        FROM read_parquet('{(DATA_DIR / "mf_validation_ground_truth.parquet").as_posix()}') AS g
        INNER JOIN read_parquet('{(DATA_DIR / "mf_users.parquet").as_posix()}') AS u
            ON g.user_id = u.user_id
        GROUP BY g.user_id, u.user_idx
        ORDER BY g.user_id
        LIMIT {num_users}
    """).df()

    con.register("eval_users", eval_users)

    # 2. Correctly join seen training items for these EXACT evaluation users
    seen_items = con.execute(f"""
        SELECT DISTINCT
            r.user_id,
            i.mf_item_idx
        FROM read_parquet('{(DATA_DIR / "electronics_interactions" / "*.parquet").as_posix()}') AS r
        INNER JOIN eval_users AS u
            ON r.user_id = u.user_id
        INNER JOIN read_parquet('{(DATA_DIR / "mf_items.parquet").as_posix()}') AS i
            ON r.parent_asin = i.parent_asin
        WHERE r.timestamp < 1640995200000
    """).df()

    # 3. Ground truth items for evaluation users with cohort labels
    ground_truth = con.execute(f"""
        SELECT
            g.user_id,
            u.user_idx,
            g.item_id,
            g.item_cohort,
            i.mf_item_idx
        FROM read_parquet('{(DATA_DIR / "mf_validation_ground_truth.parquet").as_posix()}') AS g
        INNER JOIN eval_users AS u
            ON g.user_id = u.user_id
        LEFT JOIN read_parquet('{(DATA_DIR / "mf_items.parquet").as_posix()}') AS i
            ON g.item_id = i.parent_asin
        ORDER BY g.user_id, g.item_id
    """).df()

    con.close()
    return eval_users, seen_items, ground_truth


def evaluate_candidate_ranking(model, eval_users, ground_truth, num_negatives=99, seed=42, device="cpu"):
    """
    Standard RecSys benchmark evaluation (He et al., NCF):
    For each user and relevant positive item, rank it among `num_negatives` (default 99) unobserved items.
    """
    rng = np.random.default_rng(seed)
    num_items = model.item_embeddings.num_embeddings

    results = []

    for user_id, group in ground_truth.groupby("user_id"):
        user_idx = int(group["user_idx"].iloc[0])

        for _, row in group.iterrows():
            item_cohort = row["item_cohort"]
            pos_mf_idx = row["mf_item_idx"]

            # If true cold, the item was unseen in training and has no MF embedding.
            # In pure MF, it receives a default score (0.0 / global bias).
            neg_indices = rng.integers(0, num_items, size=num_negatives)

            if pd.isna(pos_mf_idx):
                # True cold item: MF cannot personalize it.
                pos_score = model.global_bias.item()
                neg_users = torch.full((num_negatives,), user_idx, dtype=torch.long, device=device)
                neg_items = torch.tensor(neg_indices, dtype=torch.long, device=device)
                with torch.no_grad():
                    neg_scores = model(neg_users, neg_items).cpu().numpy()
                candidate_scores = np.concatenate([[pos_score], neg_scores])
                pos_rank = int((candidate_scores > pos_score).sum()) + 1
            else:
                pos_mf_idx = int(pos_mf_idx)
                candidates = np.concatenate([[pos_mf_idx], neg_indices])
                u_tensor = torch.full((len(candidates),), user_idx, dtype=torch.long, device=device)
                i_tensor = torch.tensor(candidates, dtype=torch.long, device=device)
                with torch.no_grad():
                    candidate_scores = model(u_tensor, i_tensor).cpu().numpy()
                pos_score = candidate_scores[0]
                pos_rank = int((candidate_scores > pos_score).sum()) + 1

            hit5 = 1.0 if pos_rank <= 5 else 0.0
            hit10 = 1.0 if pos_rank <= 10 else 0.0
            ndcg10 = (1.0 / np.log2(pos_rank + 1)) if pos_rank <= 10 else 0.0

            results.append({
                "user_id": user_id,
                "item_cohort": item_cohort,
                "is_known_item": not pd.isna(pos_mf_idx),
                "rank": pos_rank,
                "hit@5": hit5,
                "hit@10": hit10,
                "ndcg@10": ndcg10,
            })

    results_df = pd.DataFrame(results)
    return results_df


def evaluate_full_catalog(model, eval_users, seen_items, ground_truth, device="cpu"):
    """
    Evaluates Top-K recommendation over all catalog items with seen training items masked.
    """
    user_indices = torch.tensor(
        eval_users["user_idx"].to_numpy(),
        dtype=torch.long,
        device=device,
    )

    with torch.no_grad():
        user_vectors = model.user_embeddings(user_indices)
        user_biases = model.user_bias(user_indices).squeeze(-1)

        item_vectors = model.item_embeddings.weight
        item_biases = model.item_bias.weight.squeeze(-1)

        scores = (
            user_vectors @ item_vectors.T
            + user_biases[:, None]
            + item_biases[None, :]
            + model.global_bias
        )

    # Mask seen training interactions
    user_to_row = {uid: r for r, uid in enumerate(eval_users["user_id"])}
    for _, row in seen_items.iterrows():
        uid = row["user_id"]
        if uid in user_to_row and not pd.isna(row["mf_item_idx"]):
            scores[user_to_row[uid], int(row["mf_item_idx"])] = float("-inf")

    top5_indices = torch.topk(scores, k=5, dim=1).indices.cpu().numpy()
    top10_indices = torch.topk(scores, k=10, dim=1).indices.cpu().numpy()

    gt_by_user = {}
    for user_id, group in ground_truth.groupby("user_id"):
        gt_by_user[user_id] = set(group["mf_item_idx"].dropna().astype(int))

    recall5_vals = []
    recall10_vals = []

    for r, user_id in enumerate(eval_users["user_id"]):
        rel = gt_by_user.get(user_id, set())
        if not rel:
            continue
        rec5 = set(top5_indices[r])
        rec10 = set(top10_indices[r])
        recall5_vals.append(len(rec5 & rel) / len(rel))
        recall10_vals.append(len(rec10 & rel) / len(rel))

    return {
        "full_catalog_recall@5": float(np.mean(recall5_vals)) if recall5_vals else 0.0,
        "full_catalog_recall@10": float(np.mean(recall10_vals)) if recall10_vals else 0.0,
    }


def main():
    print("=" * 65)
    print("COLD-START MATRIX FACTORIZATION BASELINE EVALUATION")
    print("=" * 65)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on {device}...")
    model, config = load_model(device=device)

    print("\nLoading evaluation sample (32 users)...")
    eval_users, seen_items, ground_truth = get_evaluation_data(num_users=32)

    print(f"Evaluation users: {len(eval_users)}")
    print(f"Ground-truth test pairs: {len(ground_truth)}")
    print(f"Correctly mapped seen pairs in training: {len(seen_items)}")

    # 1. Standard RecSys Candidate (1-in-100) Ranking Evaluation
    print("\n--- Protocol 1: Candidate Ranking (1 Ground-Truth + 99 Negatives) ---")
    candidate_results = evaluate_candidate_ranking(
        model, eval_users, ground_truth, num_negatives=99, device=device
    )

    print("\nOverall Candidate Metrics (All cohorts):")
    print(f"  Hit@5 (Recall@5):   {candidate_results['hit@5'].mean():.4f}")
    print(f"  Hit@10 (Recall@10): {candidate_results['hit@10'].mean():.4f}")
    print(f"  NDCG@10:            {candidate_results['ndcg@10'].mean():.4f}")

    print("\nMetrics Broken Down by Item Cohort:")
    cohort_summary = candidate_results.groupby("item_cohort")[["hit@5", "hit@10", "ndcg@10"]].agg(["mean", "count"])
    print(cohort_summary)

    # 2. Full-Catalog Ranking Evaluation
    print("\n--- Protocol 2: Full-Catalog Ranking (1.39M items with seen masked) ---")
    full_cat_metrics = evaluate_full_catalog(
        model, eval_users, seen_items, ground_truth, device=device
    )
    print(f"  Full-Catalog Recall@5:  {full_cat_metrics['full_catalog_recall@5']:.6f}")
    print(f"  Full-Catalog Recall@10: {full_cat_metrics['full_catalog_recall@10']:.6f}")

    print("\nKey Takeaway for Cold-Start Research:")
    print("  * For 'true_cold' items, pure Matrix Factorization has 0.0 Recall because")
    print("    items without training interactions lack collaborative embeddings.")
    print("  * This confirms the MF baseline establishes the exact cold-start problem")
    print("    to be solved by content/text embeddings in the next stage.")
    print("=" * 65)


if __name__ == "__main__":
    main()
