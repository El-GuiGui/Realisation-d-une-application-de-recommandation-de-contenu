"""
Pre-calcul des recommandations pour tous les utilisateurs.
Génère un fichier JSON utilisable par la Lambda sans dépendances lourdes.

Lancer après avoir exécuté les 4 notebooks :
    cd proj10
    python scripts/precompute.py
"""
import json
import os
import numpy as np
import joblib
import pandas as pd
from sklearn.preprocessing import normalize
from scipy.sparse import load_npz

OUT = os.path.join(os.path.dirname(__file__), "..", "out")
CB_DIR = os.path.join(OUT, "models", "content_based")
CF_DIR = os.path.join(OUT, "models", "collaborative")
CFG_DIR = os.path.join(OUT, "models", "config")

# Chargement des modèles
print("Chargement des modèles...")
embeddings = np.load(os.path.join(CB_DIR, "embeddings_reduced.npy"))
embeddings = normalize(embeddings, norm="l2", axis=1)

with open(os.path.join(CB_DIR, "user_articles.json"), "r") as f:
    user_articles = json.load(f)

als_model = joblib.load(os.path.join(CF_DIR, "als_model.joblib"))
user_item = load_npz(os.path.join(CF_DIR, "user_item_matrix.npz"))

with open(os.path.join(CF_DIR, "user_map.json"), "r") as f:
    user_map = {int(k): v for k, v in json.load(f).items()}
with open(os.path.join(CF_DIR, "article_map_inv.json"), "r") as f:
    article_map_inv = json.load(f)

with open(os.path.join(CFG_DIR, "model_config.json"), "r") as f:
    config = json.load(f)

# old :
# alpha = 0.8

alpha = config.get("best_alpha", 1.0)

# Cold start : top 5 articles récence + popularité
cs_df = pd.read_parquet(os.path.join(CB_DIR, "cold_start_ranking.parquet"))
cs_df = cs_df.sort_values("cold_start_score", ascending=False)
cold_start_articles = [
    {"article_id": int(row["article_id"]), "score": round(float(row["cold_start_score"]), 4)}
    for _, row in cs_df.head(5).iterrows()
]

print(f"Utilisateurs : {len(user_articles):,}")
print(f"Alpha : {alpha}")


def recommend_hybrid(user_id):
    """Calcule les 5 recommandations hybrides avec scores."""
    uid_str = str(user_id)

    # Content-Based
    cb_scores = {}
    article_ids = user_articles.get(uid_str, [])
    if article_ids:
        valid_ids = [a for a in article_ids if a < len(embeddings)]
        if valid_ids:
            profile = embeddings[valid_ids].mean(axis=0)
            norm = np.linalg.norm(profile)
            if norm > 0:
                profile = profile / norm
            scores = embeddings.dot(profile)
            read = set(article_ids)
            ranked = np.argsort(scores)[::-1]
            for idx in ranked:
                if int(idx) not in read:
                    cb_scores[int(idx)] = float(scores[idx])
                if len(cb_scores) >= 50:
                    break

    # Collaborative Filtering
    cf_scores = {}
    uid_mapped = user_map.get(int(user_id))
    if uid_mapped is not None:
        rec_ids, rec_scores = als_model.recommend(
            uid_mapped, user_item[uid_mapped], N=50, filter_already_liked_items=True
        )
        for idx, score in zip(rec_ids, rec_scores):
            aid = article_map_inv.get(str(int(idx)))
            if aid is not None:
                cf_scores[aid] = float(score)

    # Fallback cold start
    if not cb_scores and not cf_scores:
        return [{"article_id": r["article_id"], "score": r["score"], "source": "cold start"} for r in cold_start_articles]

    # Un seul modèle répond
    if not cf_scores:
        sorted_cb = sorted(cb_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        return [{"article_id": a, "score": round(s, 4), "source": "content-based"} for a, s in sorted_cb]

    if not cb_scores:
        sorted_cf = sorted(cf_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        return [{"article_id": a, "score": round(s, 4), "source": "collaboratif"} for a, s in sorted_cf]

    # Hybride
    cb_min, cb_max = min(cb_scores.values()), max(cb_scores.values())
    cf_min, cf_max = min(cf_scores.values()), max(cf_scores.values())
    cb_range = (cb_max - cb_min) or 1
    cf_range = (cf_max - cf_min) or 1

    hybrid = {}
    for aid in set(cb_scores.keys()) | set(cf_scores.keys()):
        cb_n = (cb_scores.get(aid, cb_min) - cb_min) / cb_range
        cf_n = (cf_scores.get(aid, cf_min) - cf_min) / cf_range
        hybrid[aid] = alpha * cf_n + (1 - alpha) * cb_n

    sorted_recs = sorted(hybrid.items(), key=lambda x: x[1], reverse=True)[:5]
    return [{"article_id": a, "score": round(s, 4), "source": "hybride"} for a, s in sorted_recs]

# Pré-calcul
print("Pré-calcul des recommandations...")
all_recs = {}
user_ids = list(user_articles.keys())
total = len(user_ids)

for i, uid in enumerate(user_ids):
    recs = recommend_hybrid(uid)
    all_recs[uid] = recs
    if (i + 1) % 10000 == 0 or (i + 1) == total:
        print(f"  {i+1}/{total} utilisateurs traités")

# Cold start par défaut
all_recs["_cold_start"] = [{"article_id": r["article_id"], "score": r["score"], "source": "cold start"} for r in cold_start_articles]

# Sauvegarde
output_path = os.path.join(OUT, "recommendations.json")
with open(output_path, "w") as f:
    json.dump(all_recs, f)

size_mo = os.path.getsize(output_path) / 1e6
print(f"\nFichier généré : {output_path}")
print(f"Taille : {size_mo:.1f} Mo")
print(f"Utilisateurs couverts : {len(all_recs) - 1:,} + cold start")
print(f"Exemple user 0 : {all_recs.get('0', 'N/A')}")