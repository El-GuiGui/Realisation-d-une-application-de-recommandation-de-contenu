import json
import os
import numpy as np
import joblib
from sklearn.preprocessing import normalize
from scipy.sparse import load_npz

# Chemins artefacts
MODEL_DIR = os.environ.get("MODEL_DIR", "/opt/ml/model")

# Variables globales (chargees une seule fois au cold start)
embeddings_cb = None
user_articles = None
article_popularity = None
als_model = None
user_item = None
user_map = None
article_map_inv = None
config = None


def load_models():
    """Charge tous les artefacts au premier appel."""
    global embeddings_cb, user_articles, article_popularity
    global als_model, user_item, user_map, article_map_inv, config

    cb_dir = os.path.join(MODEL_DIR, "models", "content_based")
    cf_dir = os.path.join(MODEL_DIR, "models", "collaborative")
    cfg_dir = os.path.join(MODEL_DIR, "models", "config")

    # Content-Based
    embeddings_cb = np.load(os.path.join(cb_dir, "embeddings_reduced.npy"))
    embeddings_cb = normalize(embeddings_cb, norm="l2", axis=1)

    with open(os.path.join(cb_dir, "user_articles.json"), "r") as f:
        user_articles = json.load(f)

    import pandas as pd
    pop_df = pd.read_parquet(os.path.join(cb_dir, "article_popularity.parquet"))
    article_popularity = pop_df.sort_values("nb_clicks", ascending=False).index.tolist()

    # Collaborative Filtering
    als_model = joblib.load(os.path.join(cf_dir, "als_model.joblib"))
    user_item = load_npz(os.path.join(cf_dir, "user_item_matrix.npz"))

    with open(os.path.join(cf_dir, "user_map.json"), "r") as f:
        user_map = {int(k): v for k, v in json.load(f).items()}
    with open(os.path.join(cf_dir, "article_map_inv.json"), "r") as f:
        article_map_inv = json.load(f)

    # Config
    with open(os.path.join(cfg_dir, "model_config.json"), "r") as f:
        config = json.load(f)


def recommend_cb(user_id, top_n=50):
    """Content-Based : cosinus sur embeddings."""
    article_ids = user_articles.get(str(user_id), [])
    if not article_ids:
        return {}
    valid_ids = [a for a in article_ids if a < len(embeddings_cb)]
    if not valid_ids:
        return {}
    profile = embeddings_cb[valid_ids].mean(axis=0)
    norm = np.linalg.norm(profile)
    if norm > 0:
        profile = profile / norm
    scores = embeddings_cb.dot(profile)
    read = set(article_ids)
    ranked = np.argsort(scores)[::-1]
    result = {}
    for idx in ranked:
        if int(idx) not in read:
            result[int(idx)] = float(scores[idx])
        if len(result) >= top_n:
            break
    return result


def recommend_cf(user_id, top_n=50):
    """Collaborative Filtering : ALS."""
    uid_mapped = user_map.get(int(user_id))
    if uid_mapped is None:
        return {}
    rec_ids, rec_scores = als_model.recommend(
        uid_mapped, user_item[uid_mapped], N=top_n, filter_already_liked_items=True
    )
    result = {}
    for idx, score in zip(rec_ids, rec_scores):
        aid = article_map_inv.get(str(int(idx)))
        if aid is not None:
            result[aid] = float(score)
    return result


def recommend_popularity(user_id, top_n=5):
    """Fallback : articles populaires non lus."""
    read = set(user_articles.get(str(user_id), []))
    recs = []
    for aid in article_popularity:
        if aid not in read:
            recs.append({"article_id": int(aid), "score": 0.0})
        if len(recs) >= top_n:
            break
    return recs


def recommend_hybrid(user_id, top_n=5):
    """Hybride : alpha * CF + (1-alpha) * CB, fallback popularite."""
    alpha = config.get("best_alpha", 0.6)

    cb_scores = recommend_cb(user_id, top_n=50)
    cf_scores = recommend_cf(user_id, top_n=50)

    # Fallback
    if not cb_scores and not cf_scores:
        return recommend_popularity(user_id, top_n)

    if not cf_scores:
        sorted_cb = sorted(cb_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return [{"article_id": aid, "score": round(s, 4)} for aid, s in sorted_cb]

    if not cb_scores:
        sorted_cf = sorted(cf_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return [{"article_id": aid, "score": round(s, 4)} for aid, s in sorted_cf]

    # Normalisation min-max + fusion
    cb_min, cb_max = min(cb_scores.values()), max(cb_scores.values())
    cf_min, cf_max = min(cf_scores.values()), max(cf_scores.values())
    cb_range = (cb_max - cb_min) or 1
    cf_range = (cf_max - cf_min) or 1

    all_articles = set(cb_scores.keys()) | set(cf_scores.keys())
    hybrid = {}
    for aid in all_articles:
        cb_n = (cb_scores.get(aid, cb_min) - cb_min) / cb_range
        cf_n = (cf_scores.get(aid, cf_min) - cf_min) / cf_range
        hybrid[aid] = alpha * cf_n + (1 - alpha) * cb_n

    sorted_recs = sorted(hybrid.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return [{"article_id": aid, "score": round(s, 4)} for aid, s in sorted_recs]


def lambda_handler(event, context):
    """Point d'entree AWS Lambda."""
    if embeddings_cb is None:
        load_models()

    if isinstance(event.get("body"), str):
        body = json.loads(event["body"])
    else:
        body = event.get("body", event)

    user_id = body.get("user_id")
    top_n = body.get("top_n", 5)

    if user_id is None:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": "user_id requis"})
        }

    recs = recommend_hybrid(int(user_id), top_n=int(top_n))

    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps({
            "user_id": int(user_id),
            "recommendations": recs,
            "model": "hybrid",
            "alpha": config.get("best_alpha", 0.6)
        })
    }
