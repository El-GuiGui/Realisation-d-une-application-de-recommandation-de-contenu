import numpy as np
import json
import os
import joblib
from sklearn.preprocessing import normalize

# Structure out/
#   models/content_based/   -> embeddings, user_articles, popularity, pca
#   models/collaborative/   -> als_model.joblib, matrice, mappings
#   models/config/          -> model_config, articles_metadata
#   figures/                -> graphes PNG

CB_DIR = "models/content_based"
CF_DIR = "models/collaborative"
CFG_DIR = "models/config"


def out_path(base, *parts):
    return os.path.join(base, *parts)


class ContentBasedRecommender:
    """Recommandation par similarite cosinus sur embeddings articles."""

    def __init__(self, embeddings, user_articles, article_popularity=None):
        self.embeddings = normalize(embeddings, norm="l2", axis=1)
        self.user_articles = user_articles
        self.article_popularity = article_popularity

    def build_user_profile(self, user_id):
        """Profil utilisateur = moyenne normalisee des embeddings lus."""
        article_ids = self.user_articles.get(str(user_id), [])
        if not article_ids:
            return None
        valid_ids = [aid for aid in article_ids if aid < len(self.embeddings)]
        if not valid_ids:
            return None
        profile = self.embeddings[valid_ids].mean(axis=0)
        norm = np.linalg.norm(profile)
        if norm > 0:
            profile = profile / norm
        return profile

    def recommend(self, user_id, top_n=5):
        """Top N articles par similarite cosinus avec le profil user."""
        profile = self.build_user_profile(user_id)
        if profile is None:
            return self._fallback_popularity(user_id, top_n)

        scores = self.embeddings.dot(profile)
        read_articles = set(self.user_articles.get(str(user_id), []))
        ranked = np.argsort(scores)[::-1]

        recs = []
        for idx in ranked:
            if int(idx) not in read_articles:
                recs.append({"article_id": int(idx), "score": round(float(scores[idx]), 4)})
            if len(recs) >= top_n:
                break
        return recs

    def similar_articles(self, article_id, top_n=5):
        """Articles les plus proches d'un article donne."""
        if article_id >= len(self.embeddings):
            return []
        query = self.embeddings[article_id]
        scores = self.embeddings.dot(query)
        scores[article_id] = -1
        top_indices = np.argsort(scores)[::-1][:top_n]
        return [{"article_id": int(i), "score": round(float(scores[i]), 4)} for i in top_indices]

    def _fallback_popularity(self, user_id, top_n=5):
        """Fallback : articles populaires non lus."""
        if self.article_popularity is None:
            return []
        read_articles = set(self.user_articles.get(str(user_id), []))
        recs = []
        for aid in self.article_popularity:
            if aid not in read_articles:
                recs.append({"article_id": int(aid), "score": 0.0})
            if len(recs) >= top_n:
                break
        return recs

    @classmethod
    def load(cls, base_dir):
        """Charge depuis out/models/content_based/."""
        cb_dir = out_path(base_dir, CB_DIR)
        embeddings = np.load(out_path(cb_dir, "embeddings_reduced.npy"))
        with open(out_path(cb_dir, "user_articles.json"), "r") as f:
            user_articles = json.load(f)

        pop_path = out_path(cb_dir, "article_popularity.parquet")
        article_popularity = None
        if os.path.exists(pop_path):
            import pandas as pd
            pop_df = pd.read_parquet(pop_path)
            article_popularity = pop_df.sort_values("nb_clicks", ascending=False).index.tolist()

        return cls(embeddings, user_articles, article_popularity)


class CollaborativeRecommender:
    """Recommandation par factorisation matricielle ALS (implicit)."""

    def __init__(self, model, user_item_matrix, user_map, article_map_inv):
        self.model = model
        self.user_item = user_item_matrix
        self.user_map = user_map
        self.article_map_inv = article_map_inv

    def recommend(self, user_id, top_n=5):
        """Top N articles par ALS."""
        uid_mapped = self.user_map.get(int(user_id))
        if uid_mapped is None:
            return []

        rec_ids, rec_scores = self.model.recommend(
            uid_mapped, self.user_item[uid_mapped],
            N=top_n, filter_already_liked_items=True
        )

        recs = []
        for idx, score in zip(rec_ids, rec_scores):
            aid = self.article_map_inv.get(str(int(idx)))
            if aid is not None:
                recs.append({"article_id": aid, "score": round(float(score), 4)})
        return recs

    @classmethod
    def load(cls, base_dir):
        """Charge depuis out/models/collaborative/."""
        from scipy.sparse import load_npz

        cf_dir = out_path(base_dir, CF_DIR)
        model = joblib.load(out_path(cf_dir, "als_model.joblib"))
        user_item = load_npz(out_path(cf_dir, "user_item_matrix.npz"))

        with open(out_path(cf_dir, "user_map.json"), "r") as f:
            user_map = {int(k): v for k, v in json.load(f).items()}
        with open(out_path(cf_dir, "article_map_inv.json"), "r") as f:
            article_map_inv = json.load(f)

        return cls(model, user_item, user_map, article_map_inv)


class HybridRecommender:
    """Recommandation hybride : alpha * CF + (1-alpha) * CB."""

    def __init__(self, cb_model, cf_model, alpha=0.6):
        self.cb = cb_model
        self.cf = cf_model
        self.alpha = alpha

    def recommend(self, user_id, top_n=5):
        """Fusion des scores CF et CB normalises."""
        recs_cb = self.cb.recommend(user_id, top_n=50)
        recs_cf = self.cf.recommend(user_id, top_n=50)

        if not recs_cf:
            return recs_cb[:top_n]
        if not recs_cb:
            return recs_cf[:top_n]

        cb_scores = {r["article_id"]: r["score"] for r in recs_cb}
        cf_scores = {r["article_id"]: r["score"] for r in recs_cf}

        cb_min, cb_max = min(cb_scores.values()), max(cb_scores.values())
        cf_min, cf_max = min(cf_scores.values()), max(cf_scores.values())
        cb_range = cb_max - cb_min if cb_max > cb_min else 1
        cf_range = cf_max - cf_min if cf_max > cf_min else 1

        all_articles = set(cb_scores.keys()) | set(cf_scores.keys())
        hybrid = {}
        for aid in all_articles:
            cb_norm = (cb_scores.get(aid, cb_min) - cb_min) / cb_range
            cf_norm = (cf_scores.get(aid, cf_min) - cf_min) / cf_range
            hybrid[aid] = self.alpha * cf_norm + (1 - self.alpha) * cb_norm

        sorted_recs = sorted(hybrid.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return [{"article_id": aid, "score": round(s, 4)} for aid, s in sorted_recs]

    @classmethod
    def load(cls, base_dir, alpha=None):
        """Charge les deux modeles et le config."""
        cb = ContentBasedRecommender.load(base_dir)
        cf = CollaborativeRecommender.load(base_dir)

        if alpha is None:
            cfg_path = out_path(base_dir, CFG_DIR, "model_config.json")
            if os.path.exists(cfg_path):
                with open(cfg_path, "r") as f:
                    config = json.load(f)
                alpha = config.get("best_alpha", 0.6)
            else:
                alpha = 0.6

        return cls(cb, cf, alpha)


if __name__ == "__main__":
    OUT = "../out/"

    print("Test Content-Based :")
    cb = ContentBasedRecommender.load(OUT)
    print(f"  Embeddings : {cb.embeddings.shape}")
    recs = cb.recommend("0", top_n=5)
    for r in recs:
        print(f"  article_id={r['article_id']} | score={r['score']}")

    print("\nTest Collaborative Filtering :")
    try:
        cf = CollaborativeRecommender.load(OUT)
        recs = cf.recommend(0, top_n=5)
        for r in recs:
            print(f"  article_id={r['article_id']} | score={r['score']}")
    except Exception as e:
        print(f"  Non disponible : {e}")

    print("\nTest Hybride :")
    try:
        hybrid = HybridRecommender.load(OUT)
        recs = hybrid.recommend(0, top_n=5)
        for r in recs:
            print(f"  article_id={r['article_id']} | score={r['score']}")
    except Exception as e:
        print(f"  Non disponible : {e}")
