import streamlit as st
import json
import os
import sys
import pandas as pd
import numpy as np
import joblib

# Config chemins
OUT = os.path.join(os.path.dirname(__file__), "..", "out")
DATA_RAW = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
CB_DIR = os.path.join(OUT, "models", "content_based")
CF_DIR = os.path.join(OUT, "models", "collaborative")
CFG_DIR = os.path.join(OUT, "models", "config")

st.set_page_config(page_title="My Content - Recommandation", layout="wide")

st.title("My Content - Systeme de recommandation d'articles")
st.caption("MVP - Recommandation hybride (Content-Based + Collaborative Filtering)")


@st.cache_resource
def load_all_models():
    """Charge les modeles une seule fois."""
    from sklearn.preprocessing import normalize
    from scipy.sparse import load_npz

    # Content-Based
    embeddings = np.load(os.path.join(CB_DIR, "embeddings_reduced.npy"))
    embeddings = normalize(embeddings, norm="l2", axis=1)

    with open(os.path.join(CB_DIR, "user_articles.json"), "r") as f:
        user_articles = json.load(f)

    pop_df = pd.read_parquet(os.path.join(CB_DIR, "article_popularity.parquet"))
    article_popularity = pop_df.sort_values("nb_clicks", ascending=False).index.tolist()

    # Collaborative Filtering
    als_model = joblib.load(os.path.join(CF_DIR, "als_model.joblib"))
    user_item = load_npz(os.path.join(CF_DIR, "user_item_matrix.npz"))

    with open(os.path.join(CF_DIR, "user_map.json"), "r") as f:
        user_map = {int(k): v for k, v in json.load(f).items()}
    with open(os.path.join(CF_DIR, "article_map_inv.json"), "r") as f:
        article_map_inv = json.load(f)

    # Config
    with open(os.path.join(CFG_DIR, "model_config.json"), "r") as f:
        config = json.load(f)

    # Metadata articles
    articles = pd.read_csv(os.path.join(DATA_RAW, "articles_metadata.csv"))

    return {
        "embeddings": embeddings,
        "user_articles": user_articles,
        "article_popularity": article_popularity,
        "als_model": als_model,
        "user_item": user_item,
        "user_map": user_map,
        "article_map_inv": article_map_inv,
        "config": config,
        "articles": articles,
    }


def recommend_hybrid(models, user_id, top_n=5):
    """Recommandation hybride."""
    alpha = models["config"].get("best_alpha", 0.6)
    embeddings = models["embeddings"]
    user_articles = models["user_articles"]
    als_model = models["als_model"]
    user_item = models["user_item"]
    user_map = models["user_map"]
    article_map_inv = models["article_map_inv"]

    # Content-Based
    cb_scores = {}
    article_ids = user_articles.get(str(user_id), [])
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
        rec_ids, rec_scores_raw = als_model.recommend(
            uid_mapped, user_item[uid_mapped], N=50, filter_already_liked_items=True
        )
        for idx, score in zip(rec_ids, rec_scores_raw):
            aid = article_map_inv.get(str(int(idx)))
            if aid is not None:
                cf_scores[aid] = float(score)

    # Fallback popularite
    if not cb_scores and not cf_scores:
        read = set(user_articles.get(str(user_id), []))
        recs = []
        for aid in models["article_popularity"]:
            if aid not in read:
                recs.append({"article_id": int(aid), "score": 0.0, "source": "popularite"})
            if len(recs) >= top_n:
                break
        return recs

    if not cf_scores:
        sorted_cb = sorted(cb_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return [{"article_id": a, "score": round(s, 4), "source": "content-based"} for a, s in sorted_cb]

    if not cb_scores:
        sorted_cf = sorted(cf_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return [{"article_id": a, "score": round(s, 4), "source": "collaborative"} for a, s in sorted_cf]

    # Normalisation + fusion
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
    return [{"article_id": a, "score": round(s, 4), "source": "hybride"} for a, s in sorted_recs]


# Chargement
with st.spinner("Chargement des modeles..."):
    models = load_all_models()

user_ids = sorted([int(k) for k in models["user_articles"].keys()])

# Sidebar
st.sidebar.header("Parametres")

selected_user = st.sidebar.selectbox(
    "Identifiant utilisateur",
    options=user_ids,
    index=0
)

if st.sidebar.checkbox("Simuler un nouvel utilisateur (sans historique)"):
    selected_user = -1  # ID bidon


top_n = st.sidebar.slider("Nombre d'articles", min_value=1, max_value=20, value=5)

# Infos user
user_history = models["user_articles"].get(str(selected_user), [])
st.sidebar.markdown(f"**Articles lus** : {len(user_history)}")

if user_history:
    hist_cats = models["articles"][models["articles"]["article_id"].isin(user_history)]["category_id"].value_counts().head(5)
    st.sidebar.markdown("**Top categories lues** :")
    for cat_id, count in hist_cats.items():
        st.sidebar.markdown(f"- Categorie {cat_id} : {count} articles")

# Recommandations
st.subheader(f"Recommandations pour l'utilisateur {selected_user}")

if st.button("Lancer la recommandation"):
    with st.spinner("Calcul en cours..."):
        recs = recommend_hybrid(models, selected_user, top_n=top_n)

    if not recs:
        st.warning("Aucune recommandation disponible pour cet utilisateur.")
    else:
        rows = []
        for i, r in enumerate(recs, 1):
            meta = models["articles"][models["articles"]["article_id"] == r["article_id"]]
            rows.append({
                "Rang": i,
                "Article ID": r["article_id"],
                "Score": r["score"],
                "Categorie": meta["category_id"].values[0] if len(meta) > 0 else "-",
                "Mots": meta["words_count"].values[0] if len(meta) > 0 else "-",
                "Source": r["source"],
            })

        df_recs = pd.DataFrame(rows)
        st.dataframe(df_recs, use_container_width=True, hide_index=True)

        # Derniers articles lus
        st.subheader("Derniers articles lus par cet utilisateur")
        last_read = user_history[-10:] if len(user_history) > 10 else user_history
        read_rows = []
        for aid in reversed(last_read):
            meta = models["articles"][models["articles"]["article_id"] == aid]
            read_rows.append({
                "Article ID": aid,
                "Categorie": meta["category_id"].values[0] if len(meta) > 0 else "-",
                "Mots": meta["words_count"].values[0] if len(meta) > 0 else "-",
            })
        st.dataframe(pd.DataFrame(read_rows), use_container_width=True, hide_index=True)

# Footer
st.markdown("---")
st.caption(f"My Content MVP | Modele hybride (alpha={models['config'].get('best_alpha', 0.6)}) | {len(user_ids):,} utilisateurs | {len(models['articles']):,} articles")
