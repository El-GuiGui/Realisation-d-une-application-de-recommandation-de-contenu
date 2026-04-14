import streamlit as st
import json
import os
import pandas as pd
import numpy as np
import requests

# Config
OUT = os.path.join(os.path.dirname(__file__), "..", "out")
DATA_RAW = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
CB_DIR = os.path.join(OUT, "models", "content_based")
CF_DIR = os.path.join(OUT, "models", "collaborative")
CFG_DIR = os.path.join(OUT, "models", "config")

# URL local
# API_URL = os.environ.get("API_URL", "")

# URL pour lambda 
API_URL = "https://u6ej86v531.execute-api.eu-north-1.amazonaws.com/prod/recommend"


st.set_page_config(page_title="My Content - Recommandation", layout="wide")

st.title("My Content - Systeme de recommandation d'articles")

if API_URL:
    st.caption(f"Mode API Lambda : {API_URL}")
else:
    st.caption("Mode local (modeles charges en memoire)")


@st.cache_resource
def load_metadata():
    """Charge les metadonnees et l'historique"""
    articles = pd.read_csv(os.path.join(DATA_RAW, "articles_metadata.csv"))
    with open(os.path.join(CB_DIR, "user_articles.json"), "r") as f:
        user_articles = json.load(f)
    with open(os.path.join(CFG_DIR, "model_config.json"), "r") as f:
        config = json.load(f)
    return articles, user_articles, config


@st.cache_resource
def load_local_models():
    """Charge les modeles pour le mode local (sans Lambda)."""
    import joblib
    from sklearn.preprocessing import normalize
    from scipy.sparse import load_npz

    embeddings = np.load(os.path.join(CB_DIR, "embeddings_reduced.npy"))
    embeddings = normalize(embeddings, norm="l2", axis=1)

    als_model = joblib.load(os.path.join(CF_DIR, "als_model.joblib"))
    user_item = load_npz(os.path.join(CF_DIR, "user_item_matrix.npz"))

    with open(os.path.join(CF_DIR, "user_map.json"), "r") as f:
        user_map = {int(k): v for k, v in json.load(f).items()}
    with open(os.path.join(CF_DIR, "article_map_inv.json"), "r") as f:
        article_map_inv = json.load(f)

    pop_df = pd.read_parquet(os.path.join(CB_DIR, "article_popularity.parquet"))
    article_popularity = pop_df.sort_values("nb_clicks", ascending=False).index.tolist()

    cs_df = pd.read_parquet(os.path.join(CB_DIR, "cold_start_ranking.parquet"))
    cold_start_ranking = cs_df.sort_values("cold_start_score", ascending=False)

    return {
        "embeddings": embeddings,
        "als_model": als_model,
        "user_item": user_item,
        "user_map": user_map,
        "article_map_inv": article_map_inv,
        "article_popularity": article_popularity,
        "cold_start_ranking": cold_start_ranking,
    }


def recommend_via_api(user_id, top_n=5):
    """Appel a la Lambda via API Gateway."""
    try:
        response = requests.post(
            API_URL,
            json={"user_id": int(user_id), "top_n": top_n},
            timeout=30
        )
        data = response.json()
        if isinstance(data.get("body"), str):
            data = json.loads(data["body"])
        recs = data.get("recommendations", [])
        source = data.get("source", "api")
        return [{"article_id": a, "score": 0, "source": source} for a in recs] if recs and isinstance(recs[0], int) else recs
    except Exception as e:
        st.error(f"Erreur API : {e}")
        return []


def recommend_local(user_id, user_articles, models, config, top_n=5):
    """Recommandation locale hybride."""
    alpha = config.get("best_alpha", 1.0)
    embeddings = models["embeddings"]

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
    uid_mapped = models["user_map"].get(int(user_id))
    if uid_mapped is not None:
        rec_ids, rec_scores_raw = models["als_model"].recommend(
            uid_mapped, models["user_item"][uid_mapped], N=50, filter_already_liked_items=True
        )
        for idx, score in zip(rec_ids, rec_scores_raw):
            aid = models["article_map_inv"].get(str(int(idx)))
            if aid is not None:
                cf_scores[aid] = float(score)

    # Cold start
    if not cb_scores and not cf_scores:
        read = set(user_articles.get(str(user_id), []))
        recs = []
        for _, row in models["cold_start_ranking"].iterrows():
            aid = int(row["article_id"])
            if aid not in read:
                recs.append({"article_id": aid, "score": round(float(row["cold_start_score"]), 4), "source": "cold start"})
            if len(recs) >= top_n:
                break
        return recs

    if not cf_scores:
        sorted_cb = sorted(cb_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return [{"article_id": a, "score": round(s, 4), "source": "content-based"} for a, s in sorted_cb]
    if not cb_scores:
        sorted_cf = sorted(cf_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return [{"article_id": a, "score": round(s, 4), "source": "collaborative"} for a, s in sorted_cf]

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

    sorted_recs = sorted(hybrid.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return [{"article_id": a, "score": round(s, 4), "source": "hybride"} for a, s in sorted_recs]


# Chargement
with st.spinner("Chargement..."):
    articles, user_articles, config = load_metadata()
    if not API_URL:
        models = load_local_models()

user_ids = sorted([int(k) for k in user_articles.keys()])

# Sidebar
st.sidebar.header("Parametres")

simulate_new = st.sidebar.checkbox("Simuler un nouvel utilisateur (cold start)")

if simulate_new:
    selected_user = -1
    st.sidebar.markdown("**Nouvel utilisateur** : aucun historique")
else:
    selected_user = st.sidebar.selectbox(
        "Identifiant utilisateur",
        options=user_ids,
        index=0
    )
    user_history = user_articles.get(str(selected_user), [])
    st.sidebar.markdown(f"**Articles lus** : {len(user_history)}")
    if user_history:
        hist_cats = articles[articles["article_id"].isin(user_history)]["category_id"].value_counts().head(5)
        st.sidebar.markdown("**Top categories lues** :")
        for cat_id, count in hist_cats.items():
            st.sidebar.markdown(f"- Categorie {cat_id} : {count} articles")

top_n = st.sidebar.slider("Nombre d'articles", min_value=1, max_value=20, value=5)

# Recommandations
st.subheader(f"Recommandations pour l'utilisateur {selected_user}")

if st.button("Lancer la recommandation"):
    with st.spinner("Calcul en cours..."):
        if API_URL:
            recs = recommend_via_api(selected_user, top_n=top_n)
        else:
            recs = recommend_local(selected_user, user_articles, models, config, top_n=top_n)

    if not recs:
        st.warning("Aucune recommandation disponible pour cet utilisateur.")
    else:
        rows = []
        for i, r in enumerate(recs, 1):
            aid = r["article_id"] if isinstance(r, dict) else r
            meta = articles[articles["article_id"] == aid]
            rows.append({
                "Rang": i,
                "Article ID": aid,
                "Score": r.get("score", "-") if isinstance(r, dict) else "-",
                "Categorie": meta["category_id"].values[0] if len(meta) > 0 else "-",
                "Mots": meta["words_count"].values[0] if len(meta) > 0 else "-",
                "Source": r.get("source", "-") if isinstance(r, dict) else "-",
            })

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Historique
        if not simulate_new:
            user_history = user_articles.get(str(selected_user), [])
            if user_history:
                st.subheader("Derniers articles lus par cet utilisateur")
                last_read = user_history[-10:]
                read_rows = []
                for aid in reversed(last_read):
                    meta = articles[articles["article_id"] == aid]
                    read_rows.append({
                        "Article ID": aid,
                        "Categorie": meta["category_id"].values[0] if len(meta) > 0 else "-",
                        "Mots": meta["words_count"].values[0] if len(meta) > 0 else "-",
                    })
                st.dataframe(pd.DataFrame(read_rows), use_container_width=True, hide_index=True)

# Footer
st.markdown("---")
mode = "API Lambda" if API_URL else "Local"
st.caption(f"My Content MVP | Mode : {mode} | Alpha : {config.get('best_alpha', '-')} | {len(user_ids):,} utilisateurs | {len(articles):,} articles")
