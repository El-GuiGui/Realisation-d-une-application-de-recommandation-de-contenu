# My Content - MVP Systeme de Recommandation

Projet de recommandation d'articles pour la start-up My Content.
L'application recommande 5 articles pertinents par utilisateur.

## Arborescence

```
proj10/
├── notebooks/              # EDA, modeles, evaluation
│   ├── 01_eda_exploration.ipynb
│   ├── 02_modele_content_based_cosinus.ipynb
│   ├── 03_modele_collaboratif_als.ipynb
│   └── 04_evaluation_et_hybride.ipynb
├── scripts/
│   ├── models.py           # Classes CB + CF + Hybride
│   └── precompute.py       # Pre-calcul des recommandations
├── app/
│   └── app_streamlit.py    # Application Streamlit
├── aws/
│   ├── handler.py          # AWS Lambda handler
│   ├── DEPLOIEMENT.md      # Guide deploiement pas à pas
│   └── requirements.txt
├── data/
│   ├── raw/                # Donnees brutes (gitignored)
│   └── processed/          # Donnees traitees (gitignored)
├── out/                    # Artefacts modeles (gitignored)
├── tests/
├── presentation/
├── requirements.txt
└── .gitignore
```

## Donnees

Source : Globo.com - News Portal User Interactions
https://www.kaggle.com/datasets/gspmoreira/news-portal-user-interactions-by-globocom

- 2 988 181 interactions (clics)
- 322 897 utilisateurs
- 46 033 articles cliques sur 364 047 en base
- Periode : octobre-novembre 2017

## Modeles

1. **Content-Based** : similarite cosinus sur les embeddings articles (250 dimensions reduites a 50 par ACP)
2. **Collaborative Filtering** : factorisation matricielle ALS via la librairie implicit
3. **Hybride** : combinaison ponderee CB + CF (alpha optimise par grid search)
4. **Cold Start** : score recence + popularite pour les nouveaux utilisateurs

## Resultats

Evaluation sur 5 000 utilisateurs (split temporel : dernier clic = test) :

| Modele                        | Hit Ratio | Mean Reciprocal Rank | Couverture |
| ----------------------------- | --------- | -------------------- | ---------- |
| Popularite pure               | 0.0170    | 0.0061               | 28         |
| Cold Start (recence + pop)    | 0.0170    | 0.0061               | 28         |
| Content-Based (cosinus)       | 0.0026    | 0.0018               | 10 194     |
| Collaborative Filtering (ALS) | 0.2628    | 0.2086               | 206        |

## Architecture

Le systeme de recommandation est deploye en serverless sur AWS :

1. Les recommandations sont pre-calculees pour chaque utilisateur (script `precompute.py`)
2. Le fichier JSON resultant est stocke sur AWS S3
3. Une AWS Lambda charge ce JSON et repond aux requetes (user_id -> 5 articles)
4. Une API Gateway expose la Lambda via une URL REST
5. L'application Streamlit appelle cette API et affiche les resultats

Le guide de deploiement complet est dans `aws/DEPLOIEMENT.md`.

## Installation et lancement

### Pre-requis

```bash
pip install -r requirements.txt
```

### Execution des notebooks (dans l'ordre)

```bash
cd notebooks
jupyter notebook
# Executer : 01 -> 02 -> 03 -> 04
```

### Pre-calcul des recommandations

```bash
cd proj10
python scripts/precompute.py
```

### Lancement de l'application

En local ou en passant avec Lambda (apres deploiement AWS) :

```bash
streamlit run app/app_streamlit.py
```

Pour choisir l'un ou l'autre, modifier dans app/app_streamlit.py :

# URL local

'''
API_URL = os.environ.get("API_URL", "")
'''

# URL pour lambda

'''
API_URL = "https://xxxxxxxxx.execute-api.eu-north-1.amazonaws.com/prod/recommend"
'''

## Sources :

1. https://github.com/kskaran94/Content_Based_Recommender
2. https://github.com/recommenders-team/recommenders/blob/main/examples/02_model_collaborative_filtering/als_deep_dive.ipynb
3. https://arxiv.org/abs/2001.04831
4. https://actsusanli.medium.com/building-a-recommender-system-with-implicit-feedback-datasets-using-alternating-least-squares-64d4f5ba3c57
5. https://github.com/huangy22/NewsRecommender
6. https://medium.com/hacktive-devs/recommender-system-made-easy-with-scikit-surprise-569cbb689824
7. https://arxiv.org/abs/1808.00076
8. https://realpython.com/build-recommendation-engine-collaborative-filtering/
9. https://github.com/benfred/implicit
10. https://www.kaggle.com/code/gspmoreira/recommender-systems-in-python-101
11. https://github.com/gabrielspmoreira/chameleon_recsys

Reference metriques : Benjamin Wang (2021), Ranking Evaluation Metrics for Recommender Systems, Towards Data Science.
