# My Content - MVP Systeme de Recommandation

Projet de recommandation d'articles pour la start-up My Content.
L'application recommande 5 articles pertinents par utilisateur.

## Arborescence

```
proj10/
├── notebooks/          # EDA, modeles, evaluation
├── data/
│   ├── raw/            # Donnees brutes (gitignored)
│   └── processed/      # Donnees traitees (gitignored)
├── scripts/            # Classes et fonctions reutilisables
├── app/                # Application Streamlit
├── aws/                # AWS Lambda handler
├── out/                # Artefacts modeles, figures
├── tests/              # Tests unitaires
└── presentation/       # Slides PDF
```

## Donnees

Source : Globo.com - News Portal User Interactions
https://www.kaggle.com/datasets/gspmoreira/news-portal-user-interactions-by-globocom

## Modeles

1. Content-Based : similarite cosinus sur embeddings articles
2. Collaborative Filtering : factorisation matricielle ALS (implicit)
3. Hybride : combinaison ponderee des deux approches
4. Cold Start : recence + popularite (fallback nouveaux utilisateurs)

## Lancement

```bash
pip install -r requirements.txt
streamlit run app/app_streamlit.py
```

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
