# Deploiement AWS Lambda - Guide etape par etape

## Principe

Les recommandations sont pre-calculees en local (script precompute.py).
Le resultat est un fichier JSON stocke sur S3.
La Lambda fait un simple lookup dans ce JSON : user_id -> 5 articles.
Pas de dependances lourdes, pas de Docker etc.

## Pre-requis

1. Creer un compte AWS : https://aws.amazon.com/free
2. Avoir execute les 4 notebooks et le script precompute.py

## Etape 1 : Creer le bucket S3

1. Console S3 : https://s3.console.aws.amazon.com
2. "Creer un compartiment"
3. Nom : mycontent-models
4. Region : eu-west-1
5. Creer
6. Ouvrir le bucket, "Charger", uploader le fichier out/recommendations.json

## Etape 2 : Creer la Lambda

1. Console Lambda : https://console.aws.amazon.com/lambda
2. "Creer une fonction" > "Creer a partir de zero"
3. Nom : mycontent-recommend
4. Runtime : Python 3.11
5. "Creer la fonction"
6. Dans l'editeur de code, coller le contenu de aws/handler.py
7. Configuration > General : timeout = 30 secondes, memoire = 256 Mo
8. Configuration > Variables d'environnement : S3_BUCKET = mycontent-models
9. Configuration > Permissions : cliquer sur le role, ajouter la strategie AmazonS3ReadOnlyAccess

## Etape 3 : Tester la Lambda

1. Onglet "Test" dans la console Lambda
2. Creer un evenement de test :
   {"body": "{\"user_id\": 42}"}
3. Cliquer "Test"
4. Verifier que la reponse contient 5 article_id

## Etape 4 : Creer l'API Gateway

1. Console API Gateway : https://console.aws.amazon.com/apigateway
2. "Creer une API" > "API REST" > "Nouveau"
3. Nom : mycontent-api
4. Actions > "Creer une ressource" : /recommend
5. Actions > "Creer une methode" > POST sur /recommend
6. Integration : Fonction Lambda > mycontent-recommend
7. Actions > "Deployer l'API" > Nouvelle etape : prod
8. Noter l'URL affichee

## Etape 5 : Tester depuis un terminal

curl -X POST https://xxxxx.execute-api.eu-west-1.amazonaws.com/prod/recommend \
 -H "Content-Type: application/json" \
 -d '{"user_id": 42}'

## Etape 6 : Connecter Streamlit

Remplacer l'appel local par l'appel API dans app_streamlit.py.
L'URL de l'API Gateway est configurable via variable d'environnement.

## Couts

- Lambda free tier : 1 million de requetes/mois
- S3 free tier : xx Go stockage
- API Gateway free tier : 1 million d'appels/mois
- Premier appel lent (cold start 2-5s), les suivants rapides (<200ms)
