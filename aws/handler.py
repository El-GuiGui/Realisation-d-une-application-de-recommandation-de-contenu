import json
import os

# Variables globales
recommendations = None
S3_BUCKET = os.environ.get("S3_BUCKET", "mycontent-models")


def load_recommendations():
    """Charge le JSON pre-calcule depuis S3 ou local."""
    global recommendations

    local_path = "/tmp/recommendations.json"

    if not os.path.exists(local_path):
        import boto3
        s3 = boto3.client("s3")
        print("Telechargement recommendations.json depuis S3...")
        s3.download_file(S3_BUCKET, "recommendations.json", local_path)

    with open(local_path, "r") as f:
        recommendations = json.load(f)

    print(f"Recommandations chargees : {len(recommendations) - 1} users")


def lambda_handler(event, context):
    """Point d'entree AWS Lambda."""
    # Chargement au premier appel
    if recommendations is None:
        load_recommendations()

    # Parsing requete
    if isinstance(event.get("body"), str):
        body = json.loads(event["body"])
    else:
        body = event.get("body", event)

    user_id = body.get("user_id")
    if user_id is None:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": "user_id requis"})
        }

    # Lookup
    recs = recommendations.get(str(user_id))
    if recs is None:
        recs = recommendations.get("_cold_start", [])
        source = "cold_start"
    else:
        source = "hybrid"

    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps({
            "user_id": int(user_id),
            "recommendations": recs,
            "source": source
        })
    }
