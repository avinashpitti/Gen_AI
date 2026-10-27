import os
import boto3
from dotenv import load_dotenv

load_dotenv()

s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION", "ap-south-1")
)

print("--- Listing Your S3 Buckets ---")
try:
    buckets = s3.list_buckets()
    for b in buckets.get("Buckets", []):
        print(f"Bucket: {b['Name']}")
        
        # List files inside this bucket
        objects = s3.list_objects_v2(Bucket=b['Name'])
        for item in objects.get("Contents", []):
            print(f"  └── File: {item['Key']}")
except Exception as e:
    print(f"Error: {e}")