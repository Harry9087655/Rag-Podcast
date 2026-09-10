from pathlib import Path
from openai import OpenAI 
from dotenv import load_dotenv 
import os
import logging 
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()


with open(Path(__name__).resolve().parent / "funasr_aligned_test.json", 
          encoding="utf8") as f:
    data = json.load(f)
text = data[0]['text']
print(type(text))


embedd_api = os.getenv("EMBEDDING_API_KEY")

openai = OpenAI(
    api_key=embedd_api,
    base_url="https://api.deepinfra.com/v1/openai",
)
input = ["您的半拿铁周刊", "请查收"]
embeddings = openai.embeddings.create(
    model="BAAI/bge-m3",
    input=text,
    encoding_format="float"
)
print("----------Embedding---------")
logger.info(type(embeddings))
print("----------data---------")
for i in range(len(embeddings.data)):
    logger.info(f"length of {i}th data: {len(embeddings.data[i])}")
    logger.info(f"embeddings' index: {embeddings.data[i].index}")
#logger.info(embeddings.data)