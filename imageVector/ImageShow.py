from transformers import CLIPProcessor, CLIPModel
import chromadb
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
# chroma 클라이언트 초기화
client = chromadb.PersistentClient(path="chroma_db")

# 컬렉션 생성 (혹은 기존 컬렉션 가져오기)
collection = client.get_collection(name="test_images")

query_text = "a person who is eating fruit"
inputs = processor(text=query_text, return_tensors="pt", padding=True).to(device)
with torch.no_grad():
    text_features = model.get_text_features(**inputs)
query_vector = text_features.squeeze().cpu().tolist()

# 유사한 이미지 검색
results = collection.query(
    query_embeddings=[query_vector],
    n_results=5  # 상위 5개 결과 반환
)

# 결과 출력
for id, distance in zip(results["ids"][0], results["distances"][0]):
    print(f"Image: {id}, Distance: {distance}")