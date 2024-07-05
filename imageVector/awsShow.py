from transformers import CLIPProcessor, CLIPModel
import chromadb
import matplotlib.pyplot as plt
import torch
from PIL import Image
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
# chroma 클라이언트 초기화
client = chromadb.HttpClient(
    host = "16.171.169.62",
    port = 8000,  # ChromaDB 서버의 포트 (기본값은 8000)
    ssl=False,
    headers={
        "X-Api-Key": "NIwdUnNFOo73SKvW+P6OYQguw1mNJtNLO0+2pS07"
    }
)

# 컬렉션 생성 (혹은 기존 컬렉션 가져오기)
#collection = client.get_collection(name="Conversation_images")
collection = client.get_collection(name="Conversation_images_clip14")

# 검색하고 싶은 단어
query_text = "ocean"
inputs = processor(text=query_text, return_tensors="pt", padding=True).to(device)
with torch.no_grad():
    text_features = model.get_text_features(**inputs)
query_vector = text_features.squeeze().cpu().tolist()

# 유사한 이미지 검색
results = collection.query(
    query_embeddings=[query_vector],
    n_results=5  # 상위 5개 결과 반환
)

# # 결과 출력
# for id, distance in zip(results["ids"][0], results["distances"][0]):
#     print(f"Image: {id}, Distance: {distance}")

# 결과 출력 및 이미지 표시
plt.figure(figsize=(20, 4))
for i, (id, distance) in enumerate(zip(results["ids"][0], results["distances"][0])):
    print(f"Image: {id}, Distance: {distance}")
    
    # 이미지 파일 경로 (이미지 저장 위치에 따라 수정 필요)
    image_path = os.path.join("image/Conversation", id)
    
    # 이미지 열기 및 표시
    img = Image.open(image_path)
    plt.subplot(1, 5, i+1)
    plt.imshow(img)
    plt.title(f"{id}\nDistance: {distance:.4f}")
    plt.axis('off')

plt.tight_layout()
plt.show()