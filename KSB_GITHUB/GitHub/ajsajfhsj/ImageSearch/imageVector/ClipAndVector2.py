import torch
from PIL import Image
from transformers import ChineseCLIPProcessor, ChineseCLIPModel
import chromadb
import os
from tqdm import tqdm

# OFA-Sys/chinese-clip-vit-base-patch16 모델 및 프로세서 로드
device = "cuda" if torch.cuda.is_available() else "cpu"
model = ChineseCLIPModel.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16").to(device)
processor = ChineseCLIPProcessor.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")

# chroma 클라이언트 초기화
client = chromadb.PersistentClient(path="chroma_db")

# 컬렉션 생성 (혹은 기존 컬렉션 가져오기)
collection = client.create_collection(name="image_collection_v3")

# 이미지 폴더 경로 (절대 경로로 수정)
current_dir = os.getcwd()
image_folder = os.path.join(current_dir, "image", "ascs")

# 이미지 폴더 경로 출력
print(f"이미지 폴더 경로: {image_folder}")

# 이미지 파일 목록 가져오기
try:
    image_files = [f for f in os.listdir(image_folder) if f.endswith(("jpg", "jpeg", "png"))]
    total_images = len(image_files)
    print(f"폴더 내 이미지 갯수: {total_images}")
except FileNotFoundError as e:
    print(f"Error: {str(e)}")
    raise

# 이미지 처리 및 DB에 저장
for filename in tqdm(image_files, desc="이미지 처리 중"):
    image_path = os.path.join(image_folder, filename)
    try:
        # 이미지 로드 및 전처리
        image = Image.open(image_path)
        inputs = processor(images=image, return_tensors="pt", padding=True).to(device)

        # 이미지 특징 추출
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)

        # 벡터를 리스트로 변환
        image_vector = image_features.squeeze().cpu().tolist()

        # Chroma DB에 저장
        collection.add(
            embeddings=[image_vector],
            ids=[filename],
            metadatas=[{"filename": filename}]
        )
    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")

print("모든 이미지가 처리되어 DB에 저장되었습니다.")

