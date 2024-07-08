import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import chromadb
import os
from tqdm import tqdm
from chromadb.utils import embedding_functions

# CLIP 모델 및 프로세서 로드
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K").to(device)
processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")

# 임베딩 함수 정의
class CLIPEmbeddingFunction(embedding_functions.EmbeddingFunction):
    def __init__(self, model, processor, device):
        self.model = model
        self.processor = processor
        self.device = device

    def __call__(self, images):
        inputs = self.processor(images=images, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
        return image_features.cpu().numpy()

# 임베딩 함수 생성
clip_ef = CLIPEmbeddingFunction(model, processor, device)

# chroma 클라이언트 초기화
client = chromadb.PersistentClient(path="chroma_db")

# 컬렉션 생성 (혹은 기존 컬렉션 가져오기)
collection = client.create_collection(
    name="laion_clip14_1024",
    embedding_function=clip_ef,
)

# 이미지 폴더 경로
image_folder = "image/Conversation"

# 이미지 파일 목록 가져오기
image_files = [f for f in os.listdir(image_folder) if f.endswith(("jpg", "jpeg", "png", "JPG"))]
total_images = len(image_files)

print(f"폴더 내 이미지 갯수: {total_images}")

# 이미지 처리 및 DB에 저장
success_count = 0
failed_count = 0

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
            documents=[image_path],
            ids=[filename],
            metadatas=[{"filename": filename}]
        )
        success_count += 1
    except Exception as e:
        failed_count += 1
        print(f"Error processing {filename}: {str(e)}")

print(f"성공적으로 처리된 이미지: {success_count}, 실패한 이미지: {failed_count}")
print("모든 이미지가 처리되어 DB에 저장되었습니다.")
