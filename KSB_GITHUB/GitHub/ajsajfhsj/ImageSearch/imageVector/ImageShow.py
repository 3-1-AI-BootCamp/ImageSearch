import os
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import chromadb
from tqdm import tqdm
import numpy as np

# 사용할 CLIP 모델 목록
models_info = [
    {"name": "UCSC-VLAA/ViT-L-16-HTxt-Recap-CLIP", "db_path": "chroma_db/UCSC-VLAA_ViT-L-16-HTxt-Recap-CLIP"}
]

# 이미지 폴더 경로
image_folder = r'C:\Users\admin\Desktop\ascs'

# 테스트할 이미지 파일 목록 가져오기 (최대 1000개)
image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.lower().endswith(("jpg", "jpeg", "png"))][:1000]
total_images = len(image_files)
print(f"폴더 내 테스트할 이미지 갯수: {total_images}")

# 고정된 쿼리 텍스트 생성 함수
def generate_prompt(image_path):
    return "A description of the image."

# 평가 지표 계산 함수
def calculate_metrics(relevances, k):
    relevances = np.array(relevances)
    gains = 2**relevances - 1
    discounts = np.log2(np.arange(len(relevances)) + 2)
    dcg = np.sum(gains / discounts)
    idcg = np.sum(np.sort(gains)[::-1] / discounts)
    ndcg = dcg / idcg if idcg > 0 else 0.0
    mrr = np.sum(relevances / np.arange(1, len(relevances) + 1))
    map_k = np.mean([np.mean(relevances[:i+1]) for i in range(k)])
    return ndcg, mrr, map_k

# CLIP 모델 및 프로세서 로드 및 ChromaDB 생성
evaluation_results = []

for model_info in models_info:
    model_name = model_info["name"]
    db_path = model_info["db_path"]

    # 모델 및 프로세서 로드
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)

    # chroma 클라이언트 초기화
    client = chromadb.PersistentClient(path=db_path)
    collection_name = model_name.replace("/", "_")

    # 컬렉션 생성 (혹은 기존 컬렉션 가져오기)
    try:
        collection = client.get_collection(name=collection_name)
    except ValueError:
        collection = client.create_collection(name=collection_name)

    # 이미지 임베딩 생성 및 DB에 저장
    for filename in tqdm(image_files, desc=f"{model_name} 이미지 처리 중"):
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

    print(f"{model_name} 모델의 모든 이미지가 처리되어 DB에 저장되었습니다.")

    # 평가 지표 초기화
    mrr_scores = []
    map5_scores = []
    map10_scores = []
    ndcg5_scores = []
    ndcg10_scores = []

    for image_path in tqdm(image_files, desc=f"{model_name} 평가 중"):
        # 고정된 쿼리 텍스트 생성
        prompt = generate_prompt(image_path)

        # 텍스트 임베딩 생성
        inputs = processor(text=prompt, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            text_features = model.get_text_features(**inputs)
        query_vector = text_features.squeeze().cpu().tolist()

        # 유사한 이미지 검색
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=10  # 상위 10개 결과 반환
        )

        # 검색 결과에서 순위와 관련성을 평가
        if results['ids'][0]:
            retrieved_image_ids = results['ids'][0]
        else:
            continue

        relevant_image_ids = [os.path.basename(image_path)]

        # 정답 이미지가 반환된 순위를 기록
        relevances = [1 if img_id in relevant_image_ids else 0 for img_id in retrieved_image_ids]
        ndcg5, mrr, map5 = calculate_metrics(relevances, 5)
        ndcg10, _, map10 = calculate_metrics(relevances, 10)

        mrr_scores.append(mrr)
        map5_scores.append(map5)
        map10_scores.append(map10)
        ndcg5_scores.append(ndcg5)
        ndcg10_scores.append(ndcg10)

    # 평균 평가 지표 계산
    avg_mrr = np.mean(mrr_scores)
    avg_map5 = np.mean(map5_scores)
    avg_map10 = np.mean(map10_scores)
    avg_ndcg5 = np.mean(ndcg5_scores)
    avg_ndcg10 = np.mean(ndcg10_scores)

    evaluation_results.append({
        "model": model_name,
        "avg_mrr": avg_mrr,
        "avg_map5": avg_map5,
        "avg_map10": avg_map10,
        "avg_ndcg5": avg_ndcg5,
        "avg_ndcg10": avg_ndcg10
    })

# 결과 출력
for result in evaluation_results:
    print(f"모델: {result['model']}")
    print(f"  평균 MRR: {result['avg_mrr']}")
    print(f"  평균 MAP@5: {result['avg_map5']}")
    print(f"  평균 MAP@10: {result['avg_map10']}")
    print(f"  평균 NDCG@5: {result['avg_ndcg5']}")
    print(f"  평균 NDCG@10: {result['avg_ndcg10']}")
