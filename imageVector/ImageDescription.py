import csv
from transformers import CLIPProcessor, CLIPModel
import chromadb
import torch
import numpy as np
from tqdm import tqdm
import gc
import time

def calculate_metrics(ranks):
    mrr = np.mean([1 / r if r > 0 else 0 for r in ranks])
    map_5 = np.mean([1 / r if r > 0 and r <= 5 else 0 for r in ranks])
    map_10 = np.mean([1 / r if r > 0 and r <= 10 else 0 for r in ranks])
    
    def dcg_at_k(r, k):
        r = np.asfarray(r)[:k]
        return np.sum(r / np.log2(np.arange(2, r.size + 2)))

    def ndcg_at_k(r, k):
        dcg_max = dcg_at_k(sorted(r, reverse=True), k)
        if not dcg_max:
            return 0.
        return dcg_at_k(r, k) / dcg_max

    ndcg_5 = np.mean([ndcg_at_k([1 if i+1 == r else 0 for i in range(5)], 5) for r in ranks if r > 0])
    ndcg_10 = np.mean([ndcg_at_k([1 if i+1 == r else 0 for i in range(10)], 10) for r in ranks if r > 0])
    
    hit_ratio_3 = np.mean([1 if r > 0 and r <= 3 else 0 for r in ranks])

    hit_ratio_10 = np.mean([1 if r > 0 and r <= 10 else 0 for r in ranks])
    
    return mrr, map_5, map_10, ndcg_5, ndcg_10, hit_ratio_3, hit_ratio_10

# 모델 로드 시간 측정 시작
start_time = time.time()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K").to(device)
processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")

# chroma 클라이언트 초기화
client = chromadb.PersistentClient(path="chroma_db")

# 컬렉션 가져오기
collection = client.get_collection(name="test_laion_clip14_1024")

# 모델 로드 시간 측정 종료
load_time = time.time() - start_time
print(f"Model load time: {load_time:.4f} seconds")

image_descriptions = {}
errors = []  # 오류를 기록할 리스트 추가
with open('fb_sum.csv', 'r', encoding='utf-8') as f:
    csv_reader = csv.reader(f)
    next(csv_reader)  # Skip header
    for row in csv_reader:
        if len(row) != 3:
            errors.append((row, "Incorrect number of columns"))
            continue
        try:
            index, image_name, summary = row
            image_descriptions[image_name] = summary
        except ValueError as e:
            errors.append((row, str(e)))

# 이미지 처리 시간 측정 시작
start_processing_time = time.time()

results = {
    'mrr': [], 'map@5': [], 'map@10': [], 'ndcg@5': [], 'ndcg@10': [], 'hit_ratio@3': [], 'hit_ratio@10': []
}

for image_id, query_text in tqdm(list(image_descriptions.items()), desc="Processing images"):
    try:
        inputs = processor(text=query_text, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            text_features = model.get_text_features(**inputs)
        query_vector = text_features.squeeze().cpu().tolist()

        search_results = collection.query(
            query_embeddings=[query_vector],
            n_results=10
        )

        rank = 0
        if image_id in search_results["ids"][0]:
            rank = search_results["ids"][0].index(image_id) + 1

        mrr, map_5, map_10, ndcg_5, ndcg_10, hit_ratio_3, hit_ratio_10 = calculate_metrics([rank])

        results['mrr'].append(mrr)
        results['map@5'].append(map_5)
        results['map@10'].append(map_10)
        results['ndcg@5'].append(ndcg_5)
        results['ndcg@10'].append(ndcg_10)
        results['hit_ratio@3'].append(hit_ratio_3)
        results['hit_ratio@10'].append(hit_ratio_10)

    except Exception as e:
        errors.append((image_id, str(e)))  # 오류를 기록

    gc.collect()  # 메모리 관리

# 이미지 처리 시간 측정 종료
processing_time = time.time() - start_processing_time
print(f"Image processing time: {processing_time:.4f} seconds")

# 오류 로그 출력
if errors:
    print("\nErrors encountered during processing:")
    for row, error in errors:
        print(f"Row {row}: {error}")

# 최종 결과 출력
print("\nFinal Results:")
for metric, values in results.items():
    valid_values = [v for v in values if not np.isnan(v)]
    if valid_values:
        print(f"{metric.upper()}: {np.mean(valid_values):.4f}")
    else:
        print(f"{metric.upper()}: N/A (No valid values)")

print(f"\nTotal images processed: {len(results['mrr'])}")
print(f"Images with non-zero rank: {sum(1 for r in results['mrr'] if r > 0)}")

# 총 실행 시간 측정 종료
total_time = time.time() - start_time
print(f"Total execution time: {total_time:.4f} seconds")
