from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponse
from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel
from django.views.decorators.csrf import csrf_exempt
from .models import Art2D
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch
from .utils import connectDB
import chromadb
import base64



# /modelServer/art_2d/ 로 접속
def get_art_2d(request):
    art_2d_list = Art2D.objects.all()[:2]
    return render(request, 'art_2d_list.html', {'art_2d_list': art_2d_list})


# 요청 받아서 모델이 예측한 결과를 리턴하는 함수
@csrf_exempt
def process_data(request):
    if request.method == 'POST':
        input_text = request.POST.get('input_text', '')
        
        
        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)

        # inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)
        inputs = processor(text = [input_text], images=image, return_tensors="pt", padding=True)

        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image # this is the image-text similarity score
        probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities
        probs_list = probs.detach().cpu().numpy().tolist()
        
        # JSON 응답으로 데이터 반환
        return JsonResponse({'data': probs_list})
    else:
        return JsonResponse({'error': 'Invalid request method.'}, status=405)
    
    
    
# clip 모델이 요청 문장을 벡터화해서 비교하고 일치하는 이미지를 db에서 찾아서 리턴하는 로직
def sentenceToVec(input_text):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    # chroma 클라이언트 초기화
    client = chromadb.HttpClient(
        host = "16.171.169.62",
        port = 8001,  # ChromaDB 서버의 포트 (기본값은 8000)
        ssl=False,
        headers={
            "X-Api-Key": "NIwdUnNFOo73SKvW+P6OYQguw1mNJtNLO0+2pS07"
        }
    )

    # 컬렉션 생성 (혹은 기존 컬렉션 가져오기)
    #collection = client.get_collection(name="Conversation_images")
    collection = client.get_collection(name="Conversation_images_clip14")

    inputs = processor(text=input_text, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
    query_vector = text_features.squeeze().cpu().tolist()


    # 유사한 이미지 검색
    img_idx = collection.query(
        query_embeddings=[query_vector],
        n_results=5  # 상위 5개 결과 반환
    )
    
    # ID 리스트 준비
    img_ids = [id.replace('.jpg', '') for id in img_idx['ids'][0]]
    placeholders = ', '.join(['%s'] * len(img_ids))

    # 쿼리 준비
    query = f"SELECT * FROM art_2d WHERE ImageFileName IN ({placeholders})"

    # 쿼리 실행
    results = connectDB.execute_query(query, img_ids)
    
    # 이미지를 base64로 인코딩
    for row in results:
        if row['Image']:
            row['Image'] = base64.b64encode(row['Image']).decode('utf-8')

    
    return JsonResponse({'data': results})
