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
from .utils import connectDB, load_imgFile
import chromadb
import os



# /modelServer/art_2d/ 로 접속 (이미지 출력하는 테스트 로직)
def get_art_2d(request):
    art_2d_list = Art2D.objects.all()[:2]
    return render(request, 'art_2d_list.html', {'art_2d_list': art_2d_list})


# 요청 받아서 모델이 예측한 결과를 리턴하는 함수(모델 처리 결과 테스트 로직)
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

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    # processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    # # chroma 클라이언트 초기화
    # client = chromadb.HttpClient(
    #     host = "16.171.169.62",
    #     port = 8001,  # ChromaDB 서버의 포트 (기본값은 8000)
    #     ssl=False,
    #     headers={
    #         "X-Api-Key": "NIwdUnNFOo73SKvW+P6OYQguw1mNJtNLO0+2pS07"
    #     }
    # )

    # # 컬렉션 생성 (혹은 기존 컬렉션 가져오기)
    # #collection = client.get_collection(name="Conversation_images")
    # collection = client.get_collection(name="Conversation_images_clip14")

    # inputs = processor(text=input_text, return_tensors="pt", padding=True).to(device)
    # with torch.no_grad():
    #     text_features = model.get_text_features(**inputs)
    # query_vector = text_features.squeeze().cpu().tolist()


    # # 유사한 이미지 검색
    # img_idx = collection.query(
    #     query_embeddings=[query_vector],
    #     n_results=5  # 상위 5개 결과 반환
    # )
    
    # # ID 리스트 준비
    # img_ids = [id.replace('.jpg', '') for id in img_idx['ids'][0]]
    
    
    img_ids = ['kart_2d001630-C-8-81-1', 'kart_2d017783-C-8-81-1', 'kart_2d017714-C-8-81-1']
    
    # 이미지 읽어오기
    # imgFiles = load_imgFile.load_image_paths_from_ids(img_ids)
    
    # mysql에서 이미지에 관련된 정보 읽어오기
    placeholders = ', '.join(['%s'] * len(img_ids))
    
    # 쿼리 준비
    query = f"SELECT ImageFileName, ArtTitle_kor, ArtistName_kor FROM art_2d WHERE ImageFileName IN ({placeholders})"

    # 쿼리 실행
    infoData = connectDB.execute_query(query, img_ids)
    
    
    # imgFiles와 infoData의 데이터를 합치는 과정
    # combined_data = []

    # # infoData를 dictionary 형태로 변환하여 ImageFileName을 key로 사용
    # infoData_dict = {row['ImageFileName']: row for row in infoData}

    # # imgFiles의 각 파일명과 infoData를 매칭
    # for img_file in imgFiles:
    #     img_id = os.path.splitext(os.path.basename(img_file))[0]  # 파일명에서 ID 추출 (확장자 제외)
    #     img_info = infoData_dict.get(img_id)  # infoData에서 해당 이미지 파일명에 맞는 정보를 가져옴
        
    #     if img_info:
    #         combined_data.append({
    #             'image_path': img_file,  # 이미지 파일 경로
    #             'info': img_info
    #         })
    
    return infoData
