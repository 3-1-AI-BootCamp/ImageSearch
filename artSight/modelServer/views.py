from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponse
from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from django.views.decorators.csrf import csrf_exempt
from .models import Art2D
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch
from .utils import connectDB, load_imgFile
import chromadb
import open_clip



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
    
    
    
    
# # laion2B-s32B-b79K clip 모델 사용
# def sentenceToVec(input_text):
    
#     # 번역 모델과 토크나이저 로드 (Helsinki-NLP/opus-mt-ko-en)7
#     translation_model_name = 'Helsinki-NLP/opus-mt-ko-en'
#     translation_tokenizer = AutoTokenizer.from_pretrained(translation_model_name)
#     translation_model = AutoModelForSeq2SeqLM.from_pretrained(translation_model_name)
    
#     model_inputs = translation_tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
#     translated_tokens = translation_model.generate(**model_inputs)
#     translated_text = translation_tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    
#     print("input: " + input_text)
#     print('trans: ' + translated_text)

#     # clip patch-14 모델 로드
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model, _, preprocess = open_clip.create_model_and_transforms(
#         'ViT-H-14', 
#         pretrained='laion2B-s32B-b79K', 
#         device=device
#     )
#     tokenizer = open_clip.get_tokenizer('ViT-H-14')

#     # 크로마 db 초기화
#     client = chromadb.HttpClient(
#         host = "16.171.169.62",
#         port = 8001,
#         ssl=False,
#         headers={
#             "X-Api-Key": "NIwdUnNFOo73SKvW+P6OYQguw1mNJtNLO0+2pS07"
#         }
#     )

#     # 콜렉션 참조
#     collection = client.get_collection(name="test_openai_clip14")
    
    
#     # 텍스트 인코딩 후 이미지 검색 결과 리턴
#     def text_inference(text_input):
#         # 텍스트를 토큰화 후 모델에 입력하기 위해 텐서로 변환
#         text = tokenizer(text_input)
#         text = text.to(device)

#         # 모델을 통해 텍스트 특징 추출
#         with torch.no_grad():
#             text_features = model.encode_text(text)
#             text_features /= text_features.norm(dim=-1, keepdim=True)

#         # numpy 배열로 변환 후 리스트로 변환
#         query_vector = text_features.cpu().numpy()[0].tolist()

#         # 유사한 이미지 검색
#         results = collection.query(
#             query_embeddings=[query_vector],
#             n_results=3  # 상위 3개 결과 반환
#         )

#         return results
    

#     # 유사한 이미지 검색
#     img_idx = text_inference(translated_text)
    
#     # ID 리스트 준비
#     img_ids = [id.replace('.jpg', '') for id in img_idx['ids'][0]]
#     print(img_ids)
    
    
    
#     # mysql에서 이미지에 관련된 정보 읽어오기
#     placeholders = ', '.join(['%s'] * len(img_ids))
    
#     # 쿼리 준비
#     query = f"SELECT ImageFileName, ArtTitle_kor, ArtistName_kor FROM art_2d WHERE ImageFileName IN ({placeholders})"

#     # 쿼리 실행
#     infoData = connectDB.execute_query(query, img_ids)

    
#     return infoData
    
    
    
    
# clip-vit-large-patch14 모델이 요청 문장을 벡터화해서 비교하고 일치하는 이미지를 db에서 찾아서 리턴하는 로직
def sentenceToVec(input_text):
    
    # 번역 모델과 토크나이저 로드 (Helsinki-NLP/opus-mt-ko-en)7
    translation_model_name = 'Helsinki-NLP/opus-mt-ko-en'
    translation_tokenizer = AutoTokenizer.from_pretrained(translation_model_name)
    translation_model = AutoModelForSeq2SeqLM.from_pretrained(translation_model_name)
    
    model_inputs = translation_tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    translated_tokens = translation_model.generate(**model_inputs)
    translated_text = translation_tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    
    print("input: " + input_text)
    print('trans: ' + translated_text)

    # clip patch-14 모델 로드
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
    collection = client.get_collection(name="test_openai_clip14")
    # collection = client.get_collection(name="Conversation_images_clip14")

    inputs = processor(text=translated_text, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
    query_vector = text_features.squeeze().cpu().tolist()


    # 유사한 이미지 검색
    img_idx = collection.query(
        query_embeddings=[query_vector],
        n_results=3  # 상위 5개 결과 반환
    )
    
    # ID 리스트 준비
    img_ids = [id.replace('.jpg', '') for id in img_idx['ids'][0]]
    
    
    
    # mysql에서 이미지에 관련된 정보 읽어오기
    placeholders = ', '.join(['%s'] * len(img_ids))
    
    # 쿼리 준비
    query = f"SELECT ImageFileName, ArtTitle_kor, ArtistName_kor FROM art_2d WHERE ImageFileName IN ({placeholders})"

    # 쿼리 실행
    infoData = connectDB.execute_query(query, img_ids)

    
    return infoData
