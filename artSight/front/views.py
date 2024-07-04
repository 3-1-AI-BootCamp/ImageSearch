import requests
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
import json
from modelServer.views import sentenceToVec


# 초기 메인 페이지 렌더링
def mainPage(request):
    return render(request, 'search.html')


@csrf_exempt
def display_data(request):
    if request.method == 'POST':
        input_text = request.POST.get('input_text', '')
        # modelServer로 POST 요청 보내기
        response = requests.post('http://localhost:8000/modelServer/process/', data={'input_text': input_text})
        
        if response.status_code == 200:
            processed_data = response.json().get('data', 'No data received')
            return render(request, 'display_data.html', {'data': processed_data})
        else:
            return render(request, 'display_data.html', {'data': 'Error processing data'})
    else:
        return render(request, 'display_data.html')



# 사용자가 문장을 입력했을 때 이 함수로 요청이 오고 여기서 modelServer앱의 sentenceToVec함수를 호출
@csrf_exempt
def requestSearch(request):
    if request.method == 'POST':
        input_text = request.POST.get('input_text', '')
        
        # modelServer로 POST 요청 보내기
        response = sentenceToVec(input_text)
        print(response)
        
        # 리턴 받은 이미지 데이터를 템플릿으로 전달
        return render(request, 'search.html', {'response': response})
        
        
    return render(request, 'search.html')