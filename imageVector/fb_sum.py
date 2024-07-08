import csv
from transformers import BartForConditionalGeneration, BartTokenizer
import torch
from tqdm import tqdm

# BART 모델 및 토크나이저 로드
model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def summarize_text(text, max_length=76):
    inputs = tokenizer([text], max_length=1024, return_tensors="pt", truncation=True).to(device)
    summary_ids = model.generate(inputs["input_ids"], num_beams=4, max_length=max_length, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# CSV 파일 읽기 및 요약하기
input_file = 'test.csv'  # 입력 CSV 파일명
output_file = 'fb_sum.csv'  # 출력 CSV 파일명

# 전체 행 수 계산
with open(input_file, 'r', encoding='utf-8') as infile:
    row_count = sum(1 for row in csv.reader(infile)) - 1  # 헤더 제외

with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', newline='', encoding='utf-8') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    
    # 헤더 행 읽기 및 쓰기
    header = next(reader)
    writer.writerow(header + ['Summarized Text'])  # 새로운 열 추가
    
    # tqdm으로 진행률 표시
    for row in tqdm(reader, total=row_count, desc="요약 진행 중"):
        # 요약할 텍스트가 있는 열의 인덱스를 지정 (2번째 열, 인덱스 2)
        text_to_summarize = row[2]
        summarized_text = summarize_text(text_to_summarize)
        
        # 원래 행에 요약된 텍스트 추가
        writer.writerow(row + [summarized_text])

print(f"요약이 완료되었습니다. 결과가 {output_file}에 저장되었습니다.")