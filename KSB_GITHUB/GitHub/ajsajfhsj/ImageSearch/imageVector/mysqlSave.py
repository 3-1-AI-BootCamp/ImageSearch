import os
import json
import mysql.connector
from PIL import Image
import io
from tqdm import tqdm

# MySQL 연결 설정
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="1234",
    database="art_data"
)
cursor = db.cursor()

# 테이블 생성 쿼리
create_table_query = """
CREATE TABLE IF NOT EXISTS art_2d (
    ID INT AUTO_INCREMENT PRIMARY KEY,
    ImageFileName VARCHAR(255) NOT NULL,
    MainCategory VARCHAR(255) NOT NULL,
    SubCategory VARCHAR(255) NOT NULL,
    MiddleCategory VARCHAR(255) NOT NULL,
    Class_kor VARCHAR(255) NOT NULL,
    ArtTitle_kor VARCHAR(255) NOT NULL,
    ArtistName_kor VARCHAR(255) NOT NULL,
    Image LONGBLOB NOT NULL
)
"""
cursor.execute(create_table_query)

# JSON 파일이 있는 디렉토리 경로
json_dir = "C:\\Users\\admin\\Desktop\\miniProject\\image\\json"
# 이미지 파일이 있는 디렉토리 경로
image_dir = "C:\\Users\\admin\\Desktop\\miniProject\\image\\Conversation"

# JSON 파일 목록 가져오기
json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
total_files = len(json_files)

print(f"총 {total_files}개의 파일을 처리합니다.")

# JSON 파일 순회
for i, filename in enumerate(tqdm(json_files, desc="처리 중", unit="파일")):
    with open(os.path.join(json_dir, filename), 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    # 필요한 데이터 추출
    image_file_name = data['Data_Info']['ImageFileName']
    main_category = data['Object_Info']['MainCategory']
    sub_category = data['Object_Info']['SubCategory']
    middle_category = data['Object_Info']['MiddleCategory']
    class_kor = data['Description']['Class_kor']
    art_title_kor = data['Description']['ArtTitle_kor']
    artist_name_kor = data['Description']['ArtistName_kor']
    
    # 이미지 파일 찾기 및 읽기
    image_path = os.path.join(image_dir, f"{image_file_name}.jpg")
    if os.path.exists(image_path):
        with Image.open(image_path) as img:
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG')
            image_data = img_byte_arr.getvalue()
        
        # MySQL에 데이터 삽입
        insert_query = """
        INSERT INTO art_2d (ImageFileName, MainCategory, SubCategory, MiddleCategory, 
                              Class_kor, ArtTitle_kor, ArtistName_kor, Image)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        values = (image_file_name, main_category, sub_category, middle_category,
                  class_kor, art_title_kor, artist_name_kor, image_data)
        
        cursor.execute(insert_query, values)
        db.commit()
    else:
        print(f"Image file not found: {image_file_name}.jpg")

# 연결 종료
cursor.close()
db.close()

print("모든 파일 처리가 완료되었습니다.")