import pandas as pd
import os

# summary.csv 파일을 읽어들임
csv_path = 'summary.csv'
df = pd.read_csv(csv_path)

# "Warning"이 포함된 행 필터링
warnings = df[df['summary'].str.contains("Warning")]

# 이미지 폴더 경로 설정
image_folder = r'C:\Users\admin\Desktop\ascs'

# 이미지 이름 추출 및 삭제
for image_name in warnings['image_name']:
    image_path = os.path.join(image_folder, image_name)
    if os.path.exists(image_path):
        try:
            os.remove(image_path)
            print(f"{image_name} 삭제됨")
        except Exception as e:
            print(f"Error deleting {image_name}: {e}")
    else:
        print(f"File not found: {image_name}")

# "Warning"이 포함된 행 삭제
df = df[~df['summary'].str.contains("Warning")]

# 수정된 데이터를 summary.csv에 다시 저장
df.to_csv(csv_path, index=False)
print("summary.csv 업데이트 완료")
