import os
from django.conf import settings

def load_image_paths_from_ids(img_ids, image_folder=None):
    if image_folder is None:
        image_folder = os.path.join(settings.MEDIA_ROOT, 'images')
    
    image_paths = []
    
    for img_id in img_ids:
        possible_extensions = ['.jpg', '.jpeg', '.png', '.gif']
        
        for ext in possible_extensions:
            img_path = os.path.join(image_folder, f"{img_id}{ext}")
            if os.path.exists(img_path):
                # 상대 경로로 변환하고 '/'로 경로 구분자 변경
                relative_path = os.path.relpath(img_path, settings.MEDIA_ROOT).replace('\\', '/')
                image_paths.append(relative_path)
                break
        else:
            print(f"Image not found for ID: {img_id}")
    
    return image_paths