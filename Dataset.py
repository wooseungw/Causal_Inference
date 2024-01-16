import os
import json
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch
Image.LOAD_TRUNCATED_IMAGES = True

"""주소 불러오는 함수"""
def get_subdirectories(parent_directory):
    """주어진 디렉토리 내의 모든 하위 디렉토리의 경로를 리스트로 반환합니다."""
    subdirectories = [os.path.join(parent_directory, d) for d in os.listdir(parent_directory) 
                    if os.path.isdir(os.path.join(parent_directory, d))]
    return subdirectories

def get_all_subdirectories(parent_directory):
    """주어진 디렉토리 내의 첫 번째 및 두 번째 레벨 하위 디렉토리의 경로를 NumPy ndarray로 반환합니다."""
    first_level_subdirs = get_subdirectories(parent_directory)
    all_subdirs = []

    for subdir in first_level_subdirs:
        second_level_subdirs = get_subdirectories(subdir)
        all_subdirs.extend(second_level_subdirs)  # 두 번째 레벨의 디렉토리를 추가

    return np.array(all_subdirs)
"""예제"""
# category_list = get_subdirectories(os.path.join(os.getcwd(),"Dataset/Train/Image"))
# all_list = get_all_subdirectories(os.path.join(os.getcwd(),"Dataset/Train/Image"))
# print(all_list)


class BaseDataset(Dataset):
    def __init__(self, loc=os.path.join(os.getcwd(), "Dataset/Train/Image"), istrain=True):
        self.loc = loc if istrain else loc.replace("Train", "Valid")
        self.dir_names = get_all_subdirectories(self.loc)

    def __len__(self):
        return len(self.dir_names)

class QADataset(BaseDataset):
    def __init__(self, transform = None, loc=os.path.join(os.getcwd(), "Dataset/Train/Image"), istrain=True):
        super(QADataset, self).__init__(loc, istrain)
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ]) if transform == None else transform

    def __getitem__(self, index):
        dir_path = self.dir_names[index]
        # 마지막 부분(숫자) 추출
        dir_name = os.path.basename(dir_path)

        # 카테고리 이름 추출 (예: 'Cutting')
        category_name = os.path.basename(os.path.dirname(dir_path))

        # JSON 파일 경로 구성
        json_path = os.path.join(dir_path, dir_name + ".json")

        with open(json_path, 'r',encoding='utf-8') as file:
            data = json.load(file)

        question_image_path = os.path.join(dir_path, data['Questions'][0]['images'][0]['image_url'])
        question_image = Image.open(question_image_path).convert('RGB')
        question_image = self.transform(question_image)

        answer_images = []
        for answer in data['Answers']:
            answer_image_path = os.path.join(dir_path, answer['images'][0]['image_url'])
            answer_image = Image.open(answer_image_path).convert('RGB')
            answer_image = self.transform(answer_image)
            answer_images.append(answer_image)

        # 질문 이미지와 정답 이미지들을 하나의 텐서로 결합
        images_stack = torch.stack([question_image] + answer_images, dim=0)

        correct_answer_group_id = data['Answers'][0]['group_id']

        return images_stack, correct_answer_group_id, category_name

 
 