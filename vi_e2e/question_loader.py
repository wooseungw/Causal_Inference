import os
import shutil

import torch
import json
from torch.utils.data import Dataset
from PIL import Image
import numpy as np 
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class BaseQuestionLoader(Dataset):
    """Question loader base model."""

    def __init__(self, root, index_file_path='/workspace/vi_e2e/cause_effect_data_split_50%_v1.json'
                 , mode='train', transform=None):

        self.root = root
        self.transform = transform
        self.index_file_path = index_file_path
        self.mode = mode
        with open(index_file_path, 'r') as f:
            self.dir_names_dict = json.load(f)
        self.dir_names = self.dir_names_dict[mode]
        # self.dir_names = [dirname for dirname in self.dir_names if len(os.listdir(os.path.join(self.root, dirname))) >=5]
        self.dir_names = [dirname for dirname in self.dir_names if len(os.listdir(dirname) )>=5]




    def __len__(self):
        return len(self.dir_names)

    def get_dirname_and_info(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        dir_name = sorted(self.dir_names)[idx]
        # json_file_path = os.path.join(self.root, dir_name, f'{dir_name}.json')
        dir_num = dir_name.split("/")[-1]
        json_file_path = os.path.join(dir_name, f'{dir_num}.json') 
        with open(json_file_path) as fh:
            info = json.load(fh)

        return dir_name, info


class Question1Dataset(BaseQuestionLoader):
    """Code for extracting all positive samples from question1s."""

    def __getitem__(self, idx):

        dir_name, info = self.get_dirname_and_info(idx)

        question_image = info['Questions'][0]['images']
        positive_examples, = [answer['images'] for answer in info['Answers']
                              if answer['group_id'] == info["correct_answer_group_ID"][0]]
        samples1, samples2 = [], []
        for im in question_image + positive_examples:
            image = Image.open(os.path.join(
                self.root, dir_name, im['image_url']
            )).convert('RGB')

            # Augment the images twice.
            sample1, sample2 = self.transform(image)
            samples1.append(sample1)
            samples2.append(sample2)
        samples = [torch.squeeze(torch.stack(samples1), dim=0),
                   torch.squeeze(torch.stack(samples2), dim=0)]

        if samples[0].shape[0] != 4:
            # Delete question if it is invalid
            shutil.rmtree(os.path.join(self.root, dir_name))
            print(f"removed {os.path.join(self.root, dir_name)}")

        return samples


class Question2Dataset(BaseQuestionLoader):
    """Code for extracting all positive samples from question2s."""

    def __getitem__(self, idx):

        dir_name, info = self.get_dirname_and_info(idx)

        question_image = info['Questions'][0]['images']
        samples1, samples2 = [], []
        for im in question_image:
            image = Image.open(os.path.join(
                self.root, dir_name, im['image_url']
            )).convert('RGB')
            # Augment the images twice.
            sample1, sample2 = self.transform(image)
            samples1.append(sample1)
            samples2.append(sample2)
        samples = [torch.squeeze(torch.stack(samples1), dim=0),
                   torch.squeeze(torch.stack(samples2), dim=0)]
        if samples[0].shape[0] != 3:
            # Delete question if it is invalid
            shutil.rmtree(os.path.join(self.root, dir_name))
            print(f"removed {os.path.join(self.root, dir_name)}")

        return samples


class Question3Dataset(BaseQuestionLoader):
    """Code for extracting all positive samples from question3s."""

    def __getitem__(self, idx):
        dir_name, info = self.get_dirname_and_info(idx)
        positive_samples = []

        for question in info['Questions']:
            if question['group_id'] == info["correct_question_group_ID"][0]:
                positive_samples = positive_samples + question['images']

        for answer in info['Answers']:
            if answer['group_id'] == info["correct_answer_group_ID"][0]:
                positive_samples = positive_samples + answer['images']

        samples1, samples2 = [], []
        for im in positive_samples:
            image = Image.open(os.path.join(
                self.root, dir_name, im['image_url']
            )).convert('RGB')
            # Augment the images twice.
            sample1, sample2 = self.transform(image)
            samples1.append(sample1)
            samples2.append(sample2)
        samples = [torch.squeeze(torch.stack(samples1), dim=0),
                   torch.squeeze(torch.stack(samples2), dim=0)]
        if samples[0].shape[0] != 6:
            # Delete question if it is invalid
            shutil.rmtree(os.path.join(self.root, dir_name))
            print(f"removed {os.path.join(self.root, dir_name)}")

        return samples


class Question4Dataset(BaseQuestionLoader):
    """Code for extracting all positive samples from question4s."""

    def __getitem__(self, idx):
        dir_name, info = self.get_dirname_and_info(idx)
        positive_samples = []

        for question in info['Questions']:
            positive_samples = positive_samples + question['images']

        for answer in info['Answers']:
            if answer['group_id'] in info["correct_answer_group_ID"]:
                positive_samples = positive_samples + answer['images']

        samples1, samples2 = [], []
        for im in positive_samples:
            image = Image.open(os.path.join(
                self.root, dir_name, im['image_url']
            )).convert('RGB')
            # Augment the images twice.
            sample1, sample2 = self.transform(image)
            samples1.append(sample1)
            samples2.append(sample2)
        samples = [torch.squeeze(torch.stack(samples1), dim=0),
                   torch.squeeze(torch.stack(samples2), dim=0)]
        if samples[0].shape[0] != 5:
            # Delete question if it is invalid
            shutil.rmtree(os.path.join(self.root, dir_name))
            print(f"removed {os.path.join(self.root, dir_name)}")

        return samples


class Group2Dataset(BaseQuestionLoader):

    def __getitem__(self, idx):
        dir_name, info = self.get_dirname_and_info(idx)
        positive_samples, negative_examples = [], []

        for question in info['Questions']:
            positive_samples = positive_samples + question['images']

        for answer in info['Answers']:
            if answer['group_id'] in info["correct_answer_group_ID"]:
                positive_samples = positive_samples + answer['images']
            else:
                negative_examples = negative_examples + answer['images']

        samples1, samples2 = [], []
        for im in positive_samples:
            path = os.path.join(self.root, dir_name, im['image_url'])
            try:
                image = Image.open(os.path.join(
                    self.root, dir_name, im['image_url']
                )).convert('RGB')
            except:
                os.system(f'rm -rf {path}')
            sample1, sample2 = self.transform(image)
            samples1.append(sample1)
            samples2.append(sample2)
        samples = [torch.squeeze(torch.stack(samples1), dim=0),
                   torch.squeeze(torch.stack(samples2), dim=0)]
        print(samples[0].shape[0])
        
        if samples[0].shape[0] != 2:
            print(samples[0].shape[0])

            shutil.rmtree(os.path.join(self.root, dir_name))
            print(f"removed {os.path.join(self.root, dir_name)}")

        return samples

class Group2Dataset_E2E(BaseQuestionLoader):

    def __getitem__(self, idx):
        dir_name, info = self.get_dirname_and_info(idx)
        # positive_samples, negative_examples = [], []
        imgs = []
        for question in info['Questions']:
            imgs.extend(question['images'])

        for answer in info['Answers']:
            imgs.extend(answer['images'])


        transfomed_imgs = []
        for im in imgs:
            # print(im)
            path = os.path.join(dir_name, im['image_url'])
            try:
                image = Image.open(os.path.join(
                    dir_name, im['image_url']
                )).convert('RGB')
                transfomed_imgs.append(self.transform(image))

            except OSError:
                print(path)
                # os.system(f'rm -rf {path}')


        # print(len(transfomed_imgs))

        if len(transfomed_imgs) < 4:
            print(os.path.join(dir_name))
            raise f"removed {os.path.join(dir_name)}"
            # shutil.rmtree(os.path.join(dir_name))
            # print()
        answer = info['correct_answer_group_ID'][0]
        return transfomed_imgs, answer, info["category"]
