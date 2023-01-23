import glob
import os.path as osp
import random
import numpy as np
import json
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import models, transforms

class ImageTransform():
    def __init__(self, resize, mean, std):
        self.data_transform = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(
                    resize, scale=(0.5, 1.0)),  # 데이터 확장
                transforms.RandomHorizontalFlip(),  # 데이터 확장
                transforms.ToTensor(),  # 텐서로 변환
                transforms.Normalize(mean, std)  # 표준화
            ]),
            'val': transforms.Compose([
                transforms.Resize(resize),  # 리사이즈
                transforms.CenterCrop(resize),  # 이미지 중앙을 resize × resize로 자른다
                transforms.ToTensor(),  # 텐서로 변환
                transforms.Normalize(mean, std)  # 표준화
            ])
        }

    def __call__(self, img, phase='train'):
        return self.data_transform[phase](img)

# 트레이닝 / 검증에 사용할 데이터 파일 리스트를 불러옴
def make_datapath_list(phase="train"):
    rootpath = "./data/hymenoptera_data/"
    target_path = osp.join(rootpath+phase+'/**/*.jpg')

    # ret
    path_list = []

    # glob을 이용하여 하위 디렉토리의 파일 경로를 가져온다
    for path in glob.glob(target_path):
        path_list.append(path)

    return path_list

# 개미와 벌의 이미지에 대한 Dataset을 작성한다
class HymenopteraDataset(data.Dataset):
    def __init__(self, file_list, transform=None, phase='train'):
        self.file_list = file_list  # 파일 경로 리스트
        self.transform = transform  # 전처리 클래스의 인스턴스
        self.phase = phase  # train or val 지정

    # 이미지 수 반환
    def __len__(self):
        return len(self.file_list)

    # tensor 형식의 이미지와 라벨링 취득
    def __getitem__(self, index):
        # index번째의 화상을 로드
        img_path = self.file_list[index]
        img = Image.open(img_path)  # [높이][너비][색RGB]

        # 화상의 전처리를 실시
        img_transformed = self.transform(
            img, self.phase)  # torch.Size([3, 224, 224])

        # 라벨을 파일 이름에서 추출
        if self.phase == "train":
            label = img_path[30:34]
        elif self.phase == "val":
            label = img_path[28:32]

        # 라벨을 숫자로 변경
        if label == "ants":
            label = 0
        elif label == "bees":
            label = 1

        return img_transformed, label

# 모델을 학습
def train_model(net, dataloaders_dict, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-------------')

        # epoch별 학습 및 검증 루프
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()  # 모델을 훈련 모드로
            else:
                net.eval()   # 모델을 검증 모드로

            epoch_loss = 0.0  # 이 epoch에서 손실의 합
            epoch_corrects = 0  # 이 epoch에서 정답률

            # 미 학습시 검증 성능을 확인하기 위해 epoch=0의 훈련 생략
            if (epoch == 0) and (phase == 'train'):
                continue

            # 데이터 로더로 미니 배치를 꺼내는 루프
            for inputs, labels in tqdm(dataloaders_dict[phase]):
                # optimizer를 초기화
                optimizer.zero_grad()

                # 순전파(forward) 계산
                # train일 경우에만 그래디언트 활성화
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)  # 손실을 계산
                    _, preds = torch.max(outputs, 1)  # 라벨을 예측
  
                    # 훈련시에는 오차 역전파
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    # loss 합계 갱신
                    epoch_loss += loss.item() * inputs.size(0)  
                    # 정답 수의 합계 갱신
                    epoch_corrects += torch.sum(preds == labels.data)

            # epoch 당 loss와 정답률 표시
            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = epoch_corrects.double() / len(dataloaders_dict[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
                
if __name__ == "__main__":
    # 난수 시드 설정
    torch.manual_seed(1234)
    np.random.seed(1234)
    random.seed(1234)
    
    # 전처리 정보
    size = 224
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    image_transform = ImageTransform(size, mean, std)

    # 트레이닝 / 검증에 사용할 데이터 파일 리스트를 불러옴
    train_list = make_datapath_list(phase="train")
    val_list = make_datapath_list(phase="val")

    # dataset 작성
    train_dataset = HymenopteraDataset(
        file_list=train_list, transform=image_transform, phase='train')
    val_dataset = HymenopteraDataset(
        file_list=val_list, transform=image_transform, phase='val')
    
    # 미니 배치 크기 지정
    batch_size = 32

    # DataLoader 작성
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False)

    dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}

    # 학습된 VGG-16 모델을 수정
    use_pretrained = True  # 학습된 파라미터를 사용
    net = models.vgg16(pretrained=use_pretrained)
    net.classifier[6] = nn.Linear(in_features=4096, out_features=2) # 마지막을 개미와 벌 2개
    net.train()   # train 모드

    # 학습시킬 파라미터 이외에는 그래디언트를 계산하지 않고, 변하지 않게 설정
    params_to_update = []
    update_param_names = ["classifier.6.weight", "classifier.6.bias"]
    for name, param in net.named_parameters():
        if name in update_param_names:
            param.requires_grad = True
            params_to_update.append(param)
        else:
            param.requires_grad = False
    
    # 학습 및 검증 실시
    optimizer = optim.SGD(params=params_to_update, lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    num_epochs=2
    train_model(net, dataloaders_dict, criterion, optimizer, num_epochs=num_epochs)
