import numpy as np
import random
import torch, gc
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm
from utils.dataloader_image_classification import ImageTransform, make_datapath_list, HymenopteraDataset

# 모델 학습
def train_model(net, dataloaders_dict, criterion, optimizer, num_epochs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 네트워크를 GPU로 전송
    net.to(device)

    # 네트워크가 어느 정도 고정되면, 고속 연산 수행
    torch.backends.cudnn.benchmark = True

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-------------')

        # epoch별 훈련 및 검증 루프
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()
            else:
                net.eval()

            epoch_loss = 0.0  # epoch당 손실의 합
            epoch_corrects = 0  # epoch 정답률

            # 미학습시의 검증 성능을 확인하기 위해 epoch=0의 훈련은 생략
            if (epoch == 0) and (phase == 'train'):
                continue

            # 데이터 로더에서 미니 배치를 꺼내 루프
            for inputs, labels in tqdm(dataloaders_dict[phase]):
                # 네크워크 뿐 아니라 라벨과 인풋을 GPU로 전송
                inputs = inputs.to(device)
                labels = labels.to(device)

                # optimizer를 초기화
                optimizer.zero_grad()

                # 순전파(forward) 계산
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)  # 손실 계산
                    _, preds = torch.max(outputs, 1)  # 라벨 예측

                    # 훈련시에는 오차 역전파 계산
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    # 결과 계산
                    epoch_loss += loss.item() * inputs.size(0)  # loss의 합계를 갱신
                    epoch_corrects += torch.sum(preds == labels.data)

            # epoch별 결과
            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = epoch_corrects.double(
            ) / len(dataloaders_dict[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

if __name__ == "__main__":
    # 난수 시드 설정
    torch.manual_seed(1234)
    np.random.seed(1234)
    random.seed(1234)

    # 메모리 정리
    gc.collect()
    torch.cuda.empty_cache()

    # 이미지 파일 리스트 생성
    train_list = make_datapath_list(phase="train")
    val_list = make_datapath_list(phase="val")

    # Dataset 생성
    size = 224
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    image_transform = ImageTransform(size, mean, std)
    train_dataset = HymenopteraDataset(
        file_list=train_list, transform=image_transform, phase='train')
    val_dataset = HymenopteraDataset(
        file_list=val_list, transform=image_transform, phase='val')

    # DataLoader 생성
    batch_size = 16
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False)
    dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}

    # 학습된 VGG-16 모델 구조 변경
    use_pretrained = True  # 학습된 파라미터를 사용
    net = models.vgg16(pretrained=use_pretrained)
    net.classifier[6] = nn.Linear(in_features=4096, out_features=2)
    net.train()

    # 파인 튜닝으로 학습할 파라미터를 params_to_update 변수의 1~3에 저장한다
    params_to_update_1 = []
    params_to_update_2 = []
    params_to_update_3 = []
    update_param_names_1 = ["features"]
    update_param_names_2 = ["classifier.0.weight",
                            "classifier.0.bias", "classifier.3.weight", "classifier.3.bias"]
    update_param_names_3 = ["classifier.6.weight", "classifier.6.bias"]
    for name, param in net.named_parameters():
        if update_param_names_1[0] in name:
            param.requires_grad = True
            params_to_update_1.append(param)
        elif name in update_param_names_2:
            param.requires_grad = True
            params_to_update_2.append(param)
        elif name in update_param_names_3:
            param.requires_grad = True
            params_to_update_3.append(param)
        else:
            param.requires_grad = False

    # 최적화 기법 설정
    optimizer = optim.SGD([
        {'params': params_to_update_1, 'lr': 1e-4},
        {'params': params_to_update_2, 'lr': 5e-4},
        {'params': params_to_update_3, 'lr': 1e-3}
    ], momentum=0.9)

    # 학습 및 검증
    criterion = nn.CrossEntropyLoss()
    num_epochs=2
    train_model(net, dataloaders_dict, criterion, optimizer, num_epochs=num_epochs)

    # PyTorch 네트워크 파라미터 저장
    save_path = './weights_fine_tuning.pth'
    torch.save(net.state_dict(), save_path)

    # PyTorch 네트워크 파라미터 로드
    load_path = './weights_fine_tuning.pth'
    load_weights = torch.load(load_path)
    net.load_state_dict(load_weights)

    # GPU 상에 저장된 가중치를 CPU에 로드할 경우
    load_weights = torch.load(load_path, map_location={'cuda:0': 'cpu'})
    net.load_state_dict(load_weights)
