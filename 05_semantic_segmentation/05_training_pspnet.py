import random
import math
import time
import pandas as pd
import numpy as np
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim

# 모델 학습
def train_model(net, dataloaders_dict, criterion, scheduler, optimizer, num_epochs):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("사용 장치: ", device)

    # 네트워크를 GPU로
    net.to(device)

    # 네트워크가 어느 정도 고정되면 고속화
    torch.backends.cudnn.benchmark = True

    # 화상의 매수
    num_train_imgs = len(dataloaders_dict["train"].dataset)
    num_val_imgs = len(dataloaders_dict["val"].dataset)
    batch_size = dataloaders_dict["train"].batch_size

    # 반복자의 카운터 설정
    iteration = 1
    logs = []

    # 미니배치가 작으므로 3번 모아서 업데이트
    batch_multiplier = 3

    # epoch 루프
    for epoch in range(num_epochs):
        # 시작 시간 저장
        t_epoch_start = time.time()
        t_iter_start = time.time()
        # 에폭당 트레이닝 loss
        epoch_train_loss = 0.0
        # 에폭당 검증 loss
        epoch_val_loss = 0.0

        print('-------------')
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-------------')

        # epoch별 훈련 및 검증
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()  
                # 최적화 scheduler 갱신
                # -> lr이 에폭마다 바뀜
                scheduler.step() 
                optimizer.zero_grad()
                print('(train)')

            else:
                if((epoch+1) % 5 == 0):
                    net.eval()
                    print('-------------')
                    print('(val)')
                else:
                    # 검증은 다섯 번 중에 한 번만 수행
                    continue

            # 데이터 로더에서 minibatch씩 꺼내 루프
            # multiple minibatch
            # -> 3 배치 모으기
            count = 0  
            for imges, anno_class_imges in dataloaders_dict[phase]:
                # 미니배치 크기가 1이면 배치 노멀라이제이션에서 오류가 발생하므로 회피
                if imges.size()[0] == 1:
                    continue

                # GPU가 사용가능하면 GPU에 데이터를 보낸다
                imges = imges.to(device)
                anno_class_imges = anno_class_imges.to(device)

                # multiple minibatch로 파라미터 갱신
                # 3 배치를 모았다면 갱신
                if (phase == 'train') and (count == 0):
                    optimizer.step()
                    optimizer.zero_grad()
                    count = batch_multiplier

                # 순전파(forward) 계산
                # 트레이닝 모드라면 그래디언트 계산
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(imges)
                    loss = criterion(
                        outputs, anno_class_imges.long()) / batch_multiplier

                    # 훈련시에는 역전파
                    if phase == 'train':
                        loss.backward()  # 경사 계산
                        count -= 1

                        if (iteration % 10 == 0):  # 10iter에 한 번, loss를 표시
                            t_iter_finish = time.time()
                            duration = t_iter_finish - t_iter_start
                            print('반복 {} || Loss: {:.4f} || 10iter: {:.4f} sec.'.format(
                                iteration, loss.item()/batch_size*batch_multiplier, duration))
                            t_iter_start = time.time()

                        epoch_train_loss += loss.item() * batch_multiplier
                        iteration += 1

                    # 검증 시
                    else:
                        epoch_val_loss += loss.item() * batch_multiplier

        # epoch의 phase별 loss와 정답률
        t_epoch_finish = time.time()
        print('-------------')
        print('epoch {} || Epoch_TRAIN_Loss:{:.4f} ||Epoch_VAL_Loss:{:.4f}'.format(
            epoch+1, epoch_train_loss/num_train_imgs, epoch_val_loss/num_val_imgs))
        print('timer:  {:.4f} sec.'.format(t_epoch_finish - t_epoch_start))
        t_epoch_start = time.time()

        # 로그 저장
        log_epoch = {'epoch': epoch+1, 'train_loss': epoch_train_loss /
                     num_train_imgs, 'val_loss': epoch_val_loss/num_val_imgs}
        logs.append(log_epoch)
        df = pd.DataFrame(logs)
        df.to_csv("log_output.csv")

    # 최후의 네트워크를 저장
    torch.save(net.state_dict(), 'weights/pspnet50_' +
               str(epoch+1) + '.pth')


if __name__ == "__main__":
    torch.manual_seed(1234)
    np.random.seed(1234)
    random.seed(1234)
    
    # 파일 경로 리스트 작성
    from utils.psp import make_datapath_list
    rootpath = "./data/VOCdevkit/VOC2012/"
    train_img_list, train_anno_list, val_img_list, val_anno_list = make_datapath_list(
        rootpath=rootpath)

    # Dataset 작성
    from utils.psp import DataTransform, VOCDataset
    # (RGB) 색의 평균값과 표준편차
    color_mean = (0.485, 0.456, 0.406)
    color_std = (0.229, 0.224, 0.225)
    train_dataset = VOCDataset(train_img_list, train_anno_list, phase="train", transform=DataTransform(
        input_size=475, color_mean=color_mean, color_std=color_std))
    val_dataset = VOCDataset(val_img_list, val_anno_list, phase="val", transform=DataTransform(
        input_size=475, color_mean=color_mean, color_std=color_std))

    # DataLoader 작성
    batch_size = 8
    train_dataloader = data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False)
    dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}
    
    from utils.psp import PSPNet
    # 파인 튜닝을 위한 PSPNet을 작성
    # ADE20K는 클래스 수가 150
    net = PSPNet(n_classes=150)

    # ADE20K 학습된 파라미터 읽기
    state_dict = torch.load("./weights/pspnet50_ADE20K.pth")
    net.load_state_dict(state_dict)

    # 분류용 합성곱층을 출력수 21로 바꿈
    n_classes = 21
    net.decode_feature.classification = nn.Conv2d(
        in_channels=512, out_channels=n_classes, kernel_size=1, stride=1, padding=0)
    net.aux.classification = nn.Conv2d(
        in_channels=256, out_channels=n_classes, kernel_size=1, stride=1, padding=0)

    # 교체한 합성곱층을 초기화
    # loss에서 시그모이드를 적용할 것이므로 Xavier를 사용.
    def weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight.data)
            # bias도 초기화
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    net.decode_feature.classification.apply(weights_init)
    net.aux.classification.apply(weights_init)

    print('네트워크 설정 완료: 학습된 가중치를 로드했습니다')
    
    from utils.psp import PSPLoss
    criterion = PSPLoss(aux_weight=0.4)
    
    # 파인 튜닝이므로, 학습률은 작게
    # classifier층은 크게 줌
    optimizer = optim.SGD([
        {'params': net.feature_conv.parameters(), 'lr': 1e-3},
        {'params': net.feature_res_1.parameters(), 'lr': 1e-3},
        {'params': net.feature_res_2.parameters(), 'lr': 1e-3},
        {'params': net.feature_dilated_res_1.parameters(), 'lr': 1e-3},
        {'params': net.feature_dilated_res_2.parameters(), 'lr': 1e-3},
        {'params': net.pyramid_pooling.parameters(), 'lr': 1e-3},
        {'params': net.decode_feature.parameters(), 'lr': 1e-2},
        {'params': net.aux.parameters(), 'lr': 1e-2},
    ], momentum=0.9, weight_decay=0.0001)

    # 스케쥴러 설정
    # 에폭마다 lr을 변경
    def lambda_epoch(epoch):
        max_epoch = 30
        return math.pow((1-epoch/max_epoch), 0.9)

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_epoch)

    # 학습 및 검증 실행
    num_epochs = 30
    train_model(net, dataloaders_dict, criterion, scheduler, optimizer, num_epochs=num_epochs)
