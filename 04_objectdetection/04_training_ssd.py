import os.path as osp
import random
import time
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data as data

# 네트워크를 He초기화
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight.data)
        # 바이어스 항이 있는 경우의 초기화
        if m.bias is not None:  
            nn.init.constant_(m.bias, 0.0)

# 모델 학습
def train_model(net, dataloaders_dict, criterion, optimizer, num_epochs):
    # 네트워크 설정
    net.to(device)
    # 네트워크가 어느 정도 고정되면 가속
    torch.backends.cudnn.benchmark = True

    # 반복자의 카운터 설정
    iteration = 1
    # 에폭 트레이닝 loss
    epoch_train_loss = 0.0
    # 에폭 검증 loss
    epoch_val_loss = 0.0
    logs = []

    # epoch 루프
    for epoch in range(num_epochs+1):
        # 시작 시간을 저장
        t_epoch_start = time.time()
        t_iter_start = time.time()

        print('-------------')
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-------------')

        # epoch별 훈련 및 검증을 루프
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()  # 모델을 훈련모드로
                print('(train)')
            else:
                if((epoch+1) % 10 == 0):
                    net.eval()   # 모델을 검증모드로
                    print('-------------')
                    print('(val)')
                else:
                    continue

            # 데이터 로더에서 minibatch씩 꺼내 루프
            for images, targets in dataloaders_dict[phase]:
                # GPU에 데이터를 보낸다
                images = images.to(device)
                targets = [ann.to(device)
                           for ann in targets]

                # optimizer를 초기화
                optimizer.zero_grad()

                # 순전파(forward) 계산
                # train일 경우에만 그래디언트 계산 활성화
                with torch.set_grad_enabled(phase == 'train'):
                    # 순전파(forward) 계산
                    outputs = net(images)

                    # 손실 계산
                    loss_l, loss_c = criterion(outputs, targets)
                    loss = loss_l + loss_c

                    # 훈련시에는 역전파(Backpropagation)
                    if phase == 'train':
                        loss.backward()  # 경사 계산

                        # 경사가 너무 커지면 계산이 불안정해지므로 상한을 둠
                        nn.utils.clip_grad_value_(
                            net.parameters(), clip_value=2.0)

                        # 파라미터 갱신
                        optimizer.step()

                        if (iteration % 10 == 0):  # 10iter에 한 번, loss를 표시
                            t_iter_finish = time.time()
                            duration = t_iter_finish - t_iter_start
                            print('반복 {} || Loss: {:.4f} || 10iter: {:.4f} sec.'.format(
                                iteration, loss.item(), duration))
                            t_iter_start = time.time()

                        epoch_train_loss += loss.item()
                        iteration += 1

                    # 검증시
                    else:
                        epoch_val_loss += loss.item()

        # epoch의 phase 당 loss와 정답률
        t_epoch_finish = time.time()
        print('-------------')
        print('epoch {} || Epoch_TRAIN_Loss:{:.4f} ||Epoch_VAL_Loss:{:.4f}'.format(
            epoch+1, epoch_train_loss, epoch_val_loss))
        print('timer:  {:.4f} sec.'.format(t_epoch_finish - t_epoch_start))
        t_epoch_start = time.time()

        # 로그를 저장
        log_epoch = {'epoch': epoch+1,
                     'train_loss': epoch_train_loss, 'val_loss': epoch_val_loss}
        logs.append(log_epoch)
        df = pd.DataFrame(logs)
        df.to_csv("log_output.csv")

        epoch_train_loss = 0.0  # epoch의 손실합
        epoch_val_loss = 0.0  # epoch의 손실합

        # 네트워크를 저장한다
        if ((epoch+1) % 10 == 0):
            torch.save(net.state_dict(), 'weights/ssd300_' +
                       str(epoch+1) + '.pth')

if __name__ == "__main__":
    # 난수 시드 설정
    torch.manual_seed(1234)
    np.random.seed(1234)
    random.seed(1234)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("사용 중인 장치:", device)
    
    # 파일 경로 리스트를 취득
    rootpath = "./data/VOCdevkit/VOC2012/"
    from utils.ssd import make_datapath_list
    train_img_list, train_anno_list, val_img_list, val_anno_list = make_datapath_list(
        rootpath)
        
    # Dataset 작성
    voc_classes = ['aeroplane', 'bicycle', 'bird', 'boat',
                   'bottle', 'bus', 'car', 'cat', 'chair',
                   'cow', 'diningtable', 'dog', 'horse',
                   'motorbike', 'person', 'pottedplant',
                   'sheep', 'sofa', 'train', 'tvmonitor']
    color_mean = (104, 117, 123)  # (BGR) 색의 평균값
    input_size = 300  # input 크기를 300×300으로 설정

    from utils.ssd import VOCDataset, DataTransform, Anno_xml2list
    train_dataset = VOCDataset(train_img_list, train_anno_list, phase="train", transform=DataTransform(
        input_size, color_mean), transform_anno=Anno_xml2list(voc_classes))
    val_dataset = VOCDataset(val_img_list, val_anno_list, phase="val", transform=DataTransform(
        input_size, color_mean), transform_anno=Anno_xml2list(voc_classes))
    print("트레이닝 셋 :", train_dataset.__len__())
    print("검증 셋 :", val_dataset.__len__())
    
    # DataLoader 작성
    batch_size = 4
    #data : torch.utils.data
    from utils.ssd import od_collate_fn
    train_dataloader = data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=od_collate_fn)
    val_dataloader = data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=od_collate_fn)
    dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}

    ssd_cfg = {
        'num_classes': 21,  # 배경 클래스를 포함한 총 클래스 수
        'input_size': 300,  # 화상의 입력 크기
        'bbox_aspect_num': [4, 6, 6, 6, 4, 4],  # 앵커박스 화면비의 종류
        'feature_maps': [38, 19, 10, 5, 3, 1],  # 각 source의 수
        'steps': [8, 16, 32, 64, 100, 300],  # 앵커박스의 크기
        'min_sizes': [30, 60, 111, 162, 213, 264],  # 앵커박스의 최소 면적
        'max_sizes': [60, 111, 162, 213, 264, 315],  # 앵커박스의 최대 면적
        'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    }

    from utils.ssd import SSD
    net = SSD(phase="train", cfg=ssd_cfg)
    # ssd의 vgg 부분에 가중치를 로드
    vgg_weights = torch.load('./weights/vgg16_reducedfc.pth')
    net.vgg.load_state_dict(vgg_weights)
    
    # 다른 부분은 He 초기화 적용
    net.extras.apply(weights_init)
    net.loc.apply(weights_init)
    net.conf.apply(weights_init)
    
    from utils.ssd import MultiBoxLoss
    # 손실함수의 설정
    criterion = MultiBoxLoss(jaccard_thresh=0.5, neg_pos=3, device=device)

    # 최적화 기법의 설정
    optimizer = optim.SGD(net.parameters(), lr=1e-3,
                          momentum=0.9, weight_decay=5e-4)

    # 학습 및 검증 실시
    num_epochs= 20
    train_model(net, dataloaders_dict, criterion, optimizer, num_epochs=num_epochs)