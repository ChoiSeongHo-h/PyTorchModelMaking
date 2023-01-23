import os.path as osp
import random
import xml.etree.ElementTree as ET
import time
import cv2
import numpy as np
import pandas as pd
from math import sqrt
from itertools import product
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data as data
from torch.autograd import Function
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from utils.data_augumentation import Compose, ConvertFromInts, ToAbsoluteCoords, PhotometricDistort, Expand, RandomSampleCrop, RandomMirror, ToPercentCoords, Resize, SubtractMeans
from utils.match import match

# 사용되는 파일들의 리스트를 작성
def make_datapath_list(rootpath):
    # 화상 파일과 어노테이션 파일 경로 템플릿을 작성
    imgpath_template = osp.join(rootpath, 'JPEGImages', '%s.jpg')
    annopath_template = osp.join(rootpath, 'Annotations', '%s.xml')

    # 훈련 및 검증 파일들의 파일 이름 취득
    train_id_names = osp.join(rootpath + 'ImageSets/Main/train.txt')
    val_id_names = osp.join(rootpath + 'ImageSets/Main/val.txt')

    # 리턴 리스트1
    train_img_list = list()
    train_anno_list = list()

    for line in open(train_id_names):
        file_id = line.strip()  # 공백과 줄바꿈 제거
        img_path = (imgpath_template % file_id)  # 화상의 경로
        anno_path = (annopath_template % file_id)  # 어노테이션의 경로
        train_img_list.append(img_path)
        train_anno_list.append(anno_path)

    # 리턴 리스트2
    val_img_list = list()
    val_anno_list = list()

    for line in open(val_id_names):
        file_id = line.strip()  # 공백과 줄바꿈 제거
        img_path = (imgpath_template % file_id)  # 화상의 경로
        anno_path = (annopath_template % file_id)  # 어노테이션의 경로
        val_img_list.append(img_path)
        val_anno_list.append(anno_path)

    return train_img_list, train_anno_list, val_img_list, val_anno_list
    
    
# VOC2012의 Dataset 작성
# PyTorch의 Dataset 클래스를 상속
class VOCDataset(data.Dataset):
    def __init__(self, img_list, anno_list, phase, transform, transform_anno):
        self.img_list = img_list
        self.anno_list = anno_list
        self.phase = phase  # train 또는 val
        self.transform = transform  # 전처리 클래스의 인스턴스
        self.transform_anno = transform_anno  # 어노테이션 클래스의 인스턴스

    # 이미지의 개수 반환
    def __len__(self):
        return len(self.img_list)

    # 텐서 이미지, np (정규화된 bb와 클래스), 원본 이미지 크기 반환
    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)
        return im, gt

    # 텐서 이미지, np (정규화된 bb와 클래스), 원본 이미지 크기 반환
    def pull_item(self, index):
        # 1. 화상 읽기
        image_file_path = self.img_list[index]
        img = cv2.imread(image_file_path)  # [높이][폭][색BGR]
        height, width, channels = img.shape  # 화상의 크기 취득

        # 2. xml 형식의 어노테이션 정보를 리스트에 저장
        anno_file_path = self.anno_list[index]
        # self.transform_anno : 어노테이션 클래스의 인스턴스
        anno_list = self.transform_anno(anno_file_path, width, height)

        # 3. 전처리 실시
        # self.transform : 전처리 클래스의 인스턴스
        img, boxes, labels = self.transform(
            img, self.phase, anno_list[:, :4], anno_list[:, 4])

        # 색상 채널의 순서가 BGR이므로 RGB로 순서를 변경
        # img[:, :, (2, 1, 0)] : 이미지의 마지막 차원(채널)을 RGB로
        # permute(2, 0, 1) : h w c를 c h w로
        img = torch.from_numpy(img[:, :, (2, 1, 0)]).permute(2, 0, 1)

        gt = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        # 텐서 이미지, np (정규화된 bb와 클래스), 원본 이미지 크기 반환
        return img, gt, height, width


# 입력 영상의 전처리 클래스
class DataTransform():
    # utils.data_augumentation의 augumentation 함수들을 사용해 트레이닝 시 augumentation
    # bb도 동시에 augumentation 적용
    # cv는 RGB가 아니라 BGR임에 유의
    def __init__(self, input_size, color_mean):
        self.data_transform = {
            'train': Compose([
                ConvertFromInts(),  # int를 float32로 변환
                ToAbsoluteCoords(),  # 어노테이션 데이터의 규격화를 반환
                PhotometricDistort(),  # 화상의 색조 등을 임의로 변화시킴
                Expand(color_mean),  # 화상의 캔버스를 확대
                RandomSampleCrop(),  # 화상 내의 특정 부분을 무작위로 추출
                RandomMirror(),  # 화상을 반전시킨다
                ToPercentCoords(),  # 어노테이션 데이터를 0-1로 규격화
                Resize(input_size),  # 화상 크기를 input_size × input_size로 변형
                SubtractMeans(color_mean)  # BGR 색상의 평균값을 뺀다
            ]),
            'val': Compose([
                ConvertFromInts(),  # int를 float로 변환
                Resize(input_size),  # 화상 크기를 input_size × input_size로 변형
                SubtractMeans(color_mean)  # BGR 색상의 평균값을 뺀다
            ])
        }

    def __call__(self, img, phase, boxes, labels):
        return self.data_transform[phase](img, boxes, labels)


# XML 어노테이션을 리스트 형식으로 변환하는 클래스
class Anno_xml2list(object):
    # 처음에 물체 리스트를 받아 초기화
    def __init__(self, classes):
        self.classes = classes

    # 한 이미지의 bb들을 정규화
    def __call__(self, xml_path, width, height):
        # 이미지상의 정규화된 bb들
        # [[xmin, ymin, xmax, ymax, label_ind], ... ]
        ret = []

        # xml 파일 로드
        xml = ET.parse(xml_path).getroot()

        # 화상 내에 있는 물체(object)의 수만큼 반복
        for obj in xml.iter('object'):

            # 어노테이션에서 감지가 difficult로 설정된 것은 제외
            difficult = int(obj.find('difficult').text)
            if difficult == 1:
                continue

            # 한 물체의 어노테이션을 저장하는 리스트
            # ret에 추가될 것임
            bndbox = []

            name = obj.find('name').text.lower().strip()  # 물체 이름
            bbox = obj.find('bndbox')  # 바운딩 박스 정보

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            for pt in (pts):
                # VOC는 원점이 (1,1)이므로 1을 빼서 (0, 0)으로 맞춰줌
                cur_pixel = int(bbox.find(pt).text) - 1

                # 0~1으로 정규화
                if pt == 'xmin' or pt == 'xmax':  # 폭으로 나눔
                    cur_pixel /= width
                else:  # 높이로 나눔
                    cur_pixel /= height

                bndbox.append(cur_pixel)

            # 어노테이션의 클래스명 index를 취득하여 추가
            # self.classes는 초기화시 입력받은 클래스 리스트
            label_idx = self.classes.index(name)
            bndbox.append(label_idx)

            ret += [bndbox]

        return np.array(ret)
 
 
# 미니배치를 만들기 위한 함수
def od_collate_fn(batch):
    # rets
    imgs = []
    targets = []
    for sample in batch:
        # sample[0] : 텐서형 img
        imgs.append(sample[0])
        # sample[1] : np 정규화된 어노테이션 gt
        targets.append(torch.FloatTensor(sample[1]))

    # imgs : 미니 배치 크기의 리스트
    # imgs[i] : torch.Size([3, 300, 300])
    # 리스트를 torch.Size([batch_num, 3, 300, 300]) 텐서로 변환
    # 0차원으로 쌓음
    imgs = torch.stack(imgs, dim=0)

    # targets : [n, 5] 리스트
    # n : gt 물체의 수
    return imgs, targets


# vgg모듈
def make_vgg():
    layers = []
    # 색 채널 수, 계속해서 바뀜
    in_channels = 3   

    # 숫자 : 각 신경망의 conv채널 수 (레이어의 개수)
    # M : 기본모드인 floor max pooling, 소수점 버림 -> 정수
    # MC : ceil max pooling, 소수점 올림 -> 정수
    # 뒤의 레이어는 아직 붙히지 못함(1024, 1024 층)
    cfg = [64, 64, 'M', 128, 128, 'M', 256, 256,
           256, 'MC', 512, 512, 512, 'M', 512, 512, 512]

    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'MC':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            # inplace는 이전의 값 공간을 그대로 사용하여 메모리를 아낀다.
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v

    # 뒤의 레이어를 추가로 붙혀줌
    # dilation을 적용
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
               
    return nn.ModuleList(layers)
    
    
# extras모듈
def make_extras():
    layers = []
    # vgg에서 출력된 source2의 채널 수
    in_channels = 1024

    # extra모듈의 합성곱층의 각 채널수(레이어 수)
    cfg = [256, 512, 128, 256, 128, 256, 128, 256]

    # 활성함수는 다른 곳에서 구현
    # 중간 중간 1x1 conv로 채널 수를 줄임
    layers += [nn.Conv2d(in_channels, cfg[0], kernel_size=(1))]
    layers += [nn.Conv2d(cfg[0], cfg[1], kernel_size=(3), stride=2, padding=1)]
    layers += [nn.Conv2d(cfg[1], cfg[2], kernel_size=(1))]
    layers += [nn.Conv2d(cfg[2], cfg[3], kernel_size=(3), stride=2, padding=1)]
    layers += [nn.Conv2d(cfg[3], cfg[4], kernel_size=(1))]
    layers += [nn.Conv2d(cfg[4], cfg[5], kernel_size=(3))]
    layers += [nn.Conv2d(cfg[5], cfg[6], kernel_size=(1))]
    layers += [nn.Conv2d(cfg[6], cfg[7], kernel_size=(3))]
    
    return nn.ModuleList(layers)
    

# loc, conf 모듈
# loc : 앵커박스의 delta를 출력
# conf : 앵커박스에 대한 각 클래스의 신뢰도 출력
# bbox_aspect_num : 앵커박스의 수
def make_loc_conf(num_classes=21, bbox_aspect_num=[4, 6, 6, 6, 4, 4]):
    # 텐서로 바꿔서 ret할 것임
    loc_layers = []
    conf_layers = []

    # VGG의 22층(1시작 idx, 21층 in 0시작 idx) conv4_3에서의 source1에 대한 합성곱층
    # 입력 512ch, 출력 앵커박스 종류 수 x 4(delta) ch
    loc_layers += [nn.Conv2d(512, bbox_aspect_num[0]
                             * 4, kernel_size=3, padding=1)]
    # 입력 512ch, 출력 앵커박스 종류 수 x 클래스 수 ch
    conf_layers += [nn.Conv2d(512, bbox_aspect_num[0]
                              * num_classes, kernel_size=3, padding=1)]

    # VGG의 최종층의 source2에 대한 합성곱층
    # 입력 1024ch, 출력 앵커박스 종류 수 x 4(delta) ch
    loc_layers += [nn.Conv2d(1024, bbox_aspect_num[1]
                             * 4, kernel_size=3, padding=1)]
    # 입력 1024ch, 출력 앵커박스 종류 수 x 클래스 수 ch
    conf_layers += [nn.Conv2d(1024, bbox_aspect_num[1]
                              * num_classes, kernel_size=3, padding=1)]

    # extra(source3)에 대한 합성곱층
    # 입력 512ch, 출력 앵커박스 종류 수 x 4(delta) ch
    loc_layers += [nn.Conv2d(512, bbox_aspect_num[2]
                             * 4, kernel_size=3, padding=1)]
    # 입력 512ch, 출력 앵커박스 종류 수 x 클래스 수 ch
    conf_layers += [nn.Conv2d(512, bbox_aspect_num[2]
                              * num_classes, kernel_size=3, padding=1)]

    # extra(source4)에 대한 합성곱층
    # 입력 256ch, 출력 앵커박스 종류 수 x 4(delta) ch
    loc_layers += [nn.Conv2d(256, bbox_aspect_num[3]
                             * 4, kernel_size=3, padding=1)]
    # 입력 256ch, 출력 앵커박스 종류 수 x 클래스 수 ch
    conf_layers += [nn.Conv2d(256, bbox_aspect_num[3]
                              * num_classes, kernel_size=3, padding=1)]

    # extra(source5)에 대한 합성곱층
    # 입력 256ch, 출력 앵커박스 종류 수 x 4(delta) ch
    loc_layers += [nn.Conv2d(256, bbox_aspect_num[4]
                             * 4, kernel_size=3, padding=1)]
    # 입력 256ch, 출력 앵커박스 종류 수 x 클래스 수 ch
    conf_layers += [nn.Conv2d(256, bbox_aspect_num[4]
                              * num_classes, kernel_size=3, padding=1)]

    # extra(source6)에 대한 합성곱층
    # 입력 256ch, 출력 앵커박스 종류 수 x 4(delta) ch
    loc_layers += [nn.Conv2d(256, bbox_aspect_num[5]
                             * 4, kernel_size=3, padding=1)]
    # 입력 256ch, 출력 앵커박스 종류 수 x 클래스 수 ch
    conf_layers += [nn.Conv2d(256, bbox_aspect_num[5]
                              * num_classes, kernel_size=3, padding=1)]

    return nn.ModuleList(loc_layers), nn.ModuleList(conf_layers)
    

# L2Norm 모듈
# conv4_3로부터의 출력을 scale=20의 L2Norm으로 정규화
# scale은 학습 가능한 파라미터
# 파이토치의 신경망 nn.Module을 상속
class L2Norm(nn.Module):
    # conv4_3의 출력이자 source1의 차원은 512 x 38 x 38
    def __init__(self, input_channels=512, scale=20):
        # 부모의 생성자
        super(L2Norm, self).__init__()
        # weight는 512개임(채널별로 가짐)
        self.weight = nn.Parameter(torch.Tensor(input_channels))
        self.scale = scale
        # weight와 scale로 파라미터 초기화
        self.reset_parameters()
        # 분모가 0이되지 않게
        self.eps = 1e-10

    # weight와 scale로 파라미터 초기화
    def reset_parameters(self):
        # init : torch.nn.init
        init.constant_(self.weight, self.scale)

    # [batch_num, 512, 38, 38]의 conv4_3의 출력을 채널 방향으로 정규화
    def forward(self, x):
        # 각 채널에서의 38×38개의 특징량의 채널 방향의 제곱합을 계산하고 이것으로 원본을 나누어줌
        # x : [batch_num, 512, 38, 38]
        # x.pow(2) : 각 요소를 모두 제곱
        # x.pow(2).sum(dim=1, keepdim=True) : 1차원(채널방향)으로 더함
        # -> [batch_num, 1, 38, 38]
        # norm : [batch_num, 1, 38, 38]
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
        x = torch.div(x, norm)

        # 계수는 [512]
        # [512]를 [batch_num, 512, 38, 38]까지 변형
        # self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        # -> [1, 512, 1, 1]
        # [1, 512, 1, 1].expand_as(x)
        # -> [batch_num, 512, 38, 38]
        weights = self.weight.unsqueeze(
            0).unsqueeze(2).unsqueeze(3).expand_as(x)
        # 이를 element wise 곱해줌(채널별로 상수인 weight가 곱해지는 격)
        out = weights * x

        return out


# 앵커 박스를 출력하는 클래스
# 다음과 같은 설정을 받아 초기화
'''
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
'''
class DBox(object):
    def __init__(self, cfg):
        super(DBox, self).__init__()

        # 화상의 입력 크기 300
        self.image_size = cfg['input_size']
        # 각 source의 수 [38, 19, 10, 5, 3, 1]
        self.feature_maps = cfg['feature_maps']
        # source의 종류 수 6
        self.num_priors = len(cfg["feature_maps"])
        # 앵커박스의 크기 [8, 16, 32, 64, 100, 300]
        self.steps = cfg['steps']
        # 앵커박스의 최소 면적 [30, 60, 111, 162, 213, 264]
        self.min_sizes = cfg['min_sizes']
        # 앵커박스의 최대 면적 [60, 111, 162, 213, 264, 315]
        self.max_sizes = cfg['max_sizes']
        # 앵커박스의 종횡비 [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
        self.aspect_ratios = cfg['aspect_ratios']

    # 앵커박스 출력
    def make_dbox_list(self):
        # 앵커박스를 모음
        mean = []
        # feature_maps : [38, 19, 10, 5, 3, 1]
        # k : 인덱스
        # f : 38, 19, ...
        for k, f in enumerate(self.feature_maps):
            # (0~37), (0~19), ...으로 조합을 만듦
            # -> (0~37) : (0, 0) ~ (37, 37), 총 38 x 38개 그리드
            # (x, y)를 구성하려하는 것임
            # -> (j, i)로 표현
            for i, j in product(range(f), repeat=2):
                # image_size : 300
                # steps : [8, 16, 32, 64, 100, 300]
                # f_k : [38, 19, 10, 5, 3, 1] 유사
                f_k = self.image_size / self.steps[k]

                # 앵커박스의 중심좌표, 0~1로 정규화
                # (i, j) : (0, 0) ~ (37, 37)
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                # 화면비 1의 작은, 정규화된 앵커박스 [cx, cy, width, height]
                # min_sizes : [30, 60, 111, 162, 213, 264]
                s_k = self.min_sizes[k]/self.image_size
                mean += [cx, cy, s_k, s_k]

                # 화면비 1의 큰, 정규화된 앵커박스 [cx, cy, width, height]
                # max_sizes : [60, 111, 162, 213, 264, 315],
                s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]

                # 그 외 화면비의 정규화된 앵커박스 [cx, cy, width, height]
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
                    mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]

        # 앵커박스를를 텐서로 변환 [8732, 4]
        # view(-1, 4) : 2차원, 마지막 차원을 4로 고정, 1번째 차원을 맞춤
        output = torch.Tensor(mean).view(-1, 4)

        # 앵커박스가 화상 밖으로 돌출되는 것을 막기 위해 자름
        output.clamp_(max=1, min=0)

        return output


# delta를 이용해 엥커박스를 bb로 변환
def decode(loc, dbox_list):
    # boxes : bb
    # dbox_list : 앵커박스, [8732, 4]
    # -> [8732][cx, cy, width, height]
    # dbox_list[:, :2] : 8732개 앵커박스의 0번째 1번째 원소
    # -> 앵커박스의 중점
    # dbox_list[:, 2:] : 8732개 앵커박스의 2번째 3번째 원소
    # -> width, height
    # loc : delta, [8732, 4]
    # -> [8732][Δcx, Δcy, Δwidth, Δheight]
    # 공식을 이용해 bb를 구함
    # cat : [8732, 2], [8732, 2]를 1번째차원으로 합쳐 [8732, 4]로 만듦
    # boxes : [8732, 4]
    boxes = torch.cat((
        dbox_list[:, :2] + loc[:, :2] * 0.1 * dbox_list[:, 2:],
        dbox_list[:, 2:] * torch.exp(loc[:, 2:] * 0.2)), dim=1)

    # bb를 [cx, cy, width, height]에서 [xmin, ymin, xmax, ymax]으로 변경
    boxes[:, :2] -= boxes[:, 2:] / 2  # 좌표 (xmin,ymin)로 변환
    boxes[:, 2:] += boxes[:, :2]  # 좌표 (xmax,ymax)로 변환

    return boxes


# NMS 수행
# boxes는 미리 신뢰도 임계값 테스트를 거쳐 축소된 상태여야 함
def nm_suppression(boxes, scores, overlap=0.45, top_k=200):
    # nms를 통과한 bb 수
    count = 0
    # scores : [신뢰도 임계값(0.01)을 넘은 bb 수]
    # keep : [신뢰도 임계값(0.01)을 넘은 bb 수], long 형, 요소는 전부 0
    keep = scores.new(scores.size(0)).zero_().long()

    # boxes : [신뢰도 임계값(0.01)을 넘은 bb 수, 4]
    # x1 ~ y2 : [신뢰도 임계값(0.01)을 넘은 bb 수]
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    # area : [신뢰도 임계값(0.01)을 넘은 bb 수], bb의 면적
    area = torch.mul(x2 - x1, y2 - y1)

    # [신뢰도 임계값(0.01)을 넘은 bb 수, 4]만큼 텐서를 복사해둠
    tmp_x1 = boxes.new()
    tmp_y1 = boxes.new()
    tmp_x2 = boxes.new()
    tmp_y2 = boxes.new()
    tmp_w = boxes.new()
    tmp_h = boxes.new()

    # v : [신뢰도 임계값(0.01)을 넘은 bb 수], 오름차순으로 정렬된 socres
    # idx : [신뢰도 임계값(0.01)을 넘은 bb 수], v 요소의 scores에서 idx
    v, idx = scores.sort(0)

    # 상위 top_k개(200개)의 bb idx를 R꺼냄
    # 200개 존재하지 않는 경우도 있음
    # -> 그래도 문제가 발생하지 않음
    # 출력 또한 200보다 적을 수 있음, 하지만 처음에 크기가 고정되어있긴 함(keep)
    idx = idx[-top_k:]

    # numel : element의 개수
    while idx.numel() > 0:
        # 가장 신뢰도 높은 인덱스를 하나 선택
        i = idx[-1]

        # keep에 bb의 idx를 저장
        # 이것과 크게 겹치는 bb를 삭제할 것임
        # conf의 내림차순으로 쌓임
        keep[count] = i
        count += 1

        # 마지막 bb였었다면 break
        if idx.size(0) == 1:
            break

        # idx를 pop back
        idx = idx[:-1]
       
        # x1 ~ y2 : [bb 수]
        # index_select : 0번째 차원에서 idx인 인덱스를 뽑아 tmp_x1 ~ tmp_y2에 저장
        # -> 크기가 다르다면 그 전에 초기화해야 함
        # idx : [bb 수 - 1] 가장 신뢰도가 높은것이 빠짐
        tmp_x1.resize_(0)
        tmp_y1.resize_(0)
        tmp_x2.resize_(0)
        tmp_y2.resize_(0)
        torch.index_select(x1, 0, idx, out=tmp_x1)
        torch.index_select(y1, 0, idx, out=tmp_y1)
        torch.index_select(x2, 0, idx, out=tmp_x2)
        torch.index_select(y2, 0, idx, out=tmp_y2)

        # i를 제외한 임시 bb의 크기를 최대 i번째 bb의 크기로 제한
        # i번째 bb보다 왼쪽, 위쪽, 오른쪽, 아래쪽으로 튀어나올 수 없음
        tmp_x1 = torch.clamp(tmp_x1, min=x1[i])
        tmp_y1 = torch.clamp(tmp_y1, min=y1[i])
        tmp_x2 = torch.clamp(tmp_x2, max=x2[i])
        tmp_y2 = torch.clamp(tmp_y2, max=y2[i])

        # 임시 w, h를 다시 구함
        tmp_w.resize_as_(tmp_x2)
        tmp_h.resize_as_(tmp_y2)
        tmp_w = tmp_x2 - tmp_x1
        tmp_h = tmp_y2 - tmp_y1
        tmp_w = torch.clamp(tmp_w, min=0.0)
        tmp_h = torch.clamp(tmp_h, min=0.0)

        # i와 겹치는 부분의 면적을 구함, [bb 수 - 1]
        # -> i만큼 크기를 줄였으므로
        inter = tmp_w*tmp_h

        # IoU = inter / union
        # union = (area(a) + area(b) - inter)
        # area : [bb 수]
        # idx : [bb 수 - 1]
        # 원래 크기를 구함
        # rem_areas : [bb 수 - 1], 가장 신뢰도 높은 bb가 하나가 빠진것의 원래 면적
        rem_areas = torch.index_select(area, 0, idx)
        # union = (area(원래) - inter + area(신뢰도 높음))
        union = (rem_areas - inter) + area[i]
        IoU = inter/union

        # IoU가 overlap보다 작은 idx만 남긴다
        # le : <=, Less than or Equal to
        # overlap : 0.45
        # idx의 크기가 줄어듦
        idx = idx[IoU.le(overlap)]

    return keep, count


# 순전파 연산을 수행하는 클래스
class Detect():
    def __init__(self, conf_thresh=0.01, top_k=200, nms_thresh=0.45):
        self.softmax = nn.Softmax(dim=-1)
        # NMS에 넘기기 전의 신뢰도 임계값
        self.conf_thresh = conf_thresh
        # NMS에서 신뢰도 top_k만큼만 사용
        self.top_k = top_k
        # IoU 0.45 이상이면 같은 bb로 간주
        self.nms_thresh = nms_thresh

    def __call__(self, loc_data, conf_data, dbox_list):
        # loc_data : [batch_num, 8732, 4]
        num_batch = loc_data.size(0) # 배치수 = 32?
        num_dbox = loc_data.size(1) # 앵커박스 수 = 8732
        # conf_data : [batch_num, 8732, num_classes]
        num_classes = conf_data.size(2)  # 클래스 수 = 21

        # conf_data : [batch_num, 8732, num_classes]
        # self.softmax = nn.Softmax(dim=-1)
        # 마지막 차원(num_classes)에 확률 정규화
        conf_data = self.softmax(conf_data)

        # [batch_num, 21, 200, 5]
        # 5 : [conf, xmin, ymin, w, h]
        # ret
        output = torch.zeros(num_batch, num_classes, self.top_k, 5)

        # [batch_num, 8732, num_classes] -> [batch_num, num_classes, 8732]
        conf_preds = conf_data.transpose(2, 1)

        # 미니배치의 요소마다(이미지마다)
        for i in range(num_batch):
            # delta와 앵커박스로 bb를 구함
            # -> [8732][xmin, ymin, xmax, ymax] 를 구한다
            # loc_data : [batch_num, 8732, 4]
            # loc_data[i] : [8732, 4]
            # decoded_boxes : [8732, 4]
            decoded_boxes = decode(loc_data[i], dbox_list)

            # 신뢰도 conf의 복사본을 작성
            # conf_preds : [batch_num, num_classes, 8732]
            # conf_scores : [num_classes, 8732]
            conf_scores = conf_preds[i].clone()

            # 클래스별 루프
            # 배경 클래스의 index인 0 포함하지 않음
            for cl in range(1, num_classes):
                # conf_scores[cl] : [8732]
                # gt : >
                # c_mask : [8732], 0 or 1
                c_mask = conf_scores[cl].gt(self.conf_thresh)

                # scores : [임계값을 넘은 bb 수]
                scores = conf_scores[cl][c_mask]

                # nelement : 요소를 셈
                # 없다면 continue
                if scores.nelement() == 0:
                    continue

                # c_mask를 decoded_boxes에 적용할 수 있도록 크기를 변경
                # c_mask : [8732]
                # decoded_boxes : [8732, 4]
                # [8732] -> [8732, 1] -> [8732, 4]
                # l_mask : [8732, 4]
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)

                # decoded_boxes에 신뢰도 테스트 적용하여 제거
                # decoded_boxes[l_mask] : 1차원
                # boxes : [임계값을 넘은 bb 수, 4]
                boxes = decoded_boxes[l_mask].view(-1, 4)

                # NMS, 겹치는 bb 제거
                # count: NMS를 통과한 bb 수
                # ids : [임계값을 넘은 bb 수]
                # -> NMS를 통과한 것의 boxes에서의 idx, conf의 내림차순으로 정렬됨
                # -> NMS를 통과하지 못한 것은 배열의 뒤에 0으로 저장됨
                # --> 임계값을 넘은 bb 수 >= NMS를 통과한 것의 수
                ids, count = nm_suppression(
                    boxes, scores, self.nms_thresh, self.top_k)

                # output : [batch_num, 21, 200, 5]
                # output[i, cl, :count] : [NMS를 통과한 bb 수, 5], output의 일부에 다음을 저장
                # scores : [임계값을 넘은 bb 수], 신뢰도
                # ids : [임계값을 넘은 bb 수]
                # ids[:count] : [NMS를 통과한 bb 수]
                # scores[ids[:count]] : [NMS를 통과한 bb 수], 신뢰도
                # -> [ids[:count]] 연산을 통해 ids[:count]의 요소에 해당하는 인덱스들이 scores에서 선택됨
                # scores[ids[:count]].unsqueeze(1) : [NMS를 통과한 bb 수, 1], 신뢰도
                # boxes : [임계값을 넘은 bb 수, 4], bb 정보
                # boxes[ids[:count]] : [NMS를 통과한 bb 수, 4], bb 정보
                # cat : [NMS를 통과한 bb 수, 1+4] = [NMS를 통과한 bb 수, 5]
                output[i, cl, :count] = torch.cat((scores[ids[:count]].unsqueeze(1),
                                                   boxes[ids[:count]]), 1)

        # [batch_num, 21, 200, 5]
        return output


# 메인이 되는 SSD 클래스
# nn.Module을 상속
class SSD(nn.Module):
    def __init__(self, phase, cfg):
        super(SSD, self).__init__()

        # train or inference
        self.phase = phase
        # 21
        self.num_classes = cfg["num_classes"]

        # 네트워크 구성
        self.vgg = make_vgg()
        self.extras = make_extras()
        self.L2Norm = L2Norm()
        self.loc, self.conf = make_loc_conf(
            cfg["num_classes"], cfg["bbox_aspect_num"])

        # 앵커박스 작성
        dbox = DBox(cfg)
        self.dbox_list = dbox.make_dbox_list()

        # 추론시에는 Detect 클래스를 추가
        if phase == 'inference':
            self.detect = Detect()

    def forward(self, x):
        # loc와 conf로의 입력 source1-6
        sources = list()
        # loc의 출력
        loc = list()
        # conf의 출력
        conf = list()

        # vgg의 conv4_3까지 계산(0 ~ 22)
        # x : 네트워크 입력
        for k in range(23):
            x = self.vgg[k](x)

        # conv4_3의 출력을 L2Norm에 입력
        # 출력된 source1을 sources에 추가
        source1 = self.L2Norm(x)
        sources.append(source1)

        # vgg를 끝까지 계산하여 source2를 sources에 추가
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)

        # extras의 conv와 ReLU를 계산
        # k : idx
        # v : a layer
        for k, v in enumerate(self.extras):
            # extras의 내부에는 relu가 구현되어있지 않음
            x = F.relu(v(x), inplace=True)
            # source3~6을 sources에 추가
            if k % 2 == 1:
                sources.append(x)

        # zip : 여러 리스트 요소를 동시에 취득
        # sources, loc, conf 모두 6개로 구성
        # -> 루프가 6회 실시됨
        for (x, l, c) in zip(sources, self.loc, self.conf):
            # l(x) : conv를 1번 거친 loc 출력
            # -> [batch_num, 화면비 종류(4 or 6) x 4, 입력 h, 입력 w]
            # permute : [batch_num, 입력 h, 입력 w, 화면비 종류 x 4]
            # contiguous : 메모리 상에 요소를 연속적으로 배치
            # -> 추후 view를 실행하기 위해서 메모리상에 연속적으로 배치되어야 함
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            # c(x) : conv를 1번 거친 conf 출력
            # -> [batch_num, 화면비 종류 x 클래스 수(21), 입력 h, 입력 w]
            # permute : [batch_num, 입력 h, 입력 w, 화면비 종류 x 클래스 수(21)]
            # -> 결국 가변적인 것을 가장 뒤 차원으로 빼버림
            # --> 화면비 종류가 가변적
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        # o in loc : [batch_num, 입력 h, 입력 w, 화면비 종류 x 4]
        # o.view(o.size(0), -1) : [batch_num, (입력 h) x (입력 w) x (화면비 종류) x (4)]
        # cat : [batch_num, 34928]
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        # o in conf : [batch_num, 입력 h, 입력 w, 화면비 종류 x 클래스 수(21)]
        # o.view(o.size(0), -1) : [batch_num, (입력 h) x (입력 w) x (화면비 종류) x (클래스 수)]
        # cat : [batch_num, 183372]
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        # loc : [batch_num, 8732, 4]
        # -> 리스트에서 텐서로 바뀌었음, 각 bb의 delta를 의미
        loc = loc.view(loc.size(0), -1, 4)
        # conf : [batch_num, 8732, 21]
        # -> 리스트에서 텐서로 바뀌었음, 각 bb의 클래스별 신뢰도를 의미
        conf = conf.view(conf.size(0), -1, self.num_classes)

        # dbox_list = dbox.make_dbox_list()
        # -> [8732, 4]
        # --> 고정된 앵커박스일 뿐이므로 배치 사이즈는 중요하지 않음
        output = (loc, conf, self.dbox_list)

        # 추론시 Detect의 forward를 추가적으로 실행
        # NMS등을 수행
        if self.phase == "inference":
            with torch.no_grad():
                # ret : [batch_num, 21, 200, 5]
                return self.detect(output[0], output[1], output[2])

        # 학습시
        else:  
            # ret (loc, conf, dbox_list) 튜플
            return output


# loss에 대한 클래스
class MultiBoxLoss(nn.Module):
    def __init__(self, jaccard_thresh=0.5, neg_pos=3, device='cpu'):
        super(MultiBoxLoss, self).__init__()
        # 앵커박스와 GT의 IOU
        self.jaccard_thresh = jaccard_thresh
        # Negative DBox를 학습에 3배만 사용할 것임
        self.negpos_ratio = neg_pos
        self.device = device

    # loss 계산
    def forward(self, predictions, targets):
        # SSD 모델의 출력(예측)
        # loc_data : [batch_num, 8732, 4], 신경망에 의해 추론된 위치
        # conf_data : [batch_num, 8732, 21], 신경망에 의해 추론된 클래스별 신뢰도
        # dbox_list : [8732, 4]
        loc_data, conf_data, dbox_list = predictions

        num_batch = loc_data.size(0)  # 미니배치 크기 = 32?
        num_dbox = loc_data.size(1)  # 앵커박스 수 = 8732
        num_classes = conf_data.size(2)  # 클래스 수 = 21

        # conf_t_label : [batch_num, 8732], 앵커박스별로 겹치는 GT의 클래스 라벨
        conf_t_label = torch.LongTensor(num_batch, num_dbox).to(self.device)
        # loc_t : [batch_num, 8732, 4], 겹치는 GT의 위치
        loc_t = torch.Tensor(num_batch, num_dbox, 4).to(self.device)

        # 각 이미지별로
        for idx in range(num_batch):
            # targets : [num_batch, num_objs, 5], GT 정보
            # targets[idx][:, :-1] : [num_objs, 4], GT 위치 정보
            truths = targets[idx][:, :-1].to(self.device)
            # targets[idx][:, -1] : [num_objs], GT 클래스 정보
            labels = targets[idx][:, -1].to(self.device)

            # dbox_list : [8732, 4]
            dbox = dbox_list.to(self.device)

            variance = [0.1, 0.2]
            # match 실행
            # -> 이는 리뷰하지 않음
            # -> 앵커박스와 겹치는 GT를 찾음
            # conf_t_label : [batch_num, 8732], labels가 이용되어 겹치는 GT 라벨이 저장
            # -> 만약 IOU가 0.5보다 작으면 conf_t_label은 배경 클래스인 0이 지정
            # loc_t : [batch_num, 8732, 4], truths가 이용되어 겹치는 GT 위치가 저장
            match(self.jaccard_thresh, truths, dbox,
                  variance, labels, loc_t, conf_t_label, idx)

        # pos_mask : [num_batch, 8732], 앵커박스가 GT와 겹쳤는지 여부, 0 or 1
        pos_mask = conf_t_label > 0

        # loc_data : [batch_num, 8732, 4], 신경망에 의해 추론된 위치
        pos_idx = pos_mask.unsqueeze(pos_mask.dim()).expand_as(loc_data)

        # loc_p : [GT가 매칭된 앵커박스 수, 4], 신경망의 예측 위치
        loc_p = loc_data[pos_idx].view(-1, 4)
        # loc_t : [GT가 매칭된 앵커박스 수, 4], 겹치는 GT 위치
        loc_t = loc_t[pos_idx].view(-1, 4)
        # 즉, 앵커박스에서 가장 가까운 물체를 찾게끔 학습됨

        # F : torch.nn.functional
        # Huber로 loss를 계산
        # reduction='sum' : 더하여 스칼라로 만듦
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')
        
        # conf_data : [batch_num, 8732, 21], 신경망에 의해 추론된 클래스별 신뢰도
        # batch_conf : [batch_num x 8732, 21]
        # -> 크로스 엔트로피를 구하기 위해 차원 변경
        batch_conf = conf_data.view(-1, num_classes)

        # conf_t_label.view(-1) : [batch_num x 8732], 겹쳐진 GT 라벨
        # -> 이를 예측 정보인 batch_conf와 비교
        # reduction='none' : 합을 취하지 않고 차원을 보존
        # loss_c : [batch_num x 8732]
        loss_c = F.cross_entropy(
            batch_conf, conf_t_label.view(-1), reduction='none')

        # pos_mask : [num_batch, 8732], 앵커박스가 GT와 겹쳤는지 여부, 0 or 1
        # num_pos : [num_batch, 1], 미니배치별 GT를 감지한 앵커박스 수
        # -> keepdim=True을 통해 [num_batch]이 아닌 [num_batch, 1]이 됨
        num_pos = pos_mask.long().sum(1, keepdim=True)
        # loss_c.view : [batch_num x 8732] -> [num_batch, 8732]
        # Hard Negative Mining을 위한 임시적인 loss 계산
        loss_c = loss_c.view(num_batch, -1)
        # GT와 겹쳐진 앵커박스는 크로스 엔트로피 손실을 0으로 함
        # -> 배경에 대해 Hard Negative Mining을 실시하여 loss를 구하기 위해 겹쳐진것은 배제
        loss_c[pos_mask] = 0

        # loss_c : [num_batch, 8732]
        # -> ex [100, 700, 200, 50, 10]
        # loss_idx : [num_batch, 8732]
        # -> 내림차순으로 정렬한것의 idx
        # -> ex [1, 2, 0, 3, 4]
        _, loss_idx = loss_c.sort(1, descending=True)
        # idx_rank : [num_batch, 8732], loss_c 요소의 loss 순위
        # -> ex [2, 0, 1, 3, 4]
        # --> 즉, [100, 700, 200, 50, 10]의 순위
        _, idx_rank = loss_idx.sort(1)

        # 학습에 사용할 Negative DBox의 수를 최대, 미니배치별 GT를 감지한 앵커박스 수의 3배로 제한
        # num_pos : [num_batch, 1], 미니배치별 GT를 감지한 앵커박스 수
        # num_neg : [num_batch, 1], 미니배치별 학습에 사용할 Negative DBox의 수
        num_neg = torch.clamp(num_pos*self.negpos_ratio, max=num_dbox)

        # idx_rank는 배경 앵커박스의 loss 순위
        # 배경 앵커박스의 상한 num_neg보다 순위가 낮은(손실이 큰) 앵커박스를 구하는 마스크를 제작
        # idx_rank : [num_batch, 8732]
        # neg_mask : [num_batch, 8732], 배경 학습에 사용될 앵커박스인지 여부, 0 or 1
        neg_mask = idx_rank < (num_neg).expand_as(idx_rank)

        # pos_mask : [num_batch, 8732], 앵커박스가 GT와 겹쳤는지 여부, 0 or 1
        # conf_data : [batch_num, 8732, 21], 신경망에 의해 추론된 클래스별 신뢰도
        # pos_idx_mask : [batch_num, 8732, 21], Positive DBox의 conf를 추출하는 마스크
        pos_idx_mask = pos_mask.unsqueeze(2).expand_as(conf_data)
        # neg_mask : [num_batch, 8732], 배경 학습에 사용될 앵커박스인지 여부, 0 or 1
        # neg_idx_mask : [batch_num, 8732, 21], Negative DBox의 conf를 추출하는 마스크
        # -> Hard Negative Mining으로 추출된 것
        neg_idx_mask = neg_mask.unsqueeze(2).expand_as(conf_data)

        # gt 학습에 사용될 앵커와 배경 학습에 사용될 앵커를 추출
        # conf_data : [batch_num, 8732, 21], 신경망에 의해 추론된 모델별 신뢰도
        # pos_idx_mask : [batch_num, 8732, 21], Positive DBox의 conf를 추출하는 마스크
        # neg_idx_mask : [batch_num, 8732, 21], Negative DBox의 conf를 추출하는 마스크
        # pos_idx_mask+neg_idx_mask : [batch_num, 8732, 21], 학습에 사용될 앵커박스인지 여부
        # conf_hnm : [pos+neg, 21], 학습에 사용될 앵커박스
        conf_hnm = conf_data[(pos_idx_mask+neg_idx_mask).gt(0)
                             ].view(-1, num_classes)

        # conf_t_label : [batch_num, 8732], 앵커박스와 겹쳐진 GT의 클래스 라벨
        # pos_mask : [num_batch, 8732], 앵커박스가 GT와 겹쳤는지 여부, 0 or 1
        # neg_mask : [num_batch, 8732], 배경 학습에 사용될 앵커박스인지 여부, 0 or 1
        # conf_t_label_hnm : [pos+neg], 학습에 사용될 앵커박스 라벨
        conf_t_label_hnm = conf_t_label[(pos_mask+neg_mask).gt(0)]

        # confidence의 진짜 손실함수를 계산
        loss_c = F.cross_entropy(conf_hnm, conf_t_label_hnm, reduction='sum')

        # num_pos : [num_batch, 1], 미니배치별 GT를 감지한 앵커박스 수
        # N : GT를 감지한 앵커박스 총 수
        N = num_pos.sum()
        loss_l /= N
        loss_c /= N

        return loss_l, loss_c
