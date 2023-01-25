import os.path as osp
from PIL import Image
import torch.utils.data as data
import torch
import torch.nn as nn
import torch.nn.functional as F

# augumentation을 위한 import
from utils.data_augumentation import Compose, Scale, RandomRotation, RandomMirror, Resize, Normalize_Tensor


#학습및 검증을 위한 데이터파일 리스트 생성
def make_datapath_list(rootpath):
    # 화상 파일과 어노테이션 파일의 경로 템플릿을 작성
    imgpath_template = osp.join(rootpath, 'JPEGImages', '%s.jpg')
    annopath_template = osp.join(rootpath, 'SegmentationClass', '%s.png')

    # 파일에서 리스트를 읽어옴
    train_id_names = osp.join(rootpath + 'ImageSets/Segmentation/train.txt')
    val_id_names = osp.join(rootpath + 'ImageSets/Segmentation/val.txt')

    # rets
    train_img_list = list()
    train_anno_list = list()

    for line in open(train_id_names):
        # 공백과 줄바꿈 제거
        file_id = line.strip()  
        # 화상의 경로
        img_path = (imgpath_template % file_id)  
        # 어노테이션의 경로
        anno_path = (annopath_template % file_id)  
        train_img_list.append(img_path)
        train_anno_list.append(anno_path)

    # rets
    val_img_list = list()
    val_anno_list = list()

    for line in open(val_id_names):
        # 공백과 줄바꿈 제거
        file_id = line.strip()  
        # 화상의 경로
        img_path = (imgpath_template % file_id)  
        # 어노테이션의 경로
        anno_path = (annopath_template % file_id)  
        val_img_list.append(img_path)
        val_anno_list.append(anno_path)

    return train_img_list, train_anno_list, val_img_list, val_anno_list


# 데이터 전처리
class DataTransform():
    # color_mean : (R, G, B)
    # color_std : (R, G, B)
    def __init__(self, input_size, color_mean, color_std):
        self.data_transform = {
            'train': Compose([
                Scale(scale=[0.5, 1.5]),  # 화상의 확대
                RandomRotation(angle=[-10, 10]),  # 회전
                RandomMirror(),  # 랜덤 플립
                Resize(input_size),  # 리사이즈(input_size)
                Normalize_Tensor(color_mean, color_std)  # 정규화, 텐서화
            ]),
            'val': Compose([
                Resize(input_size),  # 리사이즈(input_size)
                Normalize_Tensor(color_mean, color_std)  # 정규화, 텐서화
            ])
        }

    def __call__(self, phase, img, anno_class_img):
        return self.data_transform[phase](img, anno_class_img)


# PyTorch의 Dataset 클래스를 상속
class VOCDataset(data.Dataset):
    # 데이터 파일 리스트와 전처리 클래스의 인스턴스를 받아옴
    def __init__(self, img_list, anno_list, phase, transform):
        self.img_list = img_list
        self.anno_list = anno_list
        self.phase = phase
        self.transform = transform

    #길이 반환
    def __len__(self):
        return len(self.img_list)

    # 텐서형 이미지와 어노테이션 반환
    def __getitem__(self, index):
        img, anno_class_img = self.pull_item(index)
        return img, anno_class_img

    # 텐서형 이미지와 어노테이션 반환
    def pull_item(self, index):
        # 원본
        image_file_path = self.img_list[index]
        # [높이][폭][색RGB]
        # -> PIL
        img = Image.open(image_file_path)

        # 어노테이션
        anno_file_path = self.anno_list[index]
        # [높이][폭]
        anno_class_img = Image.open(anno_file_path)

        # augumentation, 정규화, 텐서화
        img, anno_class_img = self.transform(self.phase, img, anno_class_img)

        return img, anno_class_img


# conv -> bn -> relu
class conv2DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, bias):
        super(conv2DBatchNormRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride, padding, dilation, bias=bias)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        outputs = self.relu(x)
        return outputs


# 3(conv -> bn -> relu) -> maxpool
class FeatureMap_convolution(nn.Module):
    def __init__(self):
        super(FeatureMap_convolution, self).__init__()

        # 합성곱 층1
        in_channels, out_channels, kernel_size, stride, padding, dilation, bias = 3, 64, 3, 2, 1, 1, False
        self.cbnr_1 = conv2DBatchNormRelu(
            in_channels, out_channels, kernel_size, stride, padding, dilation, bias)

        # 합성곱 층2
        in_channels, out_channels, kernel_size, stride, padding, dilation, bias = 64, 64, 3, 1, 1, 1, False
        self.cbnr_2 = conv2DBatchNormRelu(
            in_channels, out_channels, kernel_size, stride, padding, dilation, bias)

        # 합성곱 층3
        in_channels, out_channels, kernel_size, stride, padding, dilation, bias = 64, 128, 3, 1, 1, 1, False
        self.cbnr_3 = conv2DBatchNormRelu(
            in_channels, out_channels, kernel_size, stride, padding, dilation, bias)

        # MaxPooling 층
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.cbnr_1(x)
        x = self.cbnr_2(x)
        x = self.cbnr_3(x)
        outputs = self.maxpool(x)
        return outputs


# bottleNeckPSP -> n(bottleNeckldentifyPSP)
# nn.Sequential를 상속하여서 nn.Module을 상속한 것과 다르게 forward를 구현하지 않아도 됨
class ResidualBlockPSP(nn.Sequential):
    def __init__(self, n_blocks, in_channels, mid_channels, out_channels, stride, dilation):
        super(ResidualBlockPSP, self).__init__()

        # bottleNeckPSP
        self.add_module(
            "block1",
            bottleNeckPSP(in_channels, mid_channels,
                          out_channels, stride, dilation)
        )

        # n(bottleNeckldentifyPSP)
        for i in range(n_blocks - 1):
            self.add_module(
                "block" + str(i+2),
                bottleNeckIdentifyPSP(
                    out_channels, mid_channels, stride, dilation)
            )


# conv -> bn
class conv2DBatchNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, bias):
        super(conv2DBatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride, padding, dilation, bias=bias)
        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        outputs = self.batchnorm(x)
        return outputs
        
        
# 2(conv2DBatchNormRelu) -> conv2DBatchNorm
# -> skipped(conv2DBatchNorm) ->
class bottleNeckPSP(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, stride, dilation):
        super(bottleNeckPSP, self).__init__()
        
        # 2(conv2DBatchNormRelu) -> conv2DBatchNorm
        self.cbr_1 = conv2DBatchNormRelu(
            in_channels, mid_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)
        self.cbr_2 = conv2DBatchNormRelu(
            mid_channels, mid_channels, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.cb_3 = conv2DBatchNorm(
            mid_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)

        # skipped(conv2DBatchNorm)
        self.cb_residual = conv2DBatchNorm(
            in_channels, out_channels, kernel_size=1, stride=stride, padding=0, dilation=1, bias=False)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 2(conv2DBatchNormRelu) -> conv2DBatchNorm
        conv = self.cb_3(self.cbr_2(self.cbr_1(x)))
        # skipped(conv2DBatchNorm)
        residual = self.cb_residual(x)
        return self.relu(conv + residual)


# 2(conv2DBatchNormRelu) -> conv2DBatchNorm
# -> skipped() ->
class bottleNeckIdentifyPSP(nn.Module):
    def __init__(self, in_channels, mid_channels, stride, dilation):
        super(bottleNeckIdentifyPSP, self).__init__()

        # 2(conv2DBatchNormRelu) -> conv2DBatchNorm
        self.cbr_1 = conv2DBatchNormRelu(
            in_channels, mid_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)
        self.cbr_2 = conv2DBatchNormRelu(
            mid_channels, mid_channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=False)
        self.cb_3 = conv2DBatchNorm(
            mid_channels, in_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 2(conv2DBatchNormRelu) -> conv2DBatchNorm
        conv = self.cb_3(self.cbr_2(self.cbr_1(x)))
        # -> skipped() ->
        residual = x
        return self.relu(conv + residual)


# 4(AdaptiveAvgPool2d) -> 4(conv2DBatchNormRelu) -> 4(upsampling)
class PyramidPooling(nn.Module):
    def __init__(self, in_channels, pool_sizes, height, width):
        super(PyramidPooling, self).__init__()

        # forward에 사용하는 화상 크기
        self.height = height
        self.width = width

        # 각 합성곱 층의 출력 채널 수
        out_channels = int(in_channels / len(pool_sizes))

        # pool_sizes: [6, 3, 2, 1]
        # 목표 출력 사이즈 : 6x6
        self.avpool_1 = nn.AdaptiveAvgPool2d(output_size=pool_sizes[0])
        self.cbr_1 = conv2DBatchNormRelu(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)

        # 목표 출력 사이즈 : 3x3
        self.avpool_2 = nn.AdaptiveAvgPool2d(output_size=pool_sizes[1])
        self.cbr_2 = conv2DBatchNormRelu(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)

        # 목표 출력 사이즈 : 2x2
        self.avpool_3 = nn.AdaptiveAvgPool2d(output_size=pool_sizes[2])
        self.cbr_3 = conv2DBatchNormRelu(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)

        # 목표 출력 사이즈 : 1x1
        self.avpool_4 = nn.AdaptiveAvgPool2d(output_size=pool_sizes[3])
        self.cbr_4 = conv2DBatchNormRelu(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)

    def forward(self, x):
        # AdaptiveAvgPool2d -> conv2DBatchNormRelu
        out1 = self.cbr_1(self.avpool_1(x))
        # upsampling
        # align_corners=True
        # -> 끝점을 타겟에 고정시키고 보간한다
        # align_corners=False
        # -> 배율을 미리 정하여 보간시킨 후 끝점을 재정렬한다
        # -> ss에서 성능이 좋지 않은 것으로 알려져 있음
        out1 = F.interpolate(out1, size=(
            self.height, self.width), mode="bilinear", align_corners=True)

        out2 = self.cbr_2(self.avpool_2(x))
        out2 = F.interpolate(out2, size=(
            self.height, self.width), mode="bilinear", align_corners=True)

        out3 = self.cbr_3(self.avpool_3(x))
        out3 = F.interpolate(out3, size=(
            self.height, self.width), mode="bilinear", align_corners=True)

        out4 = self.cbr_4(self.avpool_4(x))
        out4 = F.interpolate(out4, size=(
            self.height, self.width), mode="bilinear", align_corners=True)

        # x : [batch_num, 2048, h, w]
        # out : [batch_num, 512, h, w]
        # 채널 차원(1)로 결합
        output = torch.cat([x, out1, out2, out3, out4], dim=1)
        # [batch_num, 4096, h, w]
        return output


# conv2DBatchNormRelu -> conv -> upsampling
class DecodePSPFeature(nn.Module):
    def __init__(self, height, width, n_classes):
        super(DecodePSPFeature, self).__init__()

        # forward에 사용하는 화상 크기
        self.height = height
        self.width = width

        self.cbr = conv2DBatchNormRelu(
            in_channels=4096, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
        self.dropout = nn.Dropout2d(p=0.1)
        # 출력시 fc를 사용하지 않고 채널 수가 클래스 수와 같은 1x1 conv 사용
        self.classification = nn.Conv2d(
            in_channels=512, out_channels=n_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.cbr(x)
        x = self.dropout(x)
        x = self.classification(x)
        output = F.interpolate(
            x, size=(self.height, self.width), mode="bilinear", align_corners=True)

        return output


# conv2DBatchNormRelu -> conv -> upsampling
class AuxiliaryPSPlayers(nn.Module):
    def __init__(self, in_channels, height, width, n_classes):
        super(AuxiliaryPSPlayers, self).__init__()

        # forward에 사용하는 화상 크기
        self.height = height
        self.width = width

        self.cbr = conv2DBatchNormRelu(
            in_channels=in_channels, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
        self.dropout = nn.Dropout2d(p=0.1)
        # 출력시 fc를 사용하지 않고 채널 수가 클래스 수와 같은 1x1 conv 사용
        self.classification = nn.Conv2d(
            in_channels=256, out_channels=n_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.cbr(x)
        x = self.dropout(x)
        x = self.classification(x)
        output = F.interpolate(
            x, size=(self.height, self.width), mode="bilinear", align_corners=True)

        return output


class PSPNet(nn.Module):
    def __init__(self, n_classes):
        super(PSPNet, self).__init__()

        # resnet50
        block_config = [3, 4, 6, 3]  
        img_size = 475
        # img_size의 1/8
        img_size_8 = 60  

        # 4개의 모듈을 구성하는 서브 네트워크 준비
        self.feature_conv = FeatureMap_convolution()
        self.feature_res_1 = ResidualBlockPSP(
            n_blocks=block_config[0], in_channels=128, mid_channels=64, out_channels=256, stride=1, dilation=1)
        self.feature_res_2 = ResidualBlockPSP(
            n_blocks=block_config[1], in_channels=256, mid_channels=128, out_channels=512, stride=2, dilation=1)
        self.feature_dilated_res_1 = ResidualBlockPSP(
            n_blocks=block_config[2], in_channels=512, mid_channels=256, out_channels=1024, stride=1, dilation=2)
        self.feature_dilated_res_2 = ResidualBlockPSP(
            n_blocks=block_config[3], in_channels=1024, mid_channels=512, out_channels=2048, stride=1, dilation=4)

        self.pyramid_pooling = PyramidPooling(in_channels=2048, pool_sizes=[
            6, 3, 2, 1], height=img_size_8, width=img_size_8)

        self.decode_feature = DecodePSPFeature(
            height=img_size, width=img_size, n_classes=n_classes)

        self.aux = AuxiliaryPSPlayers(
            in_channels=1024, height=img_size, width=img_size, n_classes=n_classes)

    def forward(self, x):
        # 피처
        x = self.feature_conv(x)
        x = self.feature_res_1(x)
        x = self.feature_res_2(x)
        x = self.feature_dilated_res_1(x)

        # Feature 모듈의 중간에서 Aux 모듈로
        output_aux = self.aux(x)  

        x = self.feature_dilated_res_2(x)

        # 피라미드
        x = self.pyramid_pooling(x)
        output = self.decode_feature(x)

        return (output, output_aux)


# 손실함수 정의
class PSPLoss(nn.Module):
    def __init__(self, aux_weight=0.4):
        super(PSPLoss, self).__init__()
        # aux_loss의 가중치
        self.aux_weight = aux_weight  

    def forward(self, outputs, targets):
        # outputs : PSPNet의 출력, [num_batch, 21, 475, 475]
        # targets : gt, [num_batch, 475, 475]
        loss = F.cross_entropy(outputs[0], targets, reduction='mean')
        loss_aux = F.cross_entropy(outputs[1], targets, reduction='mean')

        return loss+self.aux_weight*loss_aux

