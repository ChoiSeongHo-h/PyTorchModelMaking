import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import models, transforms

# 전처리 클래스
class BaseTransform():
    # 사전 설정(transforms.Compose를 사용)
    def __init__(self, resize, mean, std):
        self.base_transform = transforms.Compose([
            transforms.Resize(resize),  # 짧은 변의 길이가 resize가 됨
            transforms.CenterCrop(resize),  # 이미지 중앙을 resize × resize로 자르기
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    # 전처리된 torch의 이미지를 출력
    def __call__(self, img):
        return self.base_transform(img)

# 라벨 처리 클래스
class ILSVRCPredictor():
    # 사전설정
    def __init__(self, class_index):
        self.class_index = class_index

    # 텐서를 받아 argmax인 인덱스에 해당하는 라벨을 출력
    def predict_max(self, out):
        maxid = np.argmax(out.detach().numpy())
        predicted_label_name = self.class_index[str(maxid)][1]

        return predicted_label_name

if __name__ == "__main__":
    use_pretrained = True
    net = models.vgg16(pretrained=use_pretrained)
    net.eval()  #추론 모드
    
    # 전처리 클래스 생성
    resize = 224
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    transform = BaseTransform(resize, mean, std)
    
    # 전처리, 배치 크기의 차원을 추가
    image_file_path = './data/goldenretriever-3724972_640.jpg'
    img = Image.open(image_file_path)  # [높이][너비][색RGB]
    img_transformed = transform(img)  # torch.Size([3, 224, 224])
    inputs = img_transformed.unsqueeze_(0)  # torch.Size([1, 3, 224, 224])

    # 모델에 입력
    out = net(inputs)  # torch.Size([1, 1000])
    
    # 라벨 처리 클래스 생성후, 출력을 라벨로 변환
    ILSVRC_class_index = json.load(open('./data/imagenet_class_index.json', 'r'))
    predictor = ILSVRCPredictor(ILSVRC_class_index)
    result = predictor.predict_max(out)
    
    print("입력 화상의 예측 결과: ", result)
