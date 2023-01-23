# 파이토치를 이용한 모델 만들기 실습    
## 목적 및 목표   
- line-by-line 주석 추가 및 정리   
  - tensor의 차원을 확실히 명시하여 tensor의 flow를 이해   
  - 주석을 통한 학습, 추후 빠른 재학습
- .ipynb 파일을 .py로 리팩토링
  - 추후 빠르게 재사용할 수 있는 코드 생성   
- 신경망 구축을 위한 실력 배양   
  - 일반화된 텐서 조작 방법 이해   
  - 신경망 구축을 위한 테크닉 습득   
- 파이토치 프래임워크의 이해   
- 논문 리뷰
## 참고 서적
- 만들면서 배우는 파이토치 딥러닝   
  - https://www.hanbit.co.kr/store/books/look.php?p_code=B7628794939
  - .ipynb 파일을 다운로드 할 수 있음
## 환경
- 1650ti    
- Windows 10    
- Anaconda    
- Python 3.9.13    
- CUDA 11.3.1    
- cuDNN 8.7.0    
- PyTorch 1.21.1    
## 리뷰 내역
### 1. pre-trained 신경망 사용
- VGG 이용   
  - https://github.com/ChoiSeongHo-h/PyTorchModelMaking/blob/main/01_using_pretrained_vgg/01_using_pretrained_vgg.py
### 2. 전이학습
- classifier 구조 변경   
  - https://github.com/ChoiSeongHo-h/PyTorchModelMaking/blob/main/02_transfer_learning/02_transfer_learning.py
### 3. 파인튜닝
- classifier 구조 변경 + feature extractor 파인튜닝   
  - https://github.com/ChoiSeongHo-h/PyTorchModelMaking/blob/main/03_fine_tuning/03_fine_tuning.py
### 4. object detection
- SSD 네트워크, loss, NMS, Hard Negative Mining, 앵커박스 생성 등    
  - https://github.com/ChoiSeongHo-h/PyTorchModelMaking/blob/main/04_object_detection/utils/ssd.py
- SSD training
  - https://github.com/ChoiSeongHo-h/PyTorchModelMaking/blob/main/04_object_detection/04_training_ssd.py
- SSD inference
  - https://github.com/ChoiSeongHo-h/PyTorchModelMaking/blob/main/04_object_detection/04_inference_ssd.py
