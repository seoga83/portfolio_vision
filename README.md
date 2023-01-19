# - PORTFOLIO -

---
## portfolio 2. 길고양이 TNR(중성화사업) 위한 Object Detector
### 1. 프로젝트 개요
> 1.1 프로젝트 목적
> * CCTV 등의 영상 정보를 통해 길고양이 포획을 위한 생태 지역 확인
> * 실시간 촬영 영상 분석에 적합한 Object Detection 모델 분석

> 1.2 프로젝트 배경
> * 길고양이 개체수 증가로 인하여 중성화사업 추진
> * 길고양이 포획을 위한 위치 파악 필요
> * 인공지능 기술을 활용하여 작업의 효율화 가능

> 1.3 개발 환경
> * colab, ultralytics, OpenCV, Python

### 2. 데이터셋
> * 출처: 채용사이트 'wanted' Web API
> * 데이터수: 약 1,000여 채용공고
> * feature: 기술스택, 직군명, 자격요건, 주요업무, 우대사항, 회사명, 로고이미지, 채용공고 웹페이지, 경력사항

<video src="/etc/video/cat1_yolov5.mp4">
</video>

### 4. 분석 기법
> 4.1 TF-IDF (Term Frequency - Inverse Document Frequency)
> * 문서 집합에서 한 단어가 얼마나 중요한지를 수치적으로 나타낸 가중치
> * 한 문서에서 단어가 등장하는 빈도가 높을수록 커지고, 해당 단어를 포함하는 문서가 많을수록 반비례하여 작아진다.

&nbsp;&nbsp;&nbsp;&nbsp;<img src="/etc/img/tfidf.png" width="40%" height="40%"></img><br/>
> 4.2 KNN (K-Nearest Neighbor) 알고리즘
> * 거리 기반 분류분석 머신러닝 알고리즘
> * 새로운 데이터를 입력 받았을 때 이 데이터와 가장 근접한 데이터들의 종류가 무엇인지 확인 및 분류

&nbsp;&nbsp;&nbsp;&nbsp;<img src="/etc/img/knn.png" width="40%" height="40%"></img><br/>
> 4.3 코사인 유사도 (Cosine Similarity)
> * 두 벡터 간의 코사인 각도를 이용하여 구할 수 있는 두 벡터의 유사도
> * 문서 단어 행렬이나 TF-IDF 행렬을 통해서 문서의 유사도를 구하는 경우 각각의 특징 벡터를 이용하여 연산

&nbsp;&nbsp;&nbsp;&nbsp;<img src="/etc/img/cosine.png" width="70%" height="70%"></img><br/>
### 5. 웹 어플리케이션 구현
> * 웹페이지 접속 주소: https://cp1-datajob.herokuapp.com/

### 6. 프로젝트 회고
> * TF-IDF, 코사인 유사도는 데이터 빈도를 통해서 중요도를 판단하기 때문에 문맥을 이해하지 못하는 한계가 있으므로 딥러닝 모델을 통한 모델링 적용할 필요성
> * 데이터 직군 뿐만 아니라 범위를 넓혀서 모든 직군에 대한 데이터 처리 수행 필요
> * 사용자 로그 기록 등을 활용한 추천시스템 필요
---
## portfolio 3. 중고차 외관 손상 인식
### 1. 프로젝트 개요
> 1.1 프로젝트 주제
> * 중고차 외관 촬영 이미지를 통해 차량의 외관 손상 여부 파악
> * 딥러닝 Image Segmentation 모델링을 통해 외관 손상 부위 특정

> 1.2 프로젝트 배경
> * 중고차 사업자의 비즈니스 목적에 따라 차량 외관 손상에 대한 관리가 필요하며, 기존의 방식은 사람에 의한 검수로 인해 시간과 인력 비용이 많이 소모된다.
> * 이에 따라 딥러닝 기반 차량 외관 손상 인식을 통한 자동화를 바탕으로 시스템 개선이 필요한 상황이다.

> 1.3 기대효과
> * 중고차 사업자의 외관 손상 관리용 App에 인식 기능 탑재 가능
> * 공유 차량 관리 App에 탑재 가능
> * 차량 수리 업체에서 손상 부위 자동 인식 활용 가능

> 1.4 개발 환경
> * PixelAnnotationTool, colab, Pytorch, OpenCV, matplotlib

### 2. 프로젝트 구조
> 2.1 프로젝트 절차

&nbsp;&nbsp;&nbsp;&nbsp;<img src="/etc/img/pf2_flow.png" width="50%" height="50%"></img><br/>
> 2.2 프로젝트 수행 일정

&nbsp;&nbsp;&nbsp;&nbsp;<img src="/etc/img/pf2_schedule.png" width="90%" height="90%"></img><br/>
### 3. 데이터셋
> 3.1 PixelAnnotationTool

&nbsp;&nbsp;&nbsp;&nbsp;<img src="/etc/img/labeling.png" width="90%" height="90%"></img><br/>
> 3.2 이미지 데이터셋
> * 데이터수: 약 3,100 여장

&nbsp;&nbsp;&nbsp;&nbsp;<img src="/etc/img/pf2_dataset.png" width="90%" height="90%"></img><br/>
### 4. Image Segmentation
> 4.1 Deeplab V3+
> * Deeplab V3는 ImageNet에서 학습된 ResNet을 기본적인 특징 추출기로 사용한다. ResNet의 마지막 블록에서는 여러가지 확장비율을 사용한 Atrous Convolution을 사용해서 다양한 크기의 특징들을 뽑아낼 수 있도록 한다.
> * 이전 Deeplab 버전에서 소개되었던 Atrous Spatial Pyramid Pooling (ASPP)을 사용한다. 좋은 성능을 보였던 모델들의 특징들을 섞어놓은 모델이며, 다양한 확장비율을 가진 커널을 병렬적으로 사용한 convolution이다.
> * Deeplab V3+에서는 Encoder로 DeepLab V3를 사용하고, Decoder로 Bilinear Upsampling 대신 U-Net과 유사하게 Concat 해주는 방법을 사용한다.

&nbsp;&nbsp;&nbsp;&nbsp;<img src="/etc/img/pf2_deeplab.png" width="70%" height="70%"></img><br/>
> 4.2 모델링
> * Batch size: 2
> * Epoch: 30
> * Loss function: Cross Entropy Loss
> * Learning rate: 0.001
> * Optimizer: SGD

### 5. Image Segmentation 결과
> 5.1 성능 평가 지표

&nbsp;&nbsp;&nbsp;&nbsp;<img src="/etc/img/pf2_iou.png" width="40%" height="40%"></img><br/>
> 5.2 추론 결과 Mask

&nbsp;&nbsp;&nbsp;&nbsp;<img src="/etc/img/pf2_mask_result.png" width="80%" height="80%"></img><br/>
> 5.3 외관 손상 인식 이미지

&nbsp;&nbsp;&nbsp;&nbsp;<img src="/etc/img/pf2_result1.png" width="80%" height="80%"></img><br/>
&nbsp;&nbsp;&nbsp;&nbsp;<img src="/etc/img/pf2_result2.png" width="80%" height="80%"></img><br/>
&nbsp;&nbsp;&nbsp;&nbsp;<img src="/etc/img/pf2_result3.png" width="80%" height="80%"></img><br/>
### 6. 프로젝트 회고
> * 차량 외관 손상 유형 중 일부분인 스크래치에 대한 모델링만 진행한 점이 아쉽다.
> * 경험 부족과 시간 상의 제약으로 완성된 모델을 구현하지 못하였지만, 더 나은 개발 환경과 데이터셋으로 프로젝트를 진행해보고 싶은 욕심이 생긴다.
