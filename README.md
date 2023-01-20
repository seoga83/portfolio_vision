# - PORTFOLIO -

---
## portfolio 1. 사용자 인터페이스에서 방향 컨트롤러로 활용 가능한 제스처 분류
### 1. 프로젝트 개요
> 1.1 프로젝트 주제
> * 별도의 콘트롤러 없이 사람의 손동작이나 눈동자 움직임으로 방향을 결정할 수 있는 제스처 분류 모델링
> * 적은 수의 데이터셋에서 나올 수 있는 성능지표 확인

> 1.2 프로젝트 배경
> * 게임, 키오스크 등의 전자기기의 작동을 위해 접근에 용이한 사용자 인터페이스에 대한 연구 활발
> * 인공지능 기술을 활용하여 각종 전자기기의 편리한 사용을 위한 사용자의 요구 증가 

> 1.3 개발 환경
> * colab, Python

### 2. 데이터셋

### 3. YOLOV5 - classify 모델

### 4. 모델링 학습 및 성능 평가

### 5. EfficientNet 모델

### 6. 프로젝트 회고

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

&nbsp;&nbsp;&nbsp;&nbsp;<img src="/etc/img/cat_dataset.png" width="90%" height="90%"></img><br/>

### 3. Object Detector 모델

&nbsp;&nbsp;&nbsp;&nbsp;<img src="/etc/img/detector_model.png" width="90%" height="90%"></img><br/>
> 3.1 2-Stage Detector
> * Regional Proposal과 Classification이 순차적으로 이루어진다.
> * 기존에는 이미지에서 object detection을 위해 sliding window 방식을 이용했었다. 이 방식은 이미지에서 모든 영역을 다양한 크기의 window로 탐색하는 것이다.
> * 비효율성을 개선하기 위해 ‘물체가 있을만한‘ 영역을 빠르게 찾아내는 알고리즘이다. Regional proposal은 object의 위치를 찾는 localization 문제이다.
> * 2-stage detector에서는 classification과 localization 문제를 순차적으로 해결한다.

> 3.2 1-Stage Detector
> * 2-stage detector와 반대로 regional proposal와 classification이 동시에 이루어진다.
> * Classification과 localization 문제를 동시에 해결하는 방법이다.
> * 1-stage detector는 비교적 빠르지만 정확도가 낮고, 2-stage detector는 비교적 느리지만 정확도가 높다.

### 4. 모델링 학습 및 성능 평가
> 4.1 Faster RCNN (2-stage Detector)
> * Batch size : 2
> * Epoch : 12
> * Optimizer : SGD
> * Loss : Cross Entropy

&nbsp;&nbsp;&nbsp;&nbsp;<img src="/etc/img/faster_rcnn_detect.png" width="40%" height="40%"></img>
&nbsp;&nbsp;&nbsp;&nbsp;<img src="/etc/img/faster_rcnn_eval.png" width="40%" height="40%"></img><br/>
> 4.2 YOLOV5 (1-stage Detector)
> * Batch size : 2
> * Epoch : 30
> * Optimizer : SGD
> * Loss : Cross Entropy

&nbsp;&nbsp;&nbsp;&nbsp;<img src="/etc/img/yolov5_detect.png" width="40%" height="40%"></img>
&nbsp;&nbsp;&nbsp;&nbsp;<img src="/etc/img/yolov5_eval.png" width="40%" height="40%"></img><br/>
### 5. Object Detector 영상 재생

&nbsp;&nbsp;&nbsp;[![Video Label](https://i.ytimg.com/an_webp/4yVs88qbXwI/mqdefault_6s.webp?du=3000&sqp=CNi0p54G&rs=AOn4CLAw1mwz6htlPSZUso68jHopgBAI2g)](https://www.youtube.com/embed/4yVs88qbXwI)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[![Video Label](https://i.ytimg.com/an_webp/9WCCl-WMZZM/mqdefault_6s.webp?du=3000&sqp=CIeUp54G&rs=AOn4CLA_AbRH69UjZA83moNrp8HrzymOZA)](https://www.youtube.com/embed/9WCCl-WMZZM)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="/etc/img/video_anal.png" width="85%" height="85%"></img><br/>
### 6. 프로젝트 회고
> * Detector 모델 각각의 특징을 확인해 볼 수 있어서 좋았다.
> * 저장된 동영상 파일이 아닌 실시간 촬영 영상에 대한 모델 적용을 해보지 못한 아쉬움이 있었다.

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

### 2. 프로젝트 절차

&nbsp;&nbsp;&nbsp;&nbsp;<img src="/etc/img/pf2_flow.png" width="50%" height="50%"></img><br/>
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
> * 경험 부족과 시간 상의 제약으로 완성된 모델을 구현하지 못한 부분에 대한 아쉬움이 남는다.
