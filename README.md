충남대학교 졸업프로젝트 2019
==============================================

주제 : 목(neck) 초음파 이미지에서의 신경부위 세그멘테이션
-----------------------------------------------
#### 팀
- 11조(목아프조)

#### 팀원
- 임성근, 이금철, 이해원

#### 시작일자
- 2019.04.01 ~ 

#### 최종 마감일자
- 2019.12.16(월)

#### 지도교수님
- 장경선 교수님

#### 개발 환경
-- 하드웨어
- CPU : Intel i7-8700K(커피레이크)
- RAM : 32GB
-GPU : NVIDIA Geforce 1080 Ti

-- 소프트웨어
- language : python3.6
- development tool : pycharm, jupyter-notebook
- cuda : 10.0
- cudnn : 7
- API : keras
- OS : ubuntu 16.04


#### 개발 내용(간략히)
목 초음파 이미지에서 신경부위 세그멘테이션을 하는 모델을 만드는 것이다. 모델의 성능(정확도)을 향상시키기 위해 여러 시도를 하였다.
-- Train data set : 11270
-- Test data set: 5508
- Batch Size 조절
- Epoch 조절
- Validation split 계수 조절
- Dropout 계수 조절
- Suffle 
- Layer 조절
- Activation 함수 변경 



참고

https://github.com/GeorgeBatch/ultrasound-nerve-segmentation
