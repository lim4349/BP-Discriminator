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
목 초음파 이미지에서 신경부위 세그멘테이션을 하는 모델을 만드는 것과 모델의 성능(정확도)을 향상시키기 위해 여러가지 환경을 변화시키면서 이를 연구하는 것.

-- Train data set : 11270

-- Test data set: 5508

- Batch Size 조절
- Epoch 조절
- Validation split 계수 조절
- Dropout 계수 조절
- Suffle 
- Layer 조절
- Activation 함수 변경 




#### 데이터셋 이미지

[![image.png](https://i.postimg.cc/NGVPgPn4/image.png)](https://postimg.cc/s1YJmKsQ)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
[![image.png](https://i.postimg.cc/2yFqXtLD/image.png)](https://postimg.cc/bSrNZ3RC) 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;실제 환자 초음파 이미지&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;학습을 통해 얻은 마스킹 이미지




#### 실험 결과
[![no-earlystopping.png](https://i.postimg.cc/GhwCzQCX/no-earlystopping.png)](https://postimg.cc/JtqvzjtX)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Early Stopping을 설정하지 않고 실행한 성능 결과

&nbsp;  
  &nbsp;
  &nbsp;
  &nbsp;
  &nbsp;
  &nbsp;
  
[![earlystopping.png](https://i.postimg.cc/sXhLZkvs/earlystopping.png)](https://postimg.cc/xJ95wFv4)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Early Stopping을 설정하고 실행한 성능 결과






#### 참고 및 데이터셋 출처

https://github.com/GeorgeBatch/ultrasound-nerve-segmentation

https://www.kaggle.com/c/ultrasound-nerve-segmentation/overview
