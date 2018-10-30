
# 텐서플로우 소스 다운로드

(1) ubuntu에 git 설치  
  >> sudo apt install git
  
(2) github저장소를 clone  (tensorflow 소스와 예제를 다운로드)
  >> git clone https://github.com/tensorflow
  
  >> git clone https://github.com/tensorflow/models
  
(3) cifar10 예제를 찾는다.
  >>  cd models/tutorials/image/cifar10 
  
  >>  python cifar10.py
  
  >>  python cifar10_train.py
  
 -> Default 값: 100만 Step python 코드 안에서 학습 횟수를 줄이던지 
    모델과 tensorboard 실행만 보려면 Ctrl+C키로 중단
    
(4) 학습 모델 평가
  >> python cifar10_eval.py 
  -> 정확도는 86%정도 

(5) Tensorboard 확인
  >> tensorboard --logdir=/tmp/cifar10_train 학습된 파일이 저장된 장소
  
    https://localhost:6006으로 확인 (또는 자신이 이용하는 서버 주소와 포트)

