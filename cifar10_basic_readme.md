[CIFAR10-basic 구동] 

# 텐서플로우 소스 예제 다운로드

(1) ubuntu git install
  >> sudo apt install git

(2) github저장소를 clone  
  >> git clone https://github.com/tensorflow/models

(3) cifar10 예제를 찾는다.
  >> (위치이동) cd models/tutorials/image/cifar10 
  >> (예제실행) python cifar10.py
  >> (예제실행) python cifar10_train.py

(4) 텐서보드 통해 확인
  >> tensorboard --logdir=/tmp/cifar10_train
  (putty상에선 안보임.)

(5) 학습 모델 평가
  >> python cifar10_eval.py 
