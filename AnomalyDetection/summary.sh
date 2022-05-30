Summary

Overview
efficientnet-b1,b2를 이용하여 학습을 진행했습니다.
arcface loss, mixup 등 시도에서 별다른 성능 향상을 가져오지는 않았습니다.
다른 팀들과는 다르게 efficientnet-b6,b7에 대해서 성능이 좋지 않아서 사용하지 않았습니다.
따라서, 기존에 학습했던 모델들을 이용해서 앙상블 및 후처리에 대하여 고민을 했습니다.
첫번째로 클래스 불균형으로 인한 good의 과한 예측을 피하고자 하였습니다.
기본적으로 good인 이미지들은 어떠한 기하학적 변형을 해도 매우 확실하게 good이라고 예측할거라 가정했습니다.
너무 과하게 good으로 예측하는 경우를 조금이라도 완화시키고자 각 모델들 output에 softmax를 취한 후 앙상블했습니다..
비정상의 경우를 bad로 통일한 후 모델을 학습한 결과를 이용하여 후처리를 진행했습니다.
두번째로 위의 결과를 이용해도 헷갈려하는 클래스들(pill, zipper, toothbrush, transistor, capsule)에 대해 추가 학습을 진행했습니다.
한 개의 클래스에 대해서만 학습을 한 후 하드보팅 또는 단일 모델의 결과를 가지고 후처리를 진행했습니다.

Training
https://github.com/alswlsghd320/dacon-anomaly/blob/master/multi_train.sh에 들어가시면 자세한 학습 세팅을 아실 수 있습니다.

Inference & Post-processing
val loss가 가장 낮은 모델들 중 8개를 이용하여 TTA(rotate90 0, 90, 180, 270) 결과에 대해 softmax를 구한 후 npy형태로 저장했습니다.
해당 npy파일들을 전부 불러와 평균을 취한 후 argmax를 취해 초기 예측값들을 구했습니다.
그 후 비정상 클래스들을 (class)-bad로 수정하여 별개로 30개 클래스에 대해서 efficientnet-b4를 이용하여 5-fold 학습을 진행했습니다.
위 5-fold good-bad 모델 예측에서 1) bad로 예측하거나 2) good으로 예측했지만 softmax값이 0.999999보다 작은 인덱스들을 추출했습니다.
추출한 인덱스들 중 원래 예측값이 good으로 되어 있는 경우 해당 레이블이 아닌 2번째로 높았던 레이블로 예측하게 했습니다.
pill, zipper, toothbrush, transistor, capsule에 대해 각각 해당 레이블만 추출하여 추가로 학습을 진행했습니다.
toothbrush만 학습한 단일 모델을 이용하여 원래의 예측값이 아닌 해당 예측값들로 변경을 했습니다.
zipper는 원래의 예측값과 세 개의 zipper만 학습한 모델을 이용하여 하드보팅을 진행했습니다.
나머지 세 클래스의 경우는 어떤 방법을 사용해도 성능 하락이 있어서 적용하지 않았습니다.

Summary
각 단일 모델마다 TTA를 거친 후 softmax 계산된 결과를 저장 및 앙상블 예측을 진행했습니다.
good-bad 5-fold 앙상블 모델을 이용하여 덜 확신을 가지고 good이라고 예측한 레이블들을 변경했습니다.
그럼에도 불구하고 헷갈려하는 클래스들에 대하여 해당 클래스만 따로 학습을 진행했습니다.
그 결과 toothbrush는 단일 모델의 예측값으로 변경, zipper는 원래의 예측값과 세 개의 추가 모델들을 하드 보팅한 예측값으로 변경했습니다.

Score
effb1_384_img_size_aug_pillzipper 5fold ensemble => 0.8518
effb2_bestloss 3개 앙상블 => 0.8577
effb1_384_img_size 5fold ensemble => 0.8548
effb1_384_img_size 5fold ensemble + good-bad 후처리 적용 => 0.8729
effb2_bestloss 3개, effb1_384_img_size 5fold 앙상블 + good-bad 후처리 + toothbrush 후처리 => 0.8990
effb2_bestloss 3개, effb1_384_img_size 5fold 앙상블 + good-bad 후처리 + toothbrush,zipper 후처리 => 0.9016
effb1_384_img_size(0,4 fold), effb2_bestloss 4개, effb1_384_img_size_aug_pillzipper(0,4 fold) + good-bad 후처리 + toothbrush,zipper 후처리 => 0.9087
