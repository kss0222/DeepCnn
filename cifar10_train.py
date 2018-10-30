FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', '/tmp/cifar10_train',
                        """Directory where to write event logs """
                        """and checkpoint.""")   # 텐서보드 용 로그, 체크포인트 저장 폴더 지정
tf.app.flags.DEFINE_integer('max_steps', 2000,
                             """Number of batches to run.""") # 1000000

 def train():
   with tf.Graph().as_default():
   with tf.device('/cpu:0'):
     images, labels = cifar10.distorted_inputs()  # 이미지 왜곡해 라벨과 함께 리턴

  logits = cifar10.inference( images)  # 예측 모델 그래프 생성
  loss = cifar10.loss(logits, labels)     # loss 계산 그래프 추가
  train_op = cifar10.train(loss, global_step)  # 훈련 모델 그래프 추가

  # 체크포인트 파일 저장, 로그 후킹을 위해,
  # InteractiveSession() 대신 MonitoredTrainingSession()을 사용
  with tf.train.MonitoredTraningSession(
     checkpoint_dir=FLAGS.train_dir,
     hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),  # 최대 훈련 횟수
               tf.train.NanTensorHook(loss), _LoggerHook()],           # 로그 후킹
    config=tf.ConfigProto(
             log_device_placement=FLAGS.log_device_placement)) as mon_sess:
     while not mon_sess.should_stop():   # stop() 이 아닐때 까지
      mon_sess.run(train_op)               # 모델 훈련
 
def main(argv=None):
   cifar10.maybe_download_and_extract()     # CIFAR10 데이터 셋 다운로드 
   train()                                               # 모델 훈련

2. cifar10.py


def maybe_download_and_extract():  # Alex 사이트 CIFAR-10 다운로드 및 압축 해제 함수

    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)  # 다운로드

    tarfile.open(filepath, 'r:gz').extractall(dest_directory)                        # 압축해제





def distorted_inputs():   # CIFAR 이미지 왜곡을 통한 데이터 수 확대

  images, labels = cifar10_input.distorted_inputs(data_dir=data_dir,

                                                  batch_size=FLAGS.batch_size)  # 이미지 왜곡





def inference(images):   # 예측 모델 그래프 생성

  # conv1 정의

  with tf.variable_scope('conv1') as scope:

    kernel = _variable_with_weight_decay('weights', shape=[5,5,3,64], stddev=5e-2, wd=0)

    conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')

    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))

    pre_activation = tf.nn.bias_add(conv, biases)

    conv1 = tf.nn.relu(pre_activation, name=scope.name)  # ReLU 활성함수 정의




  # pool1 정의. max pooling.

  pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],

                         padding='SAME', name='pool1')

  # norm1 정의. local_response_normalization() 

  norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')




  # conv2 정의

  with tf.variable_scope('conv2') as scope:

    kernel =_variable_with_weight_decay('weights', shape=[5,5,64,64], stddev=5e-2, wd=0)

    conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')

    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))

    pre_activation = tf.nn.bias_add(conv, biases)

    conv2 = tf.nn.relu(pre_activation, name=scope.name)  # ReLU 활성함수 정의




  # norm2 정의

  norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')




  # pool2 정의

  pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],

                         strides=[1, 2, 2, 1], padding='SAME', name='pool2')




  # local3 정의 

  with tf.variable_scope('local3') as scope:

    reshape = tf.reshape(pool2, [FLAGS.batch_size, -1])

    dim = reshape.get_shape()[1].value

    weights = _variable_with_weight_decay('weights', shape=[dim, 384],

                                          stddev=0.04, wd=0.004)

    biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))

    local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)




  # local4 정의

  with tf.variable_scope('local4') as scope:

    weights = _variable_with_weight_decay('weights', shape=[384, 192],

                                          stddev=0.04, wd=0.004)

    biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))

    local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)




  # WX + b 정의

  with tf.variable_scope('softmax_linear') as scope:

    weights = _variable_with_weight_decay('weights', [192, NUM_CLASSES],

                                          stddev=1/192.0, wd=0.0)

    biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.0))

    softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)

  return softmax_linear




def loss(logits, labels):

  # cross entropy loss 평균 계산

  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(

      labels=labels, logits=logits, name='cross_entropy_per_example')

  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')

  tf.add_to_collection('losses', cross_entropy_mean)

  return tf.add_n(tf.get_collection('losses'), name='total_loss')





tf.app.flags.DEFINE_integer('batch_size', 128)  # 배치 데이터 크기




def train(total_loss, global_step):


  num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size

  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)




  # Exp()함수에 따른 Learning rate 관련 decay 값 정의

  lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE, global_step, decay_steps,

                                  LEARNING_RATE_DECAY_FACTOR, staircase=True)




  # Generate moving averages of all losses and associated summaries.

  loss_averages_op = _add_loss_summaries(total_loss)




  # gradients 계산 연산자 정의

  with tf.control_dependencies([loss_averages_op]):

    opt = tf.train.GradientDescentOptimizer(lr)

    grads = opt.compute_gradients(total_loss)




  # gradients 연산자 정의

  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)




  # histograms 추가

  for var in tf.trainable_variables():

    tf.summary.histogram(var.op.name, var)




  # 모든 훈련 변수들에 대한 이동 평균 추적

  variable_averages = tf.train.ExponentialMovingAverage(

      MOVING_AVERAGE_DECAY, global_step)

  variables_averages_op = variable_averages.apply(tf.trainable_variables())




  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):

    train_op = tf.no_op(name='train')

  return train_op
