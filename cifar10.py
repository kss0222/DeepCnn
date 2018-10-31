
"""Builds the CIFAR-10 network.

Summary of available functions:

 # Compute input images and labels for training. If you would like to run
 # evaluations, use inputs() instead.
 inputs, labels = distorted_inputs()

 # Compute inference on the model inputs to make a prediction.
 predictions = inference(inputs)

 # Compute the total loss of the prediction with respect to the labels.
 loss = loss(predictions, labels)

 # Create a graph to run one step of training with respect to the loss.
 train_op = train(loss, global_step)
"""
# pylint: disable=missing-docstring
# _future_ : python 2와 3의 호환성을 위해 사용
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os # OS의 자원을 사용가능하게 해주는 모듈
import re # 정규 표현식 사용하게 해주는 모듈
import sys # python interpreter에 의해 관리되는 변수나 함수에 접근
import tarfile # tar 압축을 핸들링
# six 모듈은 python2와 3 함수를 함께 사용하게 해줌.
from six.moves import urllib #urllib는 URL 관련 모듈, Dataset 원격으로 다운로드할 때 사용
import tensorflow as tf

import cifar10_input

FLAGS = tf.app.flags.FLAGS #상수 값 관리

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 128, #FLAGS.batch_size=128
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', '/tmp/cifar10_data',
                           """Path to the CIFAR-10 data directory.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")

# Global constants describing the CIFAR-10 data set.
IMAGE_SIZE = cifar10_input.IMAGE_SIZE      #24
NUM_CLASSES = cifar10_input.NUM_CLASSES    #10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN #50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL   #10000


# Constants describing the training process. 
# 이동 평균(Moving Average) : 특정 기간동안 측정된 값의 평균.
MOVING_AVERAGE_DECAY = 0.9999     # tf.train.ExponentialMovingAverage에서 사용할 가중치 값

NUM_EPOCHS_PER_DECAY = 350.0      # Learning rate 감소 후 epoch 수
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate 감소를 위한 변수
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower' # 멀티 GPU 사용시 작업 이름 구분자
# CIFAR_10의 데이터 경로
DATA_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'

# 각 layer들을 tensorboard에 시각화 하기 위해 summary만듬
def _activation_summary(x): 
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measures the sparsity of activations.

  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity',
                                       tf.nn.zero_fraction(x))

# 파라미터로 전달받은 값을 이용하여 CPU를 이용하여 처리할 변수 생성
def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  # 첫번째 CPU를 사용하겠다고 지정
  with tf.device('/cpu:0'):
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32 # python의 3항 연산 FLAGS.use_fp16이 true면 tf.float16 사용하고 false면 tf.float32사용
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype) # tf.get_Variable는 4개의 파라미터 가진 새로운 변수 생성
    return var

#  정규화 처리를 할 변수 생성.
def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
  dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
  var = _variable_on_cpu(
      name,
      shape,
      tf.truncated_normal_initializer(stddev=stddev, dtype=dtype)) #정규분포 기반의 초기화 함수를 리턴, 표준편차의 양 끝단을 잘라낸 값으로 새로운 정규분포.

  # L2 정규화 처리를 위한 코드. 
  # wd(Weight Decay)값이 None이 아닌 경우 정규화 처리를 하고 그래프에 추가. 
    if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')# tf.nn.l2_loss는 전달받은 텐서의 요소들의 제곱의 합을 2로 나누어 리턴.
    tf.add_to_collection('losses', weight_decay)
  return var

# cifar10_input.py에 있는 함수를 이용하여 학습할 데이터를 불러온다

def distorted_inputs():
  """Construct distorted input for CIFAR training using the Reader ops.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.

  Raises:
    ValueError: If no data_dir
  """
  # 데이터 경로가 지정되어있지 않으면 에러 표시
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
    
  # 데이터 경로를 조합하여 최종적으로 사용할 이미지와 라벨을 가져옴
  data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
  images, labels = cifar10_input.distorted_inputs(data_dir=data_dir,
                                                  batch_size=FLAGS.batch_size)
  if FLAGS.use_fp16:
    images = tf.cast(images, tf.float16)
    labels = tf.cast(labels, tf.float16)
  return images, labels


# eval_data라는 파라미터가 추가된 것 외에는 distorted_inputs 함수와 내용 동일
def inputs(eval_data):
  """Construct input for CIFAR evaluation using the Reader ops.

  Args:
    eval_data: bool, indicating if one should use the train or eval data set.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.

  Raises:
    ValueError: If no data_dir
  """
 
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin') # 데이터 경로를 조합하여 최종적으로 사용할 이미지와 라벨을 가져옴
  images, labels = cifar10_input.inputs(eval_data=eval_data,
                                        data_dir=data_dir,
                                        batch_size=FLAGS.batch_size)
  
# FLAGS.use_fp16 값이 true이면 이미지와 라벨 텐서의 요소들을 tf.float16 타입으로 형변환. 코드에는 False로 지정되어있으므로 무시.

  if FLAGS.use_fp16:
    images = tf.cast(images, tf.float16)
    labels = tf.cast(labels, tf.float16)
  return images, labels

# 이 소스의 가장 중요한 핵심 부분. 예측(추론)을 위한 모델을 구성하는 함수

def inference(images):
  """Build the CIFAR-10 model.

  Args:
    images: Images returned from distorted_inputs() or inputs().

  Returns:
    Logits.
  """
  # We instantiate all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().
  
  # conv1
  # 여기서 커널(필터)는 이미지에서 지정된 영역의 특징만을 ‘걸러내는’ 역할
  # xW + b의 함수를 사용해서 처리. 일단 bias는 무시, xW만 생각해본다면 입력받은 이미지에서 필터와 겹치는 부분을 x라 보고 해당 위치의 필터를 W라 
  # 보면 x1* W1 + x2 * W2 + … + xn * Wn이 되는 것이다. 3 X 3 필터의 경우, x1 * W1 + x2 * W2 + x3 * W3 + ... x9 * W9로 계산
  with tf.variable_scope('conv1') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[5, 5, 3, 64], # 커널(필터) 초기화 : 5 X 5 크기의 3채널 필터를 만들며 64개의 커널 사용.
                                         stddev=5e-2,
                                         wd=None)
    conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
    # 바이어스 초기화, 채널의 개수가 64개이므로 bias도 64개 설정. biases는 64개의 요소가 0.0으로 채워진 # vector
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    # 가중치 + 바이어스. biases는 conv의 마지막 차수와 일치하는 1차원 값
    pre_activation = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(pre_activation, name=scope.name) # 활성화 함수 relu 추가
    _activation_summary(conv1) # tensorboard에서 확인

  # pool1
  #pooling은 이미지를 축소하는 단계.필터로 주어진 영역 내에서 특정한 값(평균,최대,최소)을 뽑아낸다. 성능이 좋은 max pooling(최대값 뽑아냄)을 주로 사용.
  pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], # 필터 크기 3x3, stride는 2
                         padding='SAME', name='pool1')
  # norm1
  norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, #local response normalization라는 정규화 처리. ReLu 사용시 에러율 개선에 사용.
                    name='norm1')

  # conv2
  with tf.variable_scope('conv2') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[5, 5, 64, 64],
                                         stddev=5e-2,
                                         wd=None)
    conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(pre_activation, name=scope.name)
    _activation_summary(conv2)

  # norm2
  norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm2')
  # pool2
  pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1], padding='SAME', name='pool2')

  # local3
  # fully connected layer 
  with tf.variable_scope('local3') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    reshape = tf.reshape(pool2, [images.get_shape().as_list()[0], -1])
    dim = reshape.get_shape()[1].value
    weights = _variable_with_weight_decay('weights', shape=[dim, 384],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
    local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    _activation_summary(local3)

  # local4
  # fully connected layer 2
  with tf.variable_scope('local4') as scope:
    weights = _variable_with_weight_decay('weights', shape=[384, 192],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
    local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
    _activation_summary(local4)

  # linear layer(WX + b),
  # We don't apply softmax here because
  # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
  # and performs the softmax internally for efficiency.
  
  # softmax layer
  with tf.variable_scope('softmax_linear') as scope:
    weights = _variable_with_weight_decay('weights', [192, NUM_CLASSES],
                                          stddev=1/192.0, wd=None)
    biases = _variable_on_cpu('biases', [NUM_CLASSES],
                              tf.constant_initializer(0.0))
    softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
    _activation_summary(softmax_linear)

  return softmax_linear

# 손실 값 계산을 위한 함수 cross entropy를 이용하여 loss를 구한다.

def loss(logits, labels):
  """Add L2Loss to all the trainable variables.

  Add summary for "Loss" and "Loss/avg".
  Args:
    logits: Logits from inference().  # inference() 함수의 리턴 값
    labels: Labels from distorted_inputs or inputs(). 1-D tensor # distorted_input()함수의 리턴 값
            of shape [batch_size]

  Returns:
    Loss tensor of type float.
  """
 # Calculate the average cross entropy loss across the batch.
 # 여기서는 sparse_softmax_cross_entropy_with_logits 함수가 사용되고 있는데  softmax_cross_entropy_with_logits와의 차이라면
 # softmax_cross_entropy_with_logits # 함수가 확률분포를를 따른다면 sparse_softmax_cross_entropy_with_logits는 독점적인 확률로
 # label이 주어진다고 함. (?)

  labels = tf.cast(labels, tf.int64)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  tf.add_to_collection('losses', cross_entropy_mean)

  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).
  return tf.add_n(tf.get_collection('losses'), name='total_loss')

# tensorboard에 표시하기 위해 손실 값에 대한 summary 추가, 손실값들의 이동 평균을 구하여 리턴. 

def _add_loss_summaries(total_loss):
  """Add summaries for losses in CIFAR-10 model.

  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.

  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg') #  가장 최근 값에 가중치를 두는 tf.train.ExponentialMovingAverage을 사용.
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.summary.scalar(l.op.name + ' (raw)', l)
    tf.summary.scalar(l.op.name, loss_averages.average(l))

  return loss_averages_op

# 학습을 실행시키는 함수
def train(total_loss, global_step):
  """Train CIFAR-10 model.

  Create an optimizer and apply to all trainable variables. Add moving
  average for all trainable variables.

  Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
  Returns:
    train_op: op for training.
  """
  # Variables that affect learning rate. 
  num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

  # Decay the learning rate exponentially based on the number of steps.
  # 학습 step이 증가할 수록 러닝 rate를 기하급수적으로 감소시키도록 처리
  # train.exponential_decay -> INITIAL_LEARNING_RATE * LEARNING_RATE_DECAY_FACTOR ^ (global_step / decay_steps)
  lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE, 
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
  tf.summary.scalar('learning_rate', lr)

  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)

  # Compute gradients.
  with tf.control_dependencies([loss_averages_op]):
    opt = tf.train.GradientDescentOptimizer(lr) # 경사 하강 Optimizer 설정
    grads = opt.compute_gradients(total_loss)  # gradients 계산

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.summary.histogram(var.op.name, var)

  # Add histograms for gradients.
  for grad, var in grads:
    if grad is not None:
      tf.summary.histogram(var.op.name + '/gradients', grad)

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
  with tf.control_dependencies([apply_gradient_op]): # control_dependencies-> 오퍼레이션간의 의존성을 지정, with와 함께 
   #사용하면 파라미터로 전달된 오퍼레이션이 먼저 수행된 뒤 다음 문장, 여기서는 with문 아래 있는 # train_op = tf.no_op(name='train')이 실행. 
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

  return variables_averages_op

# 데이터 셋이 들어있는 웹사이트로부터 CIFAR-10 데이터 셋을 다운로드 받아 사용할 경로에 압축을 풀어 주는 함수

def maybe_download_and_extract():
  """Download and extract the tarball from Alex's website."""
  dest_directory = FLAGS.data_dir
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
          float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
  extracted_dir_path = os.path.join(dest_directory, 'cifar-10-batches-bin')
  if not os.path.exists(extracted_dir_path):
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)
