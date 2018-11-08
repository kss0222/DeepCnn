# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Builds the CIFAR-10 network.

Summary of available functions:

 # Compute input images and labels for training. If you would like to run
 # evaluations, use inputs() instead.
 inputs, labels = distorted_inputs()

 # Compute inference on the model inputs to make a prediction.
 predictions = inference(inputs)

 # Compute the total of the prediction with respect to the labels.
 loss = loss(predictions, labels)

 # Create a graph to run one step of training with respect to the loss.
 train_op = train(loss, global_step)
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import tarfile

from six.moves import urllib
import tensorflow as tf

import cifar10_input

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', '/tmp/cifar10_data',
                           """Path to the CIFAR-10 data directory.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")

# Global constants describing the CIFAR-10 data set.
# cifar10_input.py 에 정의되어 있던 값들
IMAGE_SIZE = cifar10_input.IMAGE_SIZE      # 24
NUM_CLASSES = cifar10_input.NUM_CLASSES    # 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN   # 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL     # 10000


# Constants describing the training process.
# tf.train.ExponentialMovingAverage에서 사용할 가중치 값. 
# 이동평균이란 특정기간동안 내에 측정된 값의 평균. 가장 최근 값에 더 큰 가중치를 두어 평균을 계산하는 방식
# learning rate 조정을 위한 것. 학습이 진행됨에 따라 기하급수적으로 감소시켜 나감.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # learning rate 감소 후의 epoch 수
LEARNING_RATE_DECAY_FACTOR = 0.1  # learning rate 감소를 위한 변수
INITIAL_LEARNING_RATE = 0.1       # 초기 learning rate

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
# 여러대의 GPU를 병렬 처리할 때 작업 이름을 구분하기 위한 구분자.
TOWER_NAME = 'tower'
# CIFAR-10의 데이터 경로
DATA_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'

# 각 layer들을 tensorboard에 시각화화기 위한 summary만드는 작업
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

# 파라미터로 전달받은 값을 cpu를 이용하여 처리할 변수 생성.
def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):  # 0번째 cpu 사용
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32         # tf.float32 사용
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype) # tf.get_variable 는 무조건 새로운 변수를 만든다.
  return var

# _variable_on_cpu(name, shape, initializer)함수를 이용하여 정규화 처리를 위한 변수 생성
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
  # 데이터 타입 설정
  dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
  var = _variable_on_cpu(
      name,
      shape,
  # 정규분포 기반의 초기화 함수. 표준편차의 양 끝값을 잘라낸 값으로 새로운 정규분포 만들어 초기화
      tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))  
  # L2 정규화 처리를 위한 코드. wd(weight decay)값이 있다면, 정규화 처리하고 그래프에 추가.
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss') # 전달받은 tensor의 요소들의 제곱의 합을 2로 나누어 리턴.
    tf.add_to_collection('losses', weight_decay)
  return var

# cifar10_input.py에 있는 같은 이름의 함수를 이용하여 학습할 데이터 불러오기
def distorted_inputs():
  """Construct distorted input for CIFAR training using the Reader ops.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.

  Raises:
    ValueError: If no data_dir
  """
  # 데이터 경로 지정되어 있지 않으면 에러 표시
  if not FLAGS.data_dir:
     raise ValueError('Please supply a data_dir')

  # 데이터 경로 조합
  data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
  # 최종적으로 사용할 이미지와 라벨을 해당 디렉토리에서 가져옴
  images, labels = cifar10_input.distorted_inputs(data_dir=data_dir,
                                                  batch_size=FLAGS.batch_size)
 
  if FLAGS.use_fp16:
    images = tf.cast(images, tf.float16)
    labels = tf.cast(labels, tf.float16)
  return images, labels

# cifar10_input.py에 있는 함수를 이용하여 평가할 데이터 불러오기
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
  data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
  images, labels = cifar10_input.inputs(eval_data=eval_data,
                                        data_dir=data_dir,
                                        batch_size=FLAGS.batch_size)
  if FLAGS.use_fp16:
    images = tf.cast(images, tf.float16)
    labels = tf.cast(labels, tf.float16)
  return images, labels

# 예측을 위한 모델을 구성하는 함수.
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
  #
  # conv1
  with tf.variable_scope('conv1') as scope:
   # 커널(필터) 초기화 : 5 x 5 크기의 3채널 필터를 만들며 64개의 커널을 사용해서 이미지에서 지정된 영역의 특징만을 걸러냄
   # x1*W1 + x2*W2 + ... x25*W25 (입력받은 이미지에서 필터와 겹치는 부분을 x라하고 해당위치의 필터를 W라 할 때)
   # http://taewan.kim/post/cnn/ 참조
   # 이 코드로 계산을 해보면 24 x 24 크기의 3채널 이미지를 입력으로 받아 5 x 5 크기의 3채널 필터 64개를 사용하고 padding이 
   # 원본 크기와 동일한 출력이 나오도록 SAME으로 지정되었으므로 24 x 24 크기에 64 채널을 가진 출력이 나올 것이다. 여기에
   # 배치 사이즈가 128이므로 최종 출력 tensor의 shape은 [128, 24, 24, 64]가 된다.
    kernel = _variable_with_weight_decay('weights',
                                         shape=[5, 5, 3, 64],
                                         stddev=5e-2,
                                         wd=None)
    conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
    # bias 초기화 채널의 갯수가 64개이므로 bias도 64개 설정. 
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    # weight+bias . bias는 conv의 마지막 차수와 일치하는 1차원
    pre_activation = tf.nn.bias_add(conv, biases)
    # 활성화 함수 Relu 추가 
    conv1 = tf.nn.relu(pre_activation, name=scope.name)
    _activation_summary(conv1) # tensorboard에서 확인하기 위한 작업

  # pool1
  # 이미지 축소하는 단계.필터로 주어진 영역 내에서 특정한 값 뽑아내는 작업. max pooling 주로 사용.
  pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], # 필터 크기가 3 x 3 , 가장 큰 값만을 추출 strides는 2
                         padding='SAME', name='pool1')
  # norm1
  # local regulation normalization. Relu 사용시 애러 개선에 
  norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
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
  with tf.variable_scope('softmax_linear') as scope:
    weights = _variable_with_weight_decay('weights', [192, NUM_CLASSES],
                                          stddev=1/192.0, wd=None)
    biases = _variable_on_cpu('biases', [NUM_CLASSES],
                              tf.constant_initializer(0.0))
    softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
    _activation_summary(softmax_linear)

  return softmax_linear

# loss값 계산. logits 파라미터는 inference() 함수의 리턴 값, labels는 distorted_input() 또는 input()의 리턴 값 중 labels부분
# cross entropy를 이용하여 loss 구함.
def loss(logits, labels):
  """Add L2Loss to all the trainable variables.

  Add summary for "Loss" and "Loss/avg".
  Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]

  Returns:
    Loss tensor of type float.
  """
  # Calculate the average cross entropy loss across the batch.
  # 여기서는 sparse_softmax_cross_entropy_with_logits 사용.softmax_cross_entropy_with_logits 함수가 확률분포를 따른다면
  #  sparse_softmax_cross_entropy_with_logits는 독점적인 확률로  label이 주어진다
  labels = tf.cast(labels, tf.int64)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  tf.add_to_collection('losses', cross_entropy_mean)

  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).
  return tf.add_n(tf.get_collection('losses'), name='total_loss')

# tensorboard에  loss 표시를 위해 sammary 추가.
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
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg') # 이동평균
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

  Create an and apply to all trainable variables. Add moving
  average for all trainable variables.

  Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
  Returns:
    train_op: op for training.
  """
  # Variables that affect learning rate.
  # 미리 정의한 변수들을 이요하여 learning rate를 조정할 파라미터 결정
  num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

  # Decay the learning rate exponentially based on the number of steps.
  # learning step이 증가할 수록 learning rate를 기하급수적으로 감소시킴
  # INITIAL_LEARNING_RATE * LEARNING_RATE_DECAY_FACTOR ^ (global_step / decay_steps)
  lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
  tf.summary.scalar('learning_rate', lr)

  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)
  # Optimizer 설정 및 tensorboard에 표시하기 위한 summary 생성 후 추가
  # Compute gradients.
  with tf.control_dependencies([loss_averages_op]):
    opt = tf.train.GradientDescentOptimizer(lr)
    grads = opt.compute_gradients(total_loss)

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
  
  
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
# tf.control_dependencies 함수는 오퍼레이션간의 의존성을 지정하는 함수로 with와 함께 사용하면 
# 파라미터로 전달된 오퍼레이션이 우선 수행된 뒤 다음 문장, 여기서는 with문 아래 있는 
# train_op = tf.no_op(name='train')이 수행된다.

with tf.control_dependencies([apply_gradient_op, variables_averages_op]): 
  train_op = tf.no_op(name='train')
  
  return variables_averages_op

# CIFAR-10 데이터 셋을 다운로드 받아 사용할 경로에 압축을 풀게 하는 합수.
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
