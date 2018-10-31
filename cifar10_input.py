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

"""Routine for decoding the CIFAR-10 binary file format."""
# Python 2와 Python 3의 호환성을 위한 import

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
# six(2 * 3)는 Python 2와 Python 3에서 차이나는 함수들을 함께 사용할 수 있게 해줌

from six.moves import xrange  # xrange는 3에서는 range
import tensorflow as tf

# Process images of this size. Note that this differs from the original CIFAR
# image size of 32 x 32. If one alters this number, then the entire model
# architecture will change and any model would need to be retrained.
# 32 X 32 사이즈의 이미지를 랜덤하게 24 X 24 사이즈로 줄이면 전체 데이터셋의 크기가 커짐
IMAGE_SIZE = 24

# Global constants describing the CIFAR-10 data set.
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000 # training data
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000  # test data

# 파일 불러와  CIFAR-10의 바이너리 데이터 읽고 파싱하여 단일 오브젝트 형태로 반환
def read_cifar10(filename_queue):
  """Reads and parses examples from CIFAR10 data files.

  Recommendation: if you want N-way read parallelism, call this function
  N times.  This will give you N independent Readers reading different
  files & positions within those files, which will give better mixing of
  examples.

  Args:
    filename_queue: A queue of strings with the filenames to read from.

  Returns:
    An object representing a single example, with the following fields:
      height: number of rows in the result (32)
      width: number of columns in the result (32)
      depth: number of color channels in the result (3)
      key: a scalar string Tensor describing the filename & record number
        for this example.
      label: an int32 Tensor with the label in the range 0..9.
      uint8image: a [height, width, depth] uint8 Tensor with the image data
  """

  class CIFAR10Record(object):
    pass
  result = CIFAR10Record()

  # Dimensions of the images in the CIFAR-10 dataset.
  # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
  # input format.
  label_bytes = 1  # 2 for CIFAR-100  0~99
  result.height = 32 # 이미지 높이
  result.width = 32  # 이미지 넓이
  result.depth = 3   # 이미지 색상 채널
  image_bytes = result.height * result.width * result.depth # 이미지를 구성하는 총 바이트 수 32x32x3
  # Every record consists of a label followed by the image, with a
  # fixed number of bytes for each.
  record_bytes = label_bytes + image_bytes #모든 레코드는 label과 label에 해당하는 image로 구성

  # Read a record, getting filenames from the filename_queue.  No
  # header or footer in the CIFAR-10 format, so we leave header_bytes
  # and footer_bytes at their default of 0.
  reader = tf.FixedLengthRecordReader(record_bytes=record_bytes) # 파일로부터 고정길이의 레코드를 출력해주는 클래스. 
                                                                 # 첫번째 파라미터인 record_bytes는 읽어올 레코드 바이트 수
  # Queue 타입(FIFO)의 자료 구조를 파라미터로 받아 그 안의 레코드로부터 Key와 Value를 받아오는 처리. 
  # key는 레코드가 포함된 파일명과 index의 구성으로 되어있으며, value는 사용할 label과 image 포함.
  result.key, value = reader.read(filename_queue)

  # Convert from a string to a vector of uint8 that is record_bytes long.
  record_bytes = tf.decode_raw(value, tf.uint8) # 문자열을 숫자형 벡터로 변환

  # The first bytes represent the label, which we convert from uint8->int32.
  result.label = tf.cast(                                        # record_bytes에서 첫 번째 바이트를 가져와 int32타입으로 변환하여 리턴. 
                                                                 # result.label은 1바이트 크기의 int32 타입 요소를 갖는 벡터.
      tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)

  # The remaining bytes after the label represent the image, which we reshape
  # from [depth * height * width] to [depth, height, width].
  # record_bytes에서 첫 바이트인 라벨을 제외한 나머지  바이트(이미지 부분)를 가져와 [3, 32, 32] 형태의 shape으로 바꾼다. 

  depth_major = tf.reshape(
      tf.strided_slice(record_bytes, [label_bytes],
                       [label_bytes + image_bytes]),
      [result.depth, result.height, result.width])
  # Convert from [depth, height, width] to [height, width, depth].
  result.uint8image = tf.transpose(depth_major, [1, 2, 0]) #1 번째 요소인 32가 맨 앞으로, 다음으로 
                                                          # 2 번째 요소인 32가 오고 0번째 요소인 3은 맨 마지막으로 가도록..[32,32,3]

  return result

# 배치를 생성하는 코드.
# shuffle_batch는 무작위로 뒤섞은 배치 생성

def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
  """Construct a queued batch of images and labels.

  Args:
    image: 3-D Tensor of [height, width, 3] of type.float32.
    label: 1-D Tensor of type.int32
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
    batch_size: Number of images per batch.
    shuffle: boolean indicating whether to use a shuffling queue.

  Returns:
    images: Images. 4D tensor of [batch_size, height, width, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  # Create a queue that shuffles the examples, and then
  # read 'batch_size' images + labels from the example queue.
  num_preprocess_threads = 16  # 배치 생성 시 16개의 thread 사용
  if shuffle:
    images, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)
  else:
    images, label_batch = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size)

  # Display the training images in the visualizer. 텐서보드에서 이미지를 보여주는 코드
  tf.summary.image('images', images)

  return images, tf.reshape(label_batch, [batch_size]) # 배치 과정을 거친 image와 label 최종 shape는 각각 [128, 32, 32, 3]과 [128]

# 이미지 왜곡 함수

def distorted_inputs(data_dir, batch_size):
  """Construct distorted input for CIFAR training using the Reader ops.

  Args:
    data_dir: Path to the CIFAR-10 data directory.
    batch_size: Number of images per batch.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  #  CIFAR-10의 이미지 파일이 담긴 data_batch_1.bin ~ data_batch_5.bin의 5개 파일에 대한 전체 경로를 요소로 하는 벡터를 만듬.
  filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i) # os.path.join 함수는 전달받은 파라미터를 이어 새로운 경로를 만드는 함수
               for i in xrange(1, 6)] 
  for f in filenames:
    if not tf.gfile.Exists(f): # 만일 배열 내에 파일 경로가 없으면 에러 발생
      raise ValueError('Failed to find file: ' + f)

  # Create a queue that produces the filenames to read.
  # 파라미터 filenames를 Queue 형태로 리턴.
  filename_queue = tf.train.string_input_producer(filenames)
  with tf.name_scope('data_augmentation'):
    
    # Read examples from files in the filename queue.
    # 아래 설명할 read_cifar10 함수로부터 label, image 정보 등을 포함한  CIFAR10Record 클래스 타입을 반환.
    read_input = read_cifar10(filename_queue) 
    # cast 함수는 첫 번째 인자로 받은 텐서 타입의 파라미터를 두 번째 인자로 받은  데이터 타입의 요소를 가진 타입으로 형전환.
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    # Image processing for training the network. Note the many random
    # distortions applied to the image.

    # Randomly crop a [height, width] section of the image.
  
    distorted_image = tf.random_crop(reshaped_image, [height, width, 3]) #첫 번째 파라미터로 받은 텐서타입의 이미지들을 
                                   # 두 번째 파라미터로 받은 크기로 무작위로 잘라 첫 번째 받은 파라미터와 같은 형태로 반환. 

    # Randomly flip the image horizontally.
    # 좌우를 랜덤하게 뒤집은 형태의 이미지를 돌려준다.
    distorted_image = tf.image.random_flip_left_right(distorted_image)

    # Because these operations are not commutative, consider randomizing
    # the order their operation.
    # NOTE: since per_image_standardization zeros the mean and makes
    # the stddev unit, this likely has no effect see tensorflow#1458.
    
    # 밝기와 대비를 랜텀하게 변형시킨 이미지를 돌려준다
    distorted_image = tf.image.random_brightness(distorted_image,
                                                 max_delta=63)
    distorted_image = tf.image.random_contrast(distorted_image,
                                               lower=0.2, upper=1.8)

    # Subtract off the mean and divide by the variance of the pixels.
    # 이미지 표준화 작업. 
    float_image = tf.image.per_image_standardization(distorted_image)

    # Set the shapes of tensors.
    # 텐서의 shape 설정
    float_image.set_shape([height, width, 3])
    read_input.label.set_shape([1])

    # Ensure that the random shuffling has good mixing properties.
    # 전체 학습용 이미지의 40%, 즉, 총 50000개의 학습 이미지 중 20000개를 사용
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                             min_fraction_of_examples_in_queue)
    print ('Filling queue with %d CIFAR images before starting to train. '
           'This will take a few minutes.' % min_queue_examples)

  # Generate a batch of images and labels by building up a queue of examples.
  # 배치 작업에 사용할 128개의 이미지를 shuffle하여 리턴
  return _generate_image_and_label_batch(float_image, read_input.label,
                                         min_queue_examples, batch_size,
                                         shuffle=True)


def inputs(eval_data, data_dir, batch_size):
  """Construct input for CIFAR evaluation using the Reader ops.

  Args:
    eval_data: bool, indicating if one should use the train or eval data set.
    data_dir: Path to the CIFAR-10 data directory.
    batch_size: Number of images per batch.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  if not eval_data:
    filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
                 for i in xrange(1, 6)]
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
  else:
    filenames = [os.path.join(data_dir, 'test_batch.bin')]
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

  for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)

  with tf.name_scope('input'):
    # Create a queue that produces the filenames to read.
    filename_queue = tf.train.string_input_producer(filenames)

    # Read examples from files in the filename queue.
    read_input = read_cifar10(filename_queue)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    # Image processing for evaluation.
    # Crop the central [height, width] of the image.
    resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
                                                           height, width)

    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_standardization(resized_image)

    # Set the shapes of tensors.
    float_image.set_shape([height, width, 3])
    read_input.label.set_shape([1])

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(num_examples_per_epoch *
                             min_fraction_of_examples_in_queue)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(float_image, read_input.label,
                                         min_queue_examples, batch_size,
                                         shuffle=False)
