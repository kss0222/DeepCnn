def evaluate():
   with tf.Graph().as_default() as g:
     images, labels = cifar10.inputs(eval_data=eval_data)  # 검증용 CIFAR10 데이터셋 읽기

    logits = cifar10.inference(images)               # 예측 모델 그래프 정의
    top_k_op = tf.nn.in_top_k(logits, labels, 1)    # 예측

    # 체크포인트 파일에서 복구를 위한 moving average 그래프 및 saver 생성
    variable_averages = tf.train.ExponentialMovingAverage(
         cifar10.MOVING_AVERAGE_DECAY)
     variables_to_restore = variable_averages.variables_to_restore()
     saver = tf.train.Saver(variables_to_restore)

     while True:
       eval_once(saver, summary_writer, top_k_op, summary_op)  # saver, top_k_op 평가

def eval_once(saver, summary_writer, top_k_op, summary_op):
   with tf.Session() as sess:
     ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
     if ckpt and ckpt.model_checkpoint_path:  # 체크포인트 파일로 부터 학습 모델 복구
      saver.restore(sess, ckpt.model_checkpoint_path)

       total_sample_count = num_iter * FLAGS.batch_size
       step = 0
       while step < num_iter and not coord.should_stop():
         predictions = sess.run([top_k_op])       # 복구된 학습 모델 예측
        true_count += np.sum(predictions)
         step += 1

      # Compute precision @ 1.
      precision = true_count / total_sample_count
      print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))
      summary.value.add(tag='Precision @ 1', simple_value=precision)
      summary_writer.add_summary(summary, global_step)
    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)


def evaluate():
  """Eval CIFAR-10 for a number of steps."""
  with tf.Graph().as_default() as g:
    # Get images and labels for CIFAR-10.
    eval_data = FLAGS.eval_data == 'test'
    images, labels = cifar10.inputs(eval_data=eval_data)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = cifar10.inference(images)

    # Calculate predictions.
    top_k_op = tf.nn.in_top_k(logits, labels, 1)

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        cifar10.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

    while True:
      eval_once(saver, summary_writer, top_k_op, summary_op)
      if FLAGS.run_once:
        break
      time.sleep(FLAGS.eval_interval_secs)


def main(argv=None):  # pylint: disable=unused-argument
  cifar10.maybe_download_and_extract()
  if tf.gfile.Exists(FLAGS.eval_dir):
    tf.gfile.DeleteRecursively(FLAGS.eval_dir)
  tf.gfile.MakeDirs(FLAGS.eval_dir)
  evaluate()


if __name__ == '__main__':
  tf.app.run()
