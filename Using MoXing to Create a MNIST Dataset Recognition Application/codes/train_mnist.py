from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import moxing.tensorflow as mox
import os

tf.flags.DEFINE_string('data_url', None, 'Dir of dataset')
tf.flags.DEFINE_string('train_url', None, 'Train Url')

flags = tf.flags.FLAGS

work_directory = flags.data_url
filenames = ['train-images-idx3-ubyte.gz','train-labels-idx1-ubyte.gz','t10k-images-idx3-ubyte.gz',
             't10k-labels-idx1-ubyte.gz']

for filename in filenames:
  filepath = os.path.join(work_directory, filename)
  if not mox.file.exists(filepath):
    raise ValueError('MNIST dataset file %s not found in %s' % (filepath, work_directory))

def main(*args):
  mnist = input_data.read_data_sets(flags.data_url, one_hot=True)


  # define the input dataset, return image and label
  def input_fn(run_mode, **kwargs):
    def gen():
      while True:
        yield mnist.train.next_batch(50)
    ds = tf.data.Dataset.from_generator(
        gen, output_types=(tf.float32, tf.int64),
        output_shapes=(tf.TensorShape([None, 784]), tf.TensorShape([None, 10])))
    return ds.make_one_shot_iterator().get_next()


  # define the model for training or evaling.
  def model_fn(inputs, run_mode, **kwargs):
    x, y_ = inputs
    W = tf.get_variable(name='W', initializer=tf.zeros([784, 10]))
    b = tf.get_variable(name='b', initializer=tf.zeros([10]))
    y = tf.matmul(x, W) + b
    cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    predictions = tf.argmax(y, 1)
    correct_predictions = tf.equal(predictions, tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    export_spec = mox.ExportSpec(inputs_dict={'images': x}, outputs_dict={'predictions': predictions}, version='model')
    return mox.ModelSpec(loss=cross_entropy, log_info={'loss': cross_entropy, 'accuracy': accuracy},
                         export_spec=export_spec)


  mox.run(input_fn=input_fn,
          model_fn=model_fn,
          optimizer_fn=mox.get_optimizer_fn('sgd', learning_rate=0.01),
          run_mode=mox.ModeKeys.TRAIN,
          batch_size=50,
          auto_batch=False,
          log_dir=flags.train_url,
          max_number_of_steps=1000,
          log_every_n_steps=10,
          export_model=mox.ExportKeys.TF_SERVING)

if __name__ == '__main__':
  tf.app.run(main=main)
