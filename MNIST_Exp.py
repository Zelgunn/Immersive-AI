import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tqdm import tqdm
import functools

def define_scope(function):
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(function.__name__):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator

class mnist_model:
    def __init__(self,
        input_placeholder, output_placeholder, dropout_placeholder, 
        image_width = 28, image_height = 28,
        kernel_size = 5, conv1_features_count = 32, conv2_features_count = 64,
        fully_connected_units_count = 1024,
        learning_rate = 1e-4
        ):
        self.input_placeholder = input_placeholder
        self.output_placeholder = output_placeholder
        self.dropout_placeholder = dropout_placeholder

        self.output_size = int(output_placeholder.get_shape()[1])

        self.image_width = int(image_width)
        self.image_height = int(image_height)
        self.image_size = int(image_width * image_height)

        self.kernel_size = int(kernel_size)
        self.conv1_features_count = int(conv1_features_count)
        self.conv2_features_count = int(conv2_features_count)

        self.fully_connected_units_count = int(fully_connected_units_count)
    
        self.learning_rate = learning_rate

        self.inference
        self.loss
        self.training

        self.evaluation

    @define_scope
    def inference(self):
        input_image_placeholder = tf.reshape(self.input_placeholder, [-1, self.image_width, self.image_height, 1])

        # 1st Layer (Convolution + ReLU + Max pooling)
        with tf.name_scope('conv1'):
            hidden_conv_layer1 = tf.layers.conv2d(
                inputs = input_image_placeholder,
                filters = self.conv1_features_count,
                kernel_size = [self.kernel_size, self.kernel_size],
                padding = "same",
                activation = tf.nn.relu
                )
            hidden_max_pool_layer1 = tf.layers.max_pooling2d(
                inputs = hidden_conv_layer1,
                pool_size = [2, 2],
                strides = [2, 2]
                )

        # 2nd Layer (Convolution + ReLU + Max pooling)
        with tf.name_scope('conv2'):
            hidden_conv_layer2 = tf.layers.conv2d(
                inputs = hidden_max_pool_layer1,
                filters = self.conv2_features_count,
                kernel_size = [self.kernel_size, self.kernel_size],
                padding = "same",
                activation = tf.nn.relu)
            hidden_max_pool_layer2 = tf.layers.max_pooling2d(
                inputs = hidden_conv_layer2,
                pool_size = [2, 2],
                strides = [2, 2])

        # 3rd Layer (Fully connected)
        with tf.name_scope('fully_connected1'):
            convoluted_image_size = int(self.image_size/16)
            fc_size = int(convoluted_image_size * self.conv2_features_count)

            hidden_max_pool_layer2_flatten = tf.reshape(hidden_max_pool_layer2, [-1, fc_size])
            hidden_fc_layer1 = tf.layers.dense(
                inputs = hidden_max_pool_layer2_flatten,
                units = self.fully_connected_units_count,
                activation = tf.nn.relu)
  
        ## Dropout
        hidden_fc_layer1_drop = tf.nn.dropout(hidden_fc_layer1, self.dropout_placeholder)

        # 4th Layer (Fully connected)
        with tf.name_scope('fully_connected2'):
            logits = tf.layers.dense(
                inputs = hidden_fc_layer1_drop,
                units = self.output_size,
                activation = tf.identity)

        return logits

    @define_scope
    def loss(self):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = self.inference, labels = self.output_placeholder)
        return tf.reduce_mean(cross_entropy)

    @define_scope
    def training(self):
        tf.summary.scalar('loss', self.loss)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)

        # Create a variable to track the global step.
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(self.loss, global_step=global_step)
            #train_op = optimizer.minimize(self.loss)
        return train_op

    @define_scope
    def evaluation(self):
        correct_inference = tf.equal(tf.argmax(self.inference, 1), tf.argmax(self.output_placeholder, 1))
        return tf.reduce_mean(tf.cast(correct_inference, tf.float32))

def main():
    # MNIST loading
    mnist = input_data.read_data_sets('/tmp/data', one_hot=True)

    # Parameters
    DIGIT_COUNT = 10
    IMAGE_SIZE = 784

    TRAINING_BATCH_SIZE = 50
    # Reduire si sur CPU (risque de prendre ~1h sinon)
    TRAINING_ITERATION_COUNT = 20000
    TESTING_BATCH_SIZE = 1000
    TESTING_ITERATION_COUNT = int(mnist.test.num_examples / TESTING_BATCH_SIZE)

    with tf.Graph().as_default():
        # Placeholders
        input_placeholder = tf.placeholder(tf.float32, [None, IMAGE_SIZE], name="Input_flat_placeholder")
        output_placeholder = tf.placeholder(tf.float32, [None, DIGIT_COUNT], name="True_output_placeholder")
        dropout_placeholder = tf.placeholder(tf.float32, name="Dropout_placeholder")

        # Model
        model = mnist_model(input_placeholder, output_placeholder, dropout_placeholder)

        #loss = model.loss
        train_op = model.training
        loss = model.loss
        # evaluation : for testing
        evaluation = model.evaluation

        summary = tf.summary.merge_all()

        init = tf.global_variables_initializer()

        saver = tf.train.Saver()

        with tf.Session() as session:
            summary_writer = tf.summary.FileWriter('/tmp/custom/MNIST_Exp/logs/', session.graph)

            print("Initializing variables...")
            session.run(init)
            print("Variables initialized !")

            # Training
            for i in tqdm(range(TRAINING_ITERATION_COUNT)):
                batch_inputs, batch_outputs = mnist.train.next_batch(TRAINING_BATCH_SIZE)
                feed_dict = {
                    input_placeholder : batch_inputs,
                    output_placeholder : batch_outputs,
                    dropout_placeholder: 0.5
                }
                _, loss_value = session.run([train_op, loss], feed_dict=feed_dict)

                if i%100 == 0:
                    summary_str = session.run(summary, feed_dict=feed_dict)
                    summary_writer.add_summary(summary_str, i)
                    summary_writer.flush()

                if i%1000 == 0 or (i + 1) == TRAINING_ITERATION_COUNT:
                    checkpoint_file = '/tmp/custom/MNIST_Exp/logs/model.ckpt'
                    saver.save(session, checkpoint_file, global_step=i)

            # Testing
            test_accuracy = 0
            for i in tqdm(range(TESTING_ITERATION_COUNT)):
                test_input, test_outputs = mnist.test.next_batch(TESTING_BATCH_SIZE)
                test_accuracy += session.run(evaluation, feed_dict={input_placeholder : test_input, output_placeholder : test_outputs, dropout_placeholder: 1.0})
            test_accuracy /= TESTING_ITERATION_COUNT
            print("Test accuracy = %g"%test_accuracy)

if __name__ == '__main__':
    main()
