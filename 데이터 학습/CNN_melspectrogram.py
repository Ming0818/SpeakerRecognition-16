import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import IPython.display
import tensorflow as tf

def shuffle(x, y):
    randomize = np.arange(len(x))
    np.random.shuffle(randomize)
    x = x[randomize]
    y = y[randomize]
    return x,y

def reshape(x_data, y_data):
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    x_data = np.reshape(x_data, (-1, 96, 431))
    y_data = np.reshape(y_data, (-1))
    y_data_one_hot = tf.one_hot(y_data, nb_classes)
    y_data_one_hot = y_data_one_hot.eval()
    return x_data, y_data_one_hot

data_path = './data/'
training_number = 85
item_num = 100
nb_classes = 12
test_num = 10

x_data = [None] * nb_classes
y_data = [None] * nb_classes
for sub_path in range(nb_classes):
    x_data[sub_path] = []
    y_data[sub_path] = []
    for i in range(item_num):
        filename = data_path + str(sub_path + 1) + '/' + "input" + str(i + 1) + '.wav'
        if os.path.exists(filename):
            raw, sr = librosa.load(filename, sr=None)
            x_data[sub_path].append(librosa.feature.melspectrogram(y=raw, sr=sr, n_mels=96, fmax=18000))
            y_data[sub_path].append(sub_path)
    print(np.shape(x_data[sub_path]))

#96 X 431 사이즈
tf.InteractiveSession().as_default()
tf.tables_initializer().run()

#train, test set 분할
x_data = np.array(x_data)
y_data = np.array(y_data)
x_train = []
y_train = []
x_test = []
y_test = []

for i in range(nb_classes):
    x_train.append(x_data[i][:training_number])
    y_train.append(y_data[i][:training_number])
    x_test.append(x_data[i][training_number:training_number+test_num])
    y_test.append(y_data[i][training_number:training_number+test_num])

#reshape
x_train, y_train = reshape(x_train, y_train)
x_test, y_test = reshape(x_test, y_test)

print(np.shape(x_train))
print(np.shape(y_train))
print(np.shape(x_test))
print(np.shape(y_test))

#shuffle
#x_train, y_train = shuffle(x_train, y_train)
#x_test, y_test = shuffle(x_test, y_test)

# CNN (3 Conv + MP) + 1 FCN + 1 Output
learning_rate = 0.001
keep_prob = tf.placeholder(tf.float32)

with tf.variable_scope("sound_classifications-7-3") as scope:
    tf.variable_scope(scope, reuse=True)

    # Input Audio data of shape 96 * 862 = 82752
    X = tf.placeholder(tf.float32, [None, 96, 431])
    X_reshaped = tf.reshape(X, [-1, 96, 431, 1])
    Y = tf.placeholder(tf.float32, [None, nb_classes])

    # Conv 1
    W1 = tf.Variable(tf.random_normal([10, 10, 1, 32], stddev=0.01))
    L1 = tf.nn.conv2d(X_reshaped, W1, strides=[1, 1, 1, 1], padding='SAME')
    L1 = tf.nn.relu(L1)
    L1 = tf.nn.max_pool(L1, ksize=[1, 2, 4, 1], strides=[1, 2, 4, 1], padding='SAME')
    L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

    # L1 ImgIn shape=(?, 96, 431, 1)
    #    Conv      ->(?, 96, 431, 32)
    #    Pool      ->(?, 48, 108, 32)

    # Conv 2
    W2 = tf.Variable(tf.random_normal([10, 10, 32, 64], stddev=0.01))
    L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
    L2 = tf.nn.relu(L2)
    L2 = tf.nn.max_pool(L2, ksize=[1, 2, 4, 1], strides=[1, 2, 4, 1], padding='SAME')
    L2 = tf.nn.dropout(L2, keep_prob=keep_prob)

    # L2 ImgIn shape=(?, 48, 108, 32)
    #    Conv      ->(?, 48, 108, 64)
    #    Pool      ->(?, 24, 27, 64)

    # Conv 3
    W3 = tf.Variable(tf.random_normal([5, 5, 64, 128], stddev=0.01))
    L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
    L3 = tf.nn.relu(L3)
    L3 = tf.nn.max_pool(L3, ksize=[1, 2, 4, 1], strides=[1, 2, 4, 1], padding='SAME')
    L3 = tf.nn.dropout(L3, keep_prob=keep_prob)
    L3_flat = tf.reshape(L3, [-1, 128 * 12 * 7])

    # L3 ImgIn shape=(?, 24, 27, 64)
    #    Conv      ->(?, 24, 27, 128)
    #    Pool      ->(?, 12, 7, 128)

    # FC
    W4 = tf.get_variable("W4", shape=[128 * 12 * 7, 256], initializer=tf.contrib.layers.xavier_initializer())
    b4 = tf.Variable(tf.random_normal([256]))
    L4 = tf.nn.relu(tf.matmul(L3_flat, W4) + b4)
    L4 = tf.nn.dropout(L4, keep_prob=keep_prob)

    # Output
    W5 = tf.get_variable("W5", shape=[256, nb_classes], initializer=tf.contrib.layers.xavier_initializer())
    b5 = tf.Variable(tf.random_normal([nb_classes]))
    logits = tf.matmul(L4, W5) + b5


#여기서 부터 보자
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Accuracy
is_correct = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# Launch graph
cost_history = []
num_data = len(x_train)
batch_size = test_num
print(num_data)



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(nb_classes):
        curr_cost = 0
        print(epoch, "of", nb_classes)
        for i in range(item_num):
            curr_cost, _ = sess.run([cost, optimizer], feed_dict={X: x_train[epoch][i], Y: y_train[epoch][i], keep_prob: 0.7})
            cost_history.append(curr_cost)
            print("cost:",  curr_cost)

    print("Accuracy: ", accuracy.eval(session=sess, feed_dict={X: x_test[:batch_size], Y: y_test[:batch_size], keep_prob: 1}))
    plt.plot(cost_history)
    plt.show()