import tensorflow as tf
import numpy as np

# 'gohome' Data Creation
idx2char = ['g', 'o', 'h', 'm', 'e'] # g = 0, o = 1, h = 2, m = 3, e = 4
x_data = [[0, 1, 2, 1, 3]] # gohom
x_one_hot = [[[1, 0, 0, 0],  # g 0
              [0, 1, 0, 0],  # o 1
              [0, 0, 1, 0],  # h 2
              [0, 1, 0, 0],  # o 1
              [0, 0, 0, 1]]] # m 3
t_data = [[1, 2, 1, 3, 4]] # ohome

num_classes = 5 # 정답 크기, 즉 one-hot 으로 나타내는 크기
input_dim = 4 # one-hot size, 즉 입력값은 0부터 3까지 총 4가지임
hidden_size= 5 # output from the RNN. 5 to directly predict ont-hot
batch_size = 1 # one sentence
sequence_length = 5 # 입력으로 들어가는 문장 길이 gohom == 5
learning_rate = 0.1

X = tf.placeholder(tf.float32, [None, sequence_length, input_dim])
T = tf.placeholder(tf.int32, [None, sequence_length])

# BasicLSTMCell or GRUCell API 로 변경 가능
cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden_size) # BasicRNNCell(rnn_size)
initial_state = cell.zero_state(batch_size, tf.float32)
outputs, _states = tf.nn.dynamic_rnn(cell, X, initial_state=initial_state, dtype=tf.float32)

weights = tf.ones([batch_size, sequence_length])
seq_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=T, weights=weights)
loss = tf.reduce_mean(seq_loss)
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

y = prediction = tf.argmax(outputs, axis=2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        loss_val, _ = sess.run([loss, train], feed_dict={X: x_one_hot, T: t_data}) # 손실함수 계산 및 가중치 / 바이어스 업데이트
        result = sess.run(y, feed_dict={X: x_one_hot}) # 입력데이터에 대한 예측(prediction) 수행

        if step % 400 == 0:
            print("step: ", step, " loss: ", loss_val, " prediction: ", result, " target: ", t_data)

            # print char using dic
            result_str = [idx2char[c] for c in np.squeeze(result)] # one-hot 방식으로 계산된 result 를 해당 숫자에 대응하는 문자로 변환
            print("Prediction: ", ''.join(result_str))