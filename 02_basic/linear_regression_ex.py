import tensorflow as tf
import numpy as np

load_data = np.loadtxt('./csv/data-01.csv', delimiter=',')

x_data = load_data[:, 0:-1]
t_data = load_data[:, [-1]]

print("x_data.shape: ", x_data.shape)
print("t_data.shape: ", t_data.shape)

W = tf.Variable(tf.random_normal([3, 1]))
b = tf.Variable(tf.random_normal([1]))

X = tf.placeholder(tf.float32, [None, 3]) # 현재 25X3 이지만 25X3이 아닌 None 을 지정하면 차후 50X3, 125X3 등으로 확장이 가능함
T = tf.placeholder(tf.float32, [None, 1]) # 현재 25X3 이지만 25X3이 아닌 None 을 지정하면 차후 50X3, 125X3 등으로 확장이 가능함
y = tf.matmul(X, W) + b # 현재 X, W, b 를 바탕으로 계산된 값

loss = tf.reduce_mean(tf.square(y - T)) # MSE 손실함수 정의
learning_rate = 1e-5 # 학습률
optimizer = tf.train.GradientDescentOptimizer(learning_rate) # 경사하강법 알고리즘 적용되는 optimizer
train = optimizer.minimize(loss) # optimizer 를 통한 손실함수 최소화

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer()) # 변수 노드(tf.Variable) 초기화

    for step in range(8001):
        # feed_dict 를 통해 입력되는 데이터를 이용하여 수행되는 연산은 loss, y, train 임
        loss_val, y_val, _ = sess.run([loss, y, train], feed_dict={X: x_data, T: t_data})

        if step % 400 == 0:
            print("step: ", step, " loss_Val: ", loss_val)

    print("Prediction is ", sess.run(y, feed_dict={X: [ [100, 98, 81] ]}))