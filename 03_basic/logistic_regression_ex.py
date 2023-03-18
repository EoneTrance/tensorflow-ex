import tensorflow as tf
import numpy as np

load_data = np.loadtxt('./csv/diabetes.csv', delimiter=',')

x_data = load_data[:, 0:-1]
t_data = load_data[:, [-1]]

print("x_data.shape: ", x_data.shape)
print("t_data.shape: ", t_data.shape)


X = tf.placeholder(tf.float32, [None, 8])
T = tf.placeholder(tf.float32, [None, 1])

W = tf.Variable(tf.random_normal([8, 1]))
b = tf.Variable(tf.random_normal([1]))

z = tf.matmul(X, W) + b # 선형회귀 값 z
y = tf.sigmoid(z) # sigmoid 로 계산

loss = -tf.reduce_mean(T*tf.log(y) + (1-T)*tf.log(1-y)) # 손실함수는 Cross Entropy
learning_rate = 0.01 # 학습률
optimizer = tf.train.GradientDescentOptimizer(learning_rate) # 경사하강법 알고리즘 적용되는 optimizer
train = optimizer.minimize(loss) # optimizer 를 통한 손실함수 최소화

# sigmoid 값 y 형상(shape) dms (759X8) dot (8X1) = 759X1 임
# 즉 y > 0.5 라는 것은 759개의 모든 데이터에 대해 y > 0.5 비교하여 총 759개의 True 또는 False 리턴함
predicted = tf.cast(y > 0.5, dtype=tf.float32)

# tf.reduce_mean = 데이터의 평균계산
# tf.cast(tf.equal(predicted, T): predicted 와 T 같으면 True, 아니면 False 를 리턴하므로,
# tf.cast 를 이용하여 1 또는 0으로 변환해서 총 759개의 1 또는 0을 가짐
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, T), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer()) # 변수 노드(tf.Variable) 초기화

    for step in range(20001):
        # feed_dict 로 입력되는 데이터를 이용하여 수행되는 연산은 loss, train
        loss_val, _ = sess.run([loss, train], feed_dict={X: x_data, T: t_data})

        if step % 500 == 0:
            print("step: ", step, " loss_Val: ", loss_val)

    # Accuracy 확인
    # feed_dict 로 입력되는 데이터를 이용하여 수행되는 연산은 y, predicted, accuracy
    y_val, predicted_val, accuracy_val = sess.run([y, predicted, accuracy], feed_dict={X: x_data, T: t_data})

    print("y_val.shape: ", y_val.shape, " predicted_val.shape: ", predicted_val.shape)
    print("accuracy:  ", accuracy_val)