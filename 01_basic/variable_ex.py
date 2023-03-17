import tensorflow as tf

# 값이 계속 업데이트되는 변수노드 정의
W1 = tf.Variable(tf.random_normal([1])) # np.random.rand(1) 비슷함
b1 = tf.Variable(tf.random_normal([1])) # np.random.rand(1) 비슷함

W2 = tf.Variable(tf.random_normal([1,2])) # np.random.rand(1,2) 비슷함
b2 = tf.Variable(tf.random_normal([1,2])) # np.random.rand(1,2) 비슷함

# 세션 생성
sess = tf.Session()

# 변수노드 값 초기화. 변수노드를 정의했다면 반드시 필요함
sess.run(tf.global_variables_initializer())

for step in range(3):
    W1 = W1 - step # W1 변수노드 업데이트
    b1 = b1 - step # b1 변수노드 업데이트

    W2 = W2 - step # W2 변수노드 업데이트
    b2 = b2 - step # b2 변수노드 업데이트

    print("step: ", step, " W1: ", sess.run(W1), " b1: ", sess.run(b1))
    print("step: ", step, " W2: ", sess.run(W2), " b2: ", sess.run(b2))

# session close
sess.close()