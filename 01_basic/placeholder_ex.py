import tensorflow as tf

# placeholder 노드 정의
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
c = a + b

# session 을 만들고 placeholder 노드를 통해 값 입력받음
sess = tf.Session()

# c: 실행하고자 하는 연산, feed_dict: 플레이스 홀더 노드에 실제 대입되는 값
print(sess.run(c, feed_dict={a: 1.0, b: 3.0}))
print(sess.run(c, feed_dict={a: [1.0, 2.0], b: [3.0, 4.0]}))

d = 100 * c

print(sess.run(d, feed_dict={a: 1.0, b: 3.0}))
print(sess.run(d, feed_dict={a: [1.0, 2.0], b: [3.0, 4.0]}))

# session close
sess.close()