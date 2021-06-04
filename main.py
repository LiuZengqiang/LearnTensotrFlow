# 优化的步长会影响拟合的结果
import tensorflow as tf
import numpy as np

steps = 1000
# y = w*x+b
# y = 2*x+1

# 先声明占位变量(真值输入、输出)
# tf.float32类型, 维度为[-1,1]
x = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 1])

# 声明变量(节点)w,b,且初始化为0.0
w = tf.Variable(tf.constant(0.0))
b = tf.Variable(tf.constant(0.0))

# 声明结果变量(节点), 计算值
y_ = w * x + b

# 定义cost节点
cost = tf.reduce_mean(tf.square(y_ - y))

# 优化器学习率为0.0000001(使用其他的值可能得到错误的结果，需要调整)
train = tf.train.GradientDescentOptimizer(0.0000001).minimize(cost)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    # init global variable
    sess.run(init)
    #
    for i in range(steps):
        # generate data
        xs = np.array([[i]])
        ys = np.array([[i * 2.0 + 0.0]])

        sess.run(train, feed_dict={x: xs, y: ys})
        print("cost:", sess.run(cost, feed_dict={x: xs, y: ys}))
        print("w:", sess.run(w))
        print("b:", sess.run(b))
