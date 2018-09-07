import gym
import tensorflow as tf
import numpy as np
import random as rand
import copy

x = tf.placeholder(tf.float32, shape=[4])
x_ = tf.reshape(x, [1, 4])


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.5, mean=1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


max_grad = 1.0
mg = tf.constant(max_grad)

W_1 = weight_variable([4, 10])
# b_1 = bias_variable([10])
h_1 = tf.nn.relu(tf.matmul(x_, W_1))
W_2 = weight_variable([10, 10])
# b_2 = bias_variable([10])
h_2 = tf.nn.relu(tf.matmul(h_1, W_2))
# keep_prob = tf.placeholder(tf.float32)
# drop_1 = tf.nn.dropout(h_2, keep_prob)
W_3 = weight_variable([10, 2])
Q_out = tf.matmul(h_2, W_3)

predict = tf.argmin(Q_out, 1)

nextQ = tf.placeholder(tf.float32, [1, 2])


# lin = mg*(err-.5*mg)
# quad = .5*err*err
# loss = tf.where(err < mg, quad, lin)
absolute = tf.abs(nextQ - Q_out)
loss = tf.where(
    absolute < mg,
    tf.square(nextQ - Q_out),
    absolute)
train_step = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

# targetNetwork
t_W_1 = weight_variable([4, 10])
# b_1 = bias_variable([10])
t_h_1 = tf.nn.relu(tf.matmul(x_, t_W_1))
t_W_2 = weight_variable([10, 10])
# b_2 = bias_variable([10])
t_h_2 = tf.nn.relu(tf.matmul(t_h_1, t_W_2))
# keep_prob = tf.placeholder(tf.float32)
# drop_1 = tf.nn.dropout(h_2, keep_prob)
t_W_3 = weight_variable([10, 2])
t_Q_out = tf.matmul(t_h_2, t_W_3)

t_predict = tf.argmin(t_Q_out, 1)
# Hubert loss function
""" 
t_err = tf.abs(nextQ - t_Q_out)
t_lin = mg*(t_err-.5*mg)
t_quad = .5*t_err*t_err
t_loss = tf.where(t_err < mg, t_quad, t_lin)
"""

# Square diff loss function
# t_abs = tf.abs(nextQ - t_Q_out)
# t_loss = tf.where(
#    t_abs < mg,
#    tf.square(nextQ - t_Q_out),
#    t_abs)
t_loss = tf.square(nextQ - t_Q_out)
t_train_step = tf.train.AdamOptimizer(learning_rate=0.01).minimize(t_loss)


init = tf.global_variables_initializer()

# learning parameters
y = 0.99
e = 0.05
num_episodes = 1000
experiences = 0
# List to contain total rewards
rList = []
batch = []
expList = [None]*1000
hundredmean = [0]*100
count = 0
mean = 0
env = gym.make('CartPole-v0')
sess = tf.Session()
sess.run(init)
for i_episode in range(num_episodes):
    observation = env.reset()
    rAll = 0
    targetQ = []
    target_n = 0
    for t in range(200):
        env.render()
        a, allQ = sess.run([predict, Q_out], feed_dict={x: observation})
        print(allQ)
        if np.random.rand(1) < e:
            a[0] = env.action_space.sample()
        observation1, r, done, _ = env.step(a[0])
        Q1 = sess.run(t_Q_out, feed_dict={x: observation1})
        minQ1 = np.min(Q1)
        targetQ = allQ
        intermediate = minQ1
        if t < 199:
            intermediate = intermediate + done*100*y
        r = intermediate
        rAll += 1
        targetQ[0, a[0]] = r
        train_step.run(session=sess, feed_dict={x: observation, nextQ: targetQ})
        if target_n >= 5 or done == 1:
            t_train_step.run(session=sess, feed_dict={x: observation, nextQ: targetQ})
            target_n = 0
        else:
            target_n += 1
        if experiences < 500:
            batch = None
        elif experiences < 1000:
            batch = rand.sample(expList[:experiences], 50)
        else:
            batch = rand.sample(expList, 50)
        if experiences >= 500:
            for experience in batch:
                allQ = sess.run(Q_out, feed_dict={x: experience[0]})
                newtargetQ = allQ
                newtargetQ[0, experience[2]] = experience[3]
                train_step.run(session=sess, feed_dict={x: experience[0], nextQ: newtargetQ})
        expList[experiences % 1000] = [observation, observation1, a[0], r]
        experiences += 1
        observation = observation1
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            e = 1./((i_episode/50) + 20)
            break
    hundredmean[count % 100] = rAll
    meanCount = 0
    for r in hundredmean:
        if r > 0:
            mean += r
            meanCount += 1
    mean /= meanCount
    count += 1
    # print("Mean after last ", meanCount, " trials: ", mean, " at episode: ", i_episode)
    if mean >= 195:
        print("Success at episode: ", i_episode)
    rList.append(rAll)
    mean = 0

for r in rList:
    mean += r
mean /= 1000

print("Average steps per episode in evaluation: ", mean)