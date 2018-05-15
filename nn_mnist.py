import gzip
import _pickle as cPickle

import tensorflow as tf
import numpy as np


# Translate a list of labels into an array of 0's and one 1.
# i.e.: 4 -> [0,0,0,0,1,0,0,0,0,0]
def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f ,encoding='latin1')
f.close()


#conjunto de entrenamiento
train_x, train_y = train_set

train_y= one_hot(train_y,10)

#Conjunto de validación
valid_x, valid_y = valid_set
valid_y= one_hot(valid_y,10)

#Conjunto de test
test_x, test_y = test_set
test_y= one_hot(test_y,10)

valid_errores=[]
train_error=[]

x = tf.placeholder("float", [None, 784])  # samples
y_ = tf.placeholder("float", [None, 10])  # labels

"""""
W1 = tf.Variable(np.float32(np.random.rand(784, 20)) * 0.1)
b1 = tf.Variable(np.float32(np.random.rand(20)) * 0.1)

W11 = tf.Variable(np.float32(np.random.rand(20, 20)) * 0.1)
b11 = tf.Variable(np.float32(np.random.rand(20)) * 0.1)

W2 = tf.Variable(np.float32(np.random.rand(20, 10)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)

h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
h2 = tf.nn.sigmoid(tf.matmul(h, W11) + b11)
y = tf.nn.softmax(tf.matmul(h2, W2) + b2)

Dos capas de 20 neuronas
91.58% de exito
"""""
"""""
W1 = tf.Variable(np.float32(np.random.rand(784, 20)) * 0.1)
b1 = tf.Variable(np.float32(np.random.rand(20)) * 0.1)

W2 = tf.Variable(np.float32(np.random.rand(20, 10)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)

h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
y = tf.nn.softmax(tf.matmul(h, W2) + b2)

Una capa de 20 neura
92.66% de exito
"""""
"""""
W1 = tf.Variable(np.float32(np.random.rand(784, 5)) * 0.1)
b1 = tf.Variable(np.float32(np.random.rand(5)) * 0.1)

W2 = tf.Variable(np.float32(np.random.rand(5, 10)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)

h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
y = tf.nn.softmax(tf.matmul(h, W2) + b2)

84.31 % de acierto
"""

W1 = tf.Variable(np.float32(np.random.rand(784, 30)) * 0.1)
b1 = tf.Variable(np.float32(np.random.rand(30)) * 0.1)

W2 = tf.Variable(np.float32(np.random.rand(30, 10)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)

h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
y = tf.nn.softmax(tf.matmul(h, W2) + b2)
"""""
92.08999999999999% de acierto
"""


"""""
W1 = tf.Variable(np.float32(np.random.rand(784, 30)) * 0.1)
b1 = tf.Variable(np.float32(np.random.rand(30)) * 0.1)

W11 = tf.Variable(np.float32(np.random.rand(30, 30)) * 0.1)
b11 = tf.Variable(np.float32(np.random.rand(30)) * 0.1)



W2 = tf.Variable(np.float32(np.random.rand(30, 10)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)

h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
h2 = tf.nn.sigmoid(tf.matmul(h, W11) + b11)

y = tf.nn.softmax(tf.matmul(h2, W2) + b2)

91.56 % de exito
"""""

loss = tf.reduce_sum(tf.square(y_ - y))

train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)  # learning rate: 0.01

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

print ("----------------------")
print ("   Start training...  ")
print ("----------------------")

batch_size = 20
epoch=0;
error=[100,0];
while(True):
    for jj in range(len(train_x) // batch_size):
        batch_xs = train_x[jj * batch_size: jj * batch_size + batch_size]
        batch_ys = train_y[jj * batch_size: jj * batch_size + batch_size]
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})

    epoch=epoch+1;

    train_error.append(sess.run(loss, feed_dict={x: batch_xs, y_: batch_ys}) / batch_size)
    error[1] = sess.run(loss, feed_dict={x: valid_x, y_: valid_y}) / len(valid_y)
    valid_errores.append(error[1])
    print ("Epoch #:", epoch, "Error: entrenamiento", train_error[-1],"Error de validacion",valid_errores[-1])


    result = sess.run(y, feed_dict={x: batch_xs})
    for b, r in zip(batch_ys, result):
        print (b, "-->", r)
    print("----------------------------------------------------------------------------------")



    if (abs(error[0] - error[1]) < 0.001 and error[0]<0.5):
        break

    error[0] = error[1]




# ---------------- Visualizing some element of the MNIST dataset --------------

import matplotlib.cm as cm
import matplotlib.pyplot as plt

p1, =plt.plot(range(epoch),train_error)
p2, =plt.plot(range(epoch),valid_errores)
plt.legend([p1, p2],['Error de entrenamiento','Error de validacion'])

plt.show()


aciertos=0

result = sess.run(y, feed_dict={x: test_x})
for b, r in zip(test_y, result):
    cierto=True
    for yy , nn in zip(b,r):
        if(yy!=round(nn)):
            cierto=False
            break
    if(cierto==True):
        aciertos=aciertos+1

porcentaje=aciertos/len(test_y)
porcentaje=porcentaje*100
print(porcentaje)




# TODO: the neural net!!
