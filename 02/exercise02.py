import tensorflow as tf

# TODO: Convert the following to TensorFlow:
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

z = x/y 
z2 = z - tf.constant(1.0)

with tf.Session() as sess:
    output = sess.run(z2, {x:[10],y:[2]})

print(output)
