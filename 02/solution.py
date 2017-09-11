# Solution is available in the other "solution.py" tab
import tensorflow as tf

# TODO: Convert the following to TensorFlow:
x = 10
y = 2
z = x/y - 1

# TODO: Print z from a session
x_node = tf.constant(10.0)
y_node = tf.constant(2.0)
one = tf.constant(1.0)
div_node = tf.div(x_node,y_node)
z_node = tf.subtract(div_node, tf.cast(1.0, tf.float32))

with tf.Session() as sess:
    # Run the tf.constant operation in the session
    output = sess.run(z_node)
    print(output)
