# Solution is available in the other "solution.py" tab
import tensorflow as tf

# TODO: Print cross entropy from session
#def run():
output = None
    
softmax_data = [0.7, 0.2, 0.1]
one_hot_data = [1.0, 0.0, 0.0]

softmax = tf.placeholder(tf.float32)
one_hot = tf.placeholder(tf.float32)
    
logarithm = tf.log(softmax)
product = tf.multiply(one_hot,logarithm)
suma = tf.reduce_sum(product)
cross_ent = tf.multiply(suma, tf.cast(-1.0, tf.float32))
    
with tf.Session() as sess:
    output = sess.run(cross_ent, feed_dict={softmax: softmax_data, one_hot: one_hot_data})
    print(output)
