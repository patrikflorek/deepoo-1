import tensorflow as tf

def iou_metric(y_true, y_pred, threshold=0.5):
    y_pred = tf.cast(y_pred > threshold, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return tf.math.divide_no_nan(intersection, union)

def dice_metric(y_true, y_pred, threshold=0.5):
    y_pred = tf.cast(y_pred > threshold, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    summation = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    return tf.math.divide_no_nan(2. * intersection, summation)
