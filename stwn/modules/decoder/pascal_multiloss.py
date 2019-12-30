from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf


def decoder(hypes, logits):
    """Apply decoder to the logits.

    Args:
      logits: Logits tensor, float - [batch_size, NUM_CLASSES].

    Return:
      logits: the logits are already decoded.
    """
    decoded_logits = {}
    decoded_logits['logits'] = logits['fcn_logits']

    num_classes = hypes['arch']['num_classes']
    with tf.name_scope('decoder'):
        logits = tf.reshape(decoded_logits['logits'], (-1, num_classes))
        decoded_logits['softmax'] = tf.nn.softmax(logits)
    return decoded_logits


def loss(hypes, decoded_logits, labels):
    """Calculate the loss from the logits and the labels.

    Args:
      logits: Logits tensor, float - [batch_size, NUM_CLASSES].
      labels: Labels tensor, int32 - [batch_size].

    Returns:
      loss: Loss tensor of type float.
    """

    num_classes = hypes['arch']['num_classes']
    logits = decoded_logits['logits']
    with tf.name_scope('loss'):
        logits = tf.reshape(logits, (-1, num_classes))
        epsilon = tf.constant(value=hypes['solver']['epsilon'])
        labels = tf.cast(dtype=tf.float32)

        assert tf.shape(logits)== tf.shape(labels)
        # softmax = tf.nn.softmax(logits) + epsilon

        if hypes['loss'] == 'xentropy':
            # cross_entropy_mean = _compute_cross_entropy_mean(hypes, labels, softmax)
            cross_entropy_mean = tf.nn.softmax_cross_entropy_with_logits(logits,labels)

        elif hypes['loss'] == 'softF1':
            cross_entropy_mean = _compute_f1(hypes, labels, softmax, epsilon)

        elif hypes['loss'] == 'softIU':
            cross_entropy_mean = _compute_soft_ui(hypes, labels, softmax, epsilon)

        reg_loss_col = tf.GraphKeys.REGULARIZATION_LOSSES

        weight_loss = tf.add_n(tf.get_collection(reg_loss_col),
                               name='reg_loss')

        total_loss = cross_entropy_mean + weight_loss

        losses = {}
        losses['total_loss'] = total_loss
        losses['xentropy'] = cross_entropy_mean
        losses['weight_loss'] = weight_loss

    return losses


def _compute_cross_entropy_mean(hypes, labels, softmax):
    head = hypes['arch']['weight']
    # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels = labels, logits=softmax)
    cross_entropy = -tf.reduce_sum(tf.multiply(labels * tf.log(softmax), head), reduction_indices=[1])

    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    return cross_entropy_mean


def _compute_f1(hypes, labels, softmax, epsilon):
    labels = tf.cast(tf.reshape(labels, (-1, 2)), tf.float32)[:, 1]
    logits = softmax[:, 1]
    true_positive = tf.reduce_sum(labels*logits)
    false_positive = tf.reduce_sum((1-labels)*logits)

    recall = true_positive / tf.reduce_sum(labels)
    precision = true_positive / (true_positive + false_positive + epsilon)

    score = 2*recall * precision / (precision + recall)
    f1_score = 1 - 2*recall * precision / (precision + recall)

    return f1_score


def _compute_soft_ui(hypes, labels, softmax, epsilon):
    intersection = tf.reduce_sum(labels*softmax, reduction_indices=0)
    union = tf.reduce_sum(labels+softmax, reduction_indices=0) - intersection + epsilon
    mean_iou = 1-tf.reduce_mean(intersection/union, name='mean_iou')

    return mean_iou


def evaluation(labels, decoded_logits, num_classes):
    """Evaluate the quality of the logits at predicting the label.

    Args:
      logits: Logits tensor, float - [batch_size, NUM_CLASSES].
      labels: Labels tensor, int32 - [batch_size], with values in the
        range [0, NUM_CLASSES).

    Returns:
      A scalar int32 tensor with the number of examples (out of batch_size)
      that were predicted correctly.
    """
    # For a classifier model, we can use the in_top_k Op.
    # It returns a bool tensor with shape [batch_size] that is true for
    # the examples where the label's is was in the top k (here k=1)
    # of all logits for that example.

    conf = tf.math.confusion_matrix(labels,decoded_logits,num_classes)

    conf = np.array(tf.shape(labels)[-1])
    locs = tf.less(labels,255)

    eval_list = {}


    logits = tf.reshape(decoded_logits['logits'], (-1, 2))
    labels = tf.reshape(labels, (-1, 2))

    pred = tf.argmax(logits, dimension=1)

    negativ = tf.to_int32(tf.equal(pred, 0))
    tn = tf.reduce_sum(negativ*labels[:, 0])
    fn = tf.reduce_sum(negativ*labels[:, 1])

    positive = tf.to_int32(tf.equal(pred, 1))
    tp = tf.reduce_sum(positive*labels[:, 1])
    fp = tf.reduce_sum(positive*labels[:, 0])

    eval_list['Acc']= (tn+tp)/(tn + fn + tp + fp)
    eval_list['Precision'] = (tp) / (tp + fp)
    eval_list['Recall'] = (tp) / (fn + tp)

    # eval_list.append(('Precision', tp/(tp + fp)))
    # eval_list.append(('True BG', tn/(tn + fp)))
    # eval_list.append(('True Street [Recall]', tp/(tp + fn)))

    return eval_list
