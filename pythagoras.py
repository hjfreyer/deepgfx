import tensorflow as tf
import math
import numpy as np

from draw import *


def circle(r):
    def render(x):
        return tf.abs(tf.norm(x[..., 0:2], axis=-1) - r)

    return render


def gety(x):
    return -x[..., 1]


def inverse(m):
    def render(x):
        return -m(x)

    return render


def intersection(masks):
    def render(x):
        ms = tf.stack([m(x) for m in masks])
        return tf.reduce_max(ms, axis=0)

    return render


def sector(start, end):
    return intersection(
        [
            transform(rotation_matrix(start), gety),
            transform(rotation_matrix(end), inverse(gety)),
        ]
    )


def frame(r, theta, a, b, side_movement, circle1_start, c1e):
    tx = r * tf.cos(theta)
    ty = r * tf.sin(theta)

    origin = tf.stack([0 * tx, 0 * ty], -1)
    corner = tf.stack([tx, 0 * ty], -1)
    far = tf.stack([tx, ty], -1)
    tri = triangle(origin, corner, far)

    adj_pointing1 = corner - origin
    adj_pointing2 = far - origin
    side_movement = tf.expand_dims(side_movement, -1)
    adj_pointing, _ = tf.linalg.normalize(
        side_movement * adj_pointing2 + (1 - side_movement) * adj_pointing1, axis=-1
    )

    opp_pointing1 = far - corner
    opp_pointing2 = far - origin

    opp_pointing, _ = tf.linalg.normalize(
        side_movement * opp_pointing2 + (1 - side_movement) * opp_pointing1, axis=-1
    )

    adj_len = tx * a
    opp_len = ty * b

    adj_end = origin + adj_pointing * tf.expand_dims(adj_len, -1)

    opp_start = corner * (1 - side_movement) + adj_end * side_movement

    adj = line_segment(origin, adj_end)
    opp = line_segment(
        opp_start,
        opp_start + opp_pointing * tf.expand_dims(opp_len, -1),
    )

    tri = clip(white, threshold(tri, 0.01))
    adj = clip(red, threshold(adj, 0.01))
    opp = clip(green, threshold(opp, 0.01))

    circ = intersection([circle(r), sector(circle1_start, c1e)])

    return layers([black, tri, adj, opp, clip(red, threshold(circ, 0.01))])


class Drawing(tf.Module):
    def __init__(self):
        super().__init__()
        self.r = tf.Variable(1.0, trainable=False, name='r')
        self.theta = tf.Variable(1.0, trainable=False, name='theta')
        self.a = tf.Variable(1.0, trainable=False, name='a')
        self.b = tf.Variable(1.0, trainable=False, name='b')
        self.movement = tf.Variable(0., trainable=False, name='b')
        self.c1s = tf.Variable(0., trainable=False  ,name='c1s')
        self.c1e = tf.Variable(1.0, trainable=False,name='c1e')

    @tf.function(
        input_signature=[
            tf.TensorSpec([None, None, None, 3], tf.float32),
        ]
    )
    def render(self, x):
        return frame(self.r, self.theta, self.a,  self.b,  self.movement,  self.c1s, self.c1e)(x)

model = Drawing()
tf.saved_model.save(model, "/tmp/tri.dgf")
