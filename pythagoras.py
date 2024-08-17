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


def compose(f, g):
    def render(x):
        ff = f(x)
        return g(ff)

    return render


def cart_to_polar(x):
    return tf.stack(
        [
            tf.norm(x[..., 0:2], axis=-1),
            tf.atan2(x[..., 1], x[..., 0]),
            x[..., 2],
        ],
        axis=-1,
    )


def polar_to_cart(x):
    return tf.stack(
        [
            x[..., 0] * tf.cos(x[..., 1]),
            x[..., 0] * tf.sin(x[..., 1]),
            x[..., 2],
        ],
        axis=-1,
    )


def lerp(f, g, a):
    return f * (1.0 - a) + g * a


def lerp_drawing(f, g, a):
    def render(x):
        return f(x) * (1.0 - a) + g(x) * a

    return render


def identity(x):
    return x


def zebra(d, mod):
    def render(x):
        return (d(x) % mod) / mod

    return render


# def frame(r, theta, a, b, movement, c1s, c1e, cp):


class Shot0(tf.experimental.ExtensionType):
    r: tf.Tensor
    theta: tf.Tensor
    a: tf.Tensor
    b: tf.Tensor
    side_movement: tf.Tensor


def shot0(args):
    r = args[..., 0]
    theta = args[..., 1]
    a = args[..., 2]
    b = args[..., 3]
    side_movement = args[..., 4]

    origin = tf.zeros_like(tf.stack([r, theta], -1))

    tx = r * tf.cos(theta)
    ty = r * tf.sin(theta)

    corner = tf.stack([tx, tf.zeros_like(ty)], -1)
    far = tf.stack([tx, ty], -1)
    tri = triangle(origin, corner, far)

    side_movement = tf.expand_dims(side_movement, -1)

    adj_dir, _ = tf.linalg.normalize(lerp(corner, far, side_movement), axis=-1)
    opp_dir, _ = tf.linalg.normalize(lerp(far - corner, far, side_movement), axis=-1)

    adj_len = tx * a
    opp_len = ty * b

    adj_end = adj_dir * tf.expand_dims(adj_len, -1)

    opp_start = lerp(corner, adj_end, side_movement)

    adj = line_segment(origin, adj_end)
    opp = line_segment(
        opp_start,
        opp_start + opp_dir * tf.expand_dims(opp_len, -1),
    )

    # def g(x):
    #     return tf.reduce_min(x % 0.1, axis=-1)

    # def quadrant(x):
    #     return (0. <= x[..., 1]) & (x[..., 1] <= math.pi/2) & (0. <= x[..., 0])
    # grid = lambda x: 0.3 * clip(white, threshold(g, 0.002))(x)
    # grid = clip(grid, quadrant)
    tri = clip(white, in_range(tri, 0.0, 0.01))
    adj = clip(red, threshold(adj, 0.01))
    opp = clip(green, threshold(opp, 0.01))

    # circ = intersection([circle(r), sector(circle1_start, c1e)])
    # circ = line_segment(tf.stack([r, 0.], axis=-1), tf.stack([r, math.pi], axis=-1))

    # circ2 = transform(translation_matrix(a / 2, b / 2), circle(a/2+b/2))
    # def circ2(x):
    #     return tf.abs( x[...,0]*(a * tf.cos(x[..., 1]) + b * tf.sin(x[..., 1]) - x[...,0]))
    # circ2 = lambda x:tf.abs(line(tf.stack([a, b], -1), x[..., 0:2]))

    return layers(
        [
            black,
            #    mask(white, zebra(tri, 0.025))
            tri,
            adj,
            opp,
        ]
    )
    # return compose(cart_to_polar, compose(lerp(identity, polar_to_cart, cp), draw))


def ease_in_out_sine(x: tf.Tensor) -> tf.Tensor:
    return -(tf.cos(math.pi * x) - 1) / 2


def ease_in_cubic(x: tf.Tensor) -> tf.Tensor:
    return x**3


def ease_out_cubic(x: tf.Tensor) -> tf.Tensor:
    return 1 - (1 - x) ** 3


class Drawing(tf.Module):
    def __init__(self):
        super().__init__()
        # self.args = tf.Variable(Shot0(r=1., theta=1., a=1., b=1., side_movement=0.))
        self.r = tf.Variable(1.0, trainable=False, name="r")
        self.theta = tf.Variable(1.0, trainable=False, name="theta")
        self.a = tf.Variable(1.0, trainable=False, name="a")
        self.b = tf.Variable(1.0, trainable=False, name="b")
        self.movement = tf.Variable(0.0, trainable=False, name="b")

    @tf.function(
        input_signature=[
            tf.TensorSpec([None, None, None, 3], tf.float32),
        ]
    )
    def render(self, x):
        return shot0(tf.stack([self.r, self.theta, self.a, self.b, self.movement]))(x)


class Video(tf.Module):
    @tf.function(
        input_signature=[
            tf.TensorSpec([None, None, None, 3], tf.float32),
            tf.TensorSpec([None, None, None], tf.float32),
        ]
    )
    def render(self, x, t):
        keyframes = tf.constant(
            [
                [1, 0, 0, 0, 0],
                [1, math.pi / 4, 0, 0, 0],
                [1, math.pi / 4, 1, 1, 0],
                [1, math.pi / 4, 1, 1, 1],
                [1, math.pi / 8, 1, 1, 1],
                [1, 3 * math.pi / 8, 1, 1, 1],
            ],
            tf.float32,
        )
        args = tf.reduce_sum(
            tf.expand_dims(ease_in_out_sine(cross_fade(6, t)), -1) * keyframes, axis=-2
        )
        return transform(
            translation_matrix(-0.7 / 2, -0.7 / 2)
            @ scale_matrix(0.5, 0.5)
            @ rotation_matrix(math.pi)
            @ translation_matrix(0.5, 0.375),
            shot0(args),
        )(x)


model = Video()
tf.saved_model.save(model, "/tmp/tri.dgf")
