import tensorflow as tf
import math
import numpy as np


def boltzmann(a, x):
  return tf.reduce_sum(x * tf.exp(a * x), axis=0) / tf.reduce_sum(tf.exp(a * x), axis=0)

def scale_matrix(sx, sy):
  return tf.constant([[1/sx, 0, 0], [0, 1/sy, 0], [0, 0, 1]])

def translation_matrix(dx, dy):
  return tf.constant([[1, 0, -dx], [0, 1, -dy], [0, 0, 1]])

def rotation_matrix(theta):
  return tf.stack([[tf.cos(theta), tf.sin(theta), 0], [-tf.sin(theta), tf.cos(theta), 0], [0, 0, 1]])


def three_points_matrix(pts):
  return tf.transpose(tf.concat([pts, tf.ones_like(pts[:, 0:1])], 1), (1, 0))

def triangle_bump(x):
  return tf.maximum(tf.minimum(x + 1, 1-x), 0)

def cross_fade(n, x):
  return tf.stack([triangle_bump(x - i) for i in range(n)])

def line(norm, x):
  norm, _ = tf.linalg.normalize(norm, axis=-1)
  # norm /= tf.norm(norm, axis=-1)
  return tf.reduce_sum(norm * x, axis=-1)

def line_2p(a, b, x):
  tangent = b - a
  norm = tf.stack([-tangent[..., 1], tangent[..., 0]], axis=-1)
  return line(norm, (x[..., 0:2] - a))

def line_segment(a, b):
  def render(x):
    tangent = b - a
    l = line_2p(a, b, x)
    start_clip = -line(tangent, (x[..., 0:2] - a))
    end_clip = line(tangent, (x[..., 0:2] - b))
    return tf.reduce_max([tf.abs(l), start_clip, end_clip], 0)
  return render

def simple_triangle(x):
  a = x[0]
  b = x[1]
  c = -(x[0] + x[1] - 1) / math.sqrt(2)
  return tf.reduce_min(tf.stack([a, b, c]), axis=0)

def smooth_simple_triangle(alpha, x):
  a = -x[0]
  b = -x[1]
  c = (x[0] + x[1] - 1)
  return -boltzmann(alpha, tf.stack([a, b, c]))

def x_axis(x):
  return x[1] > 0

def rot_from_point(pt):
  r = tf.linalg.norm(pt)
  return tf.stack([[pt[0]/r, pt[1]/ r, 0], [-pt[1]/r, pt[0]/r, 0], [0, 0, 1]])

def triangle(a, b, c):
  def render(x):
    la = line_2p(a, b, x)
    lb = line_2p(b, c, x)
    lc = line_2p(c, a, x)
    return tf.abs(tf.reduce_min(tf.stack([la, lb, lc]), axis=0))
  return render

def transform(mat, d):
  def render(x):
    x = tf.expand_dims(x, -1)
    res = mat @ x
    return d(tf.squeeze(res, -1))
  return render


def pure_color(rgba):
  def render(x):
    return rgba[tf.newaxis, tf.newaxis, tf.newaxis, :]
  return render

red = pure_color(tf.constant([1., 0, 0, 1]))
green = pure_color(tf.constant([0., 1, 0, 1]))
blue = pure_color(tf.constant([0., 0, 1, 1]))
white = pure_color(tf.constant([1., 1, 1, 1]))
black = pure_color(tf.constant([0., 0, 0, 1]))
def clip(drawing, mask_drawing):
  def render(x):
    d = drawing(x)
    m = mask_drawing(x)

    # m = tf.stack([tf.ones(m.shape), tf.ones(m.shape), tf.ones(m.shape), ], axis=-1)
    return d * tf.expand_dims(tf.where(m, 1., 0.), -1)
  return render

def composite(bf, af):
  def render(x):
    a = af(x)
    b = bf(x)

    rgb_a = a[..., 0:3]
    rgb_b = b[..., 0:3]

    alpha_a = a[..., 3:4]
    alpha_b = b[..., 3:4]

    return tf.concat([rgb_a + rgb_b * (1 - alpha_a), alpha_a + alpha_b * (1-alpha_a)], axis=-1)
  return render

def layers(l):
  if len(l) == 1:
    return l[0]
  h, *t = l
  return composite(h, layers(t))

# X = transform(translation_matrix(0.1, 0.1), X)

def threshold(drawing, limit):
  def render(x):
    d = drawing(x)
    return d < limit
  return render

def peek(drawing):
  def render(x):
    d = drawing(x)
    print(d)
    return d
  return render
