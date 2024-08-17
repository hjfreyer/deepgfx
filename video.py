#!/usr/bin/env python3

import tensorflow as tf

import argparse
import cv2
import pathlib
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=pathlib.Path)
    parser.add_argument("output", type=pathlib.Path)
    args = parser.parse_args()

    model = tf.saved_model.load(args.model)

    print(model.render.concrete_functions)

    WIDTH = 800
    HEIGHT = 600
    FPS = 30
    SEC = 5
    FRAMES = FPS * SEC

    video_path = str(args.output)
    fourcc = cv2.VideoWriter_fourcc(*"vp09")
    video_writer = cv2.VideoWriter(video_path, fourcc, FPS, (WIDTH, HEIGHT))

    for sec in range(SEC):
        print("Second: ", sec)

        X = tf.stack(
            tf.meshgrid(
                tf.linspace(0.0, 1, WIDTH),
                tf.linspace(0.0, 0.75, HEIGHT),
                indexing="ij",
            ),
            axis=-1,
        )
        X = X[:, :, tf.newaxis, :]
        X = tf.concat([X, tf.ones_like(X[..., 0:1])], -1)

        T = tf.linspace(tf.constant(sec, tf.float32), (sec + 1) * (FPS - 1) / FPS, FPS)
        T = T[tf.newaxis, tf.newaxis, :]

        frames = tf.cast(255 * model.render(X, T), tf.uint8).numpy()

        for framen in range(frames.shape[2]):
            video_writer.write(np.transpose(frames[:, :, framen, 2::-1], (1, 0, 2)))

    video_writer.release()


if __name__ == "__main__":
    main()
