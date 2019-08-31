from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from datetime import datetime 
import os

import cv2
import numpy as np
import tensorflow as tf

from UGATIT import UGATIT


_FRAME_SIZE = (256, 256)
_DEFAULT_OUTPUT_DIR = "outputs"
_DEFAULT_CODEC = "mp4v"
_DEFAULT_FRAMES_PER_SECOND = 20.0
_DEFAULT_CAMERA_ID = 0

def main(args):
    h, w = _FRAME_SIZE
    x = tf.placeholder(tf.float32, [None, h, w, 3])
    model = UGATIT()
    y = model.generate(x) 

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        restorer = tf.train.Saver()
        restorer.restore(sess, args.checkpoint)

        cap = cv2.VideoCapture(_DEFAULT_CAMERA_ID)
        if not cap.isOpened():
            raise Exception("Unable to read camera feed")
        # by default use timestamp to name recorded videos 
        video_path = os.path.join(
            _DEFAULT_OUTPUT_DIR,
            '{}.mp4'.format(datetime.now().strftime("%Y-%m-%d-%H%M%S"))
        )
        fourcc = cv2.VideoWriter_fourcc(*_DEFAULT_CODEC)
        video_out = cv2.VideoWriter(
            filename=video_path,
            fourcc=fourcc,
            fps=_DEFAULT_FRAMES_PER_SECOND,
            frameSize=(w * 2, h) if args.sidebyside else _FRAME_SIZE
        )
        window_message = "Press [SPACE BAR] to stop"
        while cap.isOpened():
            ret, orig = cap.read()
            if ret:
                orig = cv2.resize(orig, _FRAME_SIZE).astype(np.float32)
                out = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB) / 127.5 - 1.
                out = sess.run(y, feed_dict={x: np.expand_dims(out, 0)})
                out = (np.squeeze(out) + 1.) / 2.
                out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
                frame = np.hstack((orig / 255., out)) if args.sidebyside else out
                cv2.imshow(window_message, frame)
                if args.save:
                    video_out.write((frame * 255).astype(np.uint8)) 
                if cv2.waitKey(1) & 0xFF == ord(" "):
                    break
            else:
                break
        video_out.release()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter) 
    parser.add_argument(
        "-c", "--checkpoint",
        type=str,
        default="checkpoints/UGATIT_100_epoch_generator_only",
        help="path to checkpoint"
    )
    parser.add_argument(
        "-sbs", "--sidebyside",
        action="store_true",
        help="show input and output video side-by-side"
    )
    parser.add_argument(
        "-s", "--save",
        action="store_true",
        help="save output video to default directory: outputs/"
    )
    args = parser.parse_args()
    main(args)
