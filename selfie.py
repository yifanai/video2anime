from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import cv2
import numpy as np
import tensorflow as tf

from UGATIT import UGATIT


_IMAGE_SIZE = (256, 256)
_DEFAULT_OUTPUT_DIR = "outputs"

def main(args):
    if not os.path.exists(args.input): 
        raise FileNotFoundError("Input image does not exist")
    h, w = _IMAGE_SIZE 
    x = tf.placeholder(tf.float32, [None, h, w, 3])
    model = UGATIT()
    y = model.generate(x) 

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        restorer = tf.train.Saver()
        restorer.restore(sess, args.checkpoint)
        orig = cv2.imread(args.input)
        orig = cv2.resize(orig, _IMAGE_SIZE).astype(np.float32)
        out = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB) / 127.5 - 1.
        out = sess.run(y, feed_dict={x: np.expand_dims(out, 0)})
        out = (np.squeeze(out) + 1.) / 2.
        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        window_message = "Press any key to close" 
        if args.sidebyside:
            cv2.imshow(window_message, np.hstack((orig / 255., out)))
        else:
            cv2.imshow(window_message, out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        if args.save:
            base_filename = os.path.basename(args.input).split(".")[0] 
            out_path = os.path.join(_DEFAULT_OUTPUT_DIR, base_filename + "_out.png")
            save_out = (out * 255).astype(np.uint8)
            cv2.imwrite(out_path, save_out)
            print("output image saved to: {}".format(out_path))
            if args.sidebyside:  # also save side-by-side image 
                sbs_path = os.path.join(_DEFAULT_OUTPUT_DIR, base_filename + "_sbs.png")
                cv2.imwrite(sbs_path, np.hstack((orig.astype(np.uint8), save_out)))
                print("side-by-side image saved to: {}".format(sbs_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter) 
    parser.add_argument("-i", "--input", type=str, required=True, help="path to input image")
    parser.add_argument(
        "-c", "--checkpoint",
        type=str,
        default="checkpoints/UGATIT_100_epoch_generator_only",
        help="path to checkpoint"
    )
    parser.add_argument(
        "-sbs", "--sidebyside",
        action="store_true",
        help="show input and output image side-by-side"
    )
    parser.add_argument(
        "-s", "--save",
        action="store_true",
        help="save output image to default directory: outputs/"
    )
    args = parser.parse_args()
    main(args)
