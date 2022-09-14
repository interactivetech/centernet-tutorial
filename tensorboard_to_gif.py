import argparse
import os
import shutil

import imageio
import matplotlib.pyplot as _
import matplotlib as mpl
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def arguments():
    parser = argparse.ArgumentParser(description='Create a gif from tensorboard file images')
    parser.add_argument(
        '--filename',
        type=str,
        help='Path to tensorboard file'
    )
    parser.add_argument(
        '--tag',
        type=str,
        help='Name of the image in the tensorboard e.g. `test/image`.'
    )
    parser.add_argument(
        '--output',
        default="./out.gif",
        type=str,
        help='File to store the final result'
    )
    parser.add_argument(
        '--start',
        type=int,
        help='First image in the gif (corresponds to tensorboard step)'
    )
    parser.add_argument(
        '--stop',
        type=int,
        help='Last image in the gif (corresponds to tensorboard step)'
    )
    return parser.parse_args()


def remove(path):
    """ remove directory or file `path` """
    if os.path.isfile(path):
        os.remove(path)  # remove the file
    elif os.path.isdir(path):
        shutil.rmtree(path)  # remove dir and all contains
    else:
        raise ValueError("file {} is not a file or dir.".format(path))


def save_images_from_event(fn, tag, start=-1, stop=np.inf, output_dir="./"):
    assert os.path.isdir(output_dir)

    sess = tf.Session()
    image_str_placeholder = tf.placeholder(tf.string)
    im_tf = tf.image.decode_image(image_str_placeholder)

    names = []

    for e in tf.train.summary_iterator(fn):

        if e.step < start:
            continue

        if e.step > stop:
            break

        image_strings = [
            v.image.encoded_image_string
            for v in e.summary.value if v.tag == tag
        ]

        for img_str in image_strings:
            im = sess.run(im_tf, feed_dict={image_str_placeholder: img_str})
            output_fn = os.path.realpath(f'{output_dir}/im_{e.step:06d}.png')
            mpl.image.imsave(output_fn, im)
            names.append(output_fn)

    return names


def  image_list_to_gif(filenames, output_fn):
    kwargs = {'duration': .5}
    with imageio.get_writer(output_fn, mode='I', **kwargs) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)


if __name__ == "__main__":
    args = arguments()
    tmp_output_dir = "/tmp/image_to_gif/"

    if not os.path.isdir(tmp_output_dir):
        os.mkdir(tmp_output_dir)

    names = save_images_from_event(
        args.filename,
        args.tag,
        start=args.start,
        stop=args.stop,
        output_dir=tmp_output_dir
    )

    image_list_to_gif(names, args.output)
    remove(tmp_output_dir)
    print("GIF CREATED:", args.output)