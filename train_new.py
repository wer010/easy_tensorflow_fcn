import tensorflow as tf
import commentjson
import sys
import logging
logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO, stream=sys.stdout)

import collections
from stwn.modules.tensorvision import train as train
from stwn.modules.tensorvision import utils as utils

flags = tf.app.flags

flags.DEFINE_string('name', None, 'Append a name Tag to run.')
flags.DEFINE_string('project', None, 'Append a name Tag to run.')
flags.DEFINE_string('hypes', None, 'File storing model parameters.')
flags.DEFINE_string('mod', None, 'Modifier for model parameters.')
flags.DEFINE_boolean('save', True, ('Whether to save the run. In case --nosave (default) '
                       'output will be saved to the folder TV_DIR_RUNS/debug '
                       'hence it will get overwritten by further runs.'))
FLAGS = flags.FLAGS


def dict_merge(dct, merge_dct):
    """ Recursive dict merge. Inspired by :meth:``dict.update()``, instead of
    updating only top-level keys, dict_merge recurses down into dicts nested
    to an arbitrary depth, updating keys. The ``merge_dct`` is merged into
    ``dct``.
    :param dct: dict onto which the merge is executed
    :param merge_dct: dct merged into dct
    :return: None
    """
    for k, v in merge_dct.iteritems():
        if (k in dct and isinstance(dct[k], dict) and
                isinstance(merge_dct[k], collections.Mapping)):
            dict_merge(dct[k], merge_dct[k])
        else:
            dct[k] = merge_dct[k]

def main(_):
    utils.set_gpus_to_use()
    # logging.getLogger().setLevel(logging.INFO)
    if FLAGS.hypes is None:
        logging.error("No hype file is given.")
        logging.info("Usage: python train.py --hypes hypes/KittiClass.json")
        exit(1)

    # 加载hypes
    with open(FLAGS.hypes, 'r') as f:
        logging.info("f: %s", f)
        hypes = commentjson.load(f)
    # 没什么卵用的一个flag，先留着
    if FLAGS.mod is not None:
        import ast
        mod_dict = ast.literal_eval(tf.app.flags.FLAGS.mod)
        dict_merge(hypes, mod_dict)

    # utils.load_plugins()

    utils.set_dirs(hypes, tf.app.flags.FLAGS.hypes)

    train.maybe_download_and_extract(hypes)
    logging.info("Initialize training folder")
    train.initialize_training_folder(hypes)
    logging.info("Start training")
    train.do_training(hypes)

if __name__ == '__main__':
    tf.app.run()
