import tensorflow as tf
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from atapy.data_accessor import DataAccessor

""" Common functions related to Tensorflow handling """

BATCH_SIZE = 64

def features_only(x, y):
    return x

def targets_only(x, y):
    return y

def dataset_creator(accessor: 'DataAccessor', single_instance_generator_function: Callable,
                    batch_size: int = BATCH_SIZE, prefetch: int = 2, randomize: bool = True,
                    **generator_args) -> tf.data.Dataset:
    generator_creator = single_instance_generator_function(accessor, randomize=randomize, **generator_args)
    x, y = next(generator_creator())
    x_shapes = {}
    x_types = {}
    for key, tensor in x.items():
        x_shapes.update({key: tensor.shape})
        x_types.update({key: tensor.dtype})
    y_shapes = {}
    y_types = {}
    for key, tensor in y.items():
        y_shapes.update({key: tensor.shape})
        y_types.update({key: tensor.dtype})
    dataset = tf.data.Dataset.from_generator(generator_creator,
                                             output_types=(x_types, y_types),
                                             output_shapes=(x_shapes, y_shapes))
    return dataset.batch(batch_size).prefetch(prefetch)
