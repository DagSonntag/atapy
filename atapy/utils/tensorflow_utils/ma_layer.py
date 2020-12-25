import tensorflow as tf
import numpy as np
import math


class MaLayer(tf.keras.layers.Layer):
    def __init__(self, output_memory_length: int, input_window_length: int = None, method: str = 'wma',
                 period: int = None, **kwargs):
        """

        :param output_memory_length: The number of outputs to use where the first output is the current ma_value, the
        second the ma_value for the last timestep and so on
        :param input_window_length: How many nodes that should be used by the input to fit the ma_value. Note that
        only some will have weights for certain methods
        :param method: The ma method to use. Supported methods are ['sma', 'wma', 'ema']
        :param period: The period to use to 'prefit' the weights mathematically.
        :param kwargs: Other arguments to the layer super-method
        """
        super().__init__(**kwargs)
        self.method = method
        self.period = period
        self.output_memory_length = output_memory_length
        self.input_window_length = input_window_length if input_window_length is not None else period
        if method in ('sma', 'wma', 'ema', 'zlema'):
            self.ma_unit = tf.keras.layers.Dense(1, activation='linear', name="{}_layer".format(method), use_bias=False)
        elif method in ('hma', ):
            inp_n = self.input_window_length  # The n variable with regards to number of inputs
            inp_m = math.floor(inp_n / 2)
            inp_s = math.floor(math.sqrt(inp_n))
            per_n = self.period  # The n variable with regards to weights
            per_m = math.floor(per_n / 2)
            per_s = math.floor(math.sqrt(per_n))
            self.first = MaLayer(output_memory_length=inp_s+output_memory_length, input_window_length=inp_m,
                                 method='wma', period=per_m, name="hma_first_wma")
            self.second = MaLayer(output_memory_length=inp_s+output_memory_length, input_window_length=inp_n,
                                  method='wma', period=per_n, name="hma_second_wma")
            self.combined = MaLayer(output_memory_length=output_memory_length,
                                    input_window_length=inp_s, method='wma', period=per_s,
                                    name="hma_combined_wma")
        else:
            raise NotImplementedError

    def build(self, input_shape):
        super().build(input_shape)
        if input_shape[1] < self.output_memory_length + self.input_window_length:
            raise ValueError("The input must be larger or equal to the used input window + nr_of_outputs")
        if self.method == 'sma':
            # Use the whole input window, but only put weights on some (allowing the other to be fitted later)
            self.ma_unit.build((input_shape[0], self.input_window_length))
            if self.period is not None:
                sub_weights = np.array([[1]] * self.period) / self.period
                zeros = np.zeros((self.input_window_length - self.period, 1))
                self.ma_unit.set_weights([np.concatenate([sub_weights, zeros])])
        elif self.method == 'wma':
            # Use the whole input window, but only put weights on some (allowing the other to be fitted later)
            self.ma_unit.build((input_shape[0], self.input_window_length))
            if self.period is not None:
                sub_weights = np.array([[i] for i in range(self.period, 0, -1)]) / sum(list(range(1, self.period + 1)))
                zeros = np.zeros((self.input_window_length - self.period, 1))
                self.ma_unit.set_weights([np.concatenate([sub_weights, zeros])])
        elif self.method == 'ema':
            # Use the whole length of the input window
            self.ma_unit.build((input_shape[0], self.input_window_length))
            if self.period is not None:
                alpha = 2 / (1 + self.period)
                sub_weights = np.array([[alpha * (1 - alpha) ** i] for i in range(self.input_window_length)])
                # For the last (first) entry, don't multiply with alpha
                sub_weights[self.input_window_length - 1, 0] = (1 - alpha) ** (self.input_window_length - 1)
                self.ma_unit.set_weights([sub_weights])
        elif self.method == 'zlema':
            # Use the whole length of the input window
            self.ma_unit.build((input_shape[0], self.input_window_length))
            if self.period is not None:
                alpha = 2 / (1 + self.period)
                p_shift = math.floor((self.period - 1) / 2)  # 1.5 -> 1
                weight_first = np.array([[(2 * alpha) * (1 - alpha) ** i] for i in range(self.input_window_length)])
                weight_second = np.array([[-alpha * (1 - alpha) ** (i - p_shift)] if i >= p_shift else [0] for i in
                                          range(self.input_window_length)])  # shifted math.floor((n-1)/2) steps
                # For the values that does not have a corresponding second negative weight, remove the positive weights
                weight_first[-p_shift:, 0] = 0
                # For the last (first) entry, don't multiply with alpha ( note that no corresponding value in
                # second weights exist)
                weight_first[-p_shift, 0] = (1 - alpha) ** (self.input_window_length - p_shift)
                self.ma_unit.set_weights([weight_first+weight_second])
        elif self.method == 'hma':
            inp_n = self.input_window_length
            inp_s = math.floor(math.sqrt(inp_n))
            self.first.build(input_shape)
            self.second.build(input_shape)
            self.combined.build((input_shape[0], inp_s+self.output_memory_length))

        print("weights {}".format(self.weights))

    def call(self, inputs):
        inp = inputs
        # Use a sliding window on the inputs to calculate the outputs
        if self.method in ('sma', 'wma', 'ema', 'zlema'):
            outputs = [self.ma_unit(inp[:, i:(self.input_window_length + i)]) for i in range(self.output_memory_length)]
            out = tf.keras.layers.concatenate(outputs)
        elif self.method in ('hma', ):
            first_out = self.first(inputs)
            second_out = self.second(inputs)
            out = self.combined(2*first_out-second_out)
        else:
            raise NotImplementedError
        return out
