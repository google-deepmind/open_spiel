# C++ Tensorflow-based AlphaZero

This is a C++ implementation of the AlphaZero algorithm based on Tensorflow.

<span style="color:red"> Important note: despite our best efforts, we have been
unable to get the TF-based C++ AlphaZero to work externally.</span> For detailed
accounts of the current status, please see the discussion on the
[original PR](https://github.com/deepmind/open_spiel/issues/172#issuecomment-653582904)
and a
[recent attempt](https://github.com/deepmind/open_spiel/issues/539#issuecomment-805305939).
If you are interested in using C++ AlphaZero, we recommend you use the
[Libtorch-based C++ AlphaZero](https://github.com/deepmind/open_spiel/tree/master/open_spiel/algorithms/alpha_zero_torch)
instead, which is confirmed to work externally. As it mirrors the Tensorflow
version, the documentation below is still mostly applicable. As always, we
welcome contributions to fix the TF-based AlphaZero.

For more information on the algorithm, please take a look at the
[full documentation](https://github.com/deepmind/open_spiel/blob/master/docs/alpha_zero.md).

[TensorflowCC library](https://github.com/mrdaliri/tensorflow_cc/tree/open_spiel)
should be installed on your machine. Please see
[this fork of tensorflow_cc](https://github.com/mrdaliri/tensorflow_cc/tree/open_spiel)
for instructions on building and installing.

After having a working TensorflowCC API, you just need to set
`BUILD_WITH_TENSORFLOW_CC` flag to `ON` before building OpenSpiel.
