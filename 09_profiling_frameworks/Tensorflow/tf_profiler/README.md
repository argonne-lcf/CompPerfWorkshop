# TF Profiling Tool


Note the lines `tf.profiler.experimental.start('logdir')` and `tf.profiler.experimental.stop()` in the code.  This sets up and tears down the profiling tool built in to tensorflow.  Some screen shots below - the dominant ops are XLA compile (Well, we know that) and beyond that the main operation is conv2D backprop - a very compute heavy operation.  We may get some performance improvement further with reduced precision - see the `reduced_precision` folder!
