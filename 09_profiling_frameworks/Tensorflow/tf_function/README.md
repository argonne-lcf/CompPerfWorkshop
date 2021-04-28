#TF Function and Graph Compilation 

Using line profiler showed us that the largest computation use, by far, was the train loop and is subcalls.  Here, we'll wrap those functions in `@tf.function` decorators to improve performance with graph compilation.

We also can enable XLA Fusion (for GPU or CPU) to speed up the computations by fusing small ops together.  Beyond this, we'll have to run the TF Profiler.

That is in the next folder.