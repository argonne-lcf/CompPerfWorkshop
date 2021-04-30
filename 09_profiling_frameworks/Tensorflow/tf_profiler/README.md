# TF Profiling Tool


Note the lines `tf.profiler.experimental.start('logdir')` and `tf.profiler.experimental.stop()` in the code.  This sets up and tears down the profiling tool built in to tensorflow.  See the screenshots below - the main operation is conv2D backprop - a very compute heavy operation.  We may get some performance improvement further with reduced precision - see the `reduced_precision` folder.


# Running the tf profiler

When you've captured your profile data, tensorboard will dump it into the folder `logdir` (as above) and you will have to view it.  The simplest way, for this application, is to copy it to your own laptop if you have tensorflow installed.  If not, you can run tensorboard on Theta and use an ssh port forward to view it on your own laptop.

Whatever you do, you can open tensorboard like so:
`tensorboard --logdir [your/own/path/to/logdir/]`

Next, open your browser and navigate to `localhost:6006:` (or, whatever port you forwarded to) and you'll see a screen like the one below:

