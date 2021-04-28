#Line by Line profiling

# Installation
 TODO TODO TODO


The first, and most easily accessible, profiling tool, is the line profiling tool.

Run the profiling tool using `kernprof` instead of python.  This is only for single-node performance.  For example:
```bash
kernprof -l train_GAN.py
```

(or, `~/.local/bin/kernprof -l train_GAN.py`)

This will dump the output for 3 functions, the biggest compute users, into a file `train_GAN.py.lprof`.  Let's dump out the line by line calls:

```bash
python -m line_profiler train_GAN.py.lprof
```

First, we see that the main training function is 15.4 seconds, but 80% is the training loop.  20% is initialization overhead which is only appearing large because the total runtime is so small.
```
Total time: 15.424 s
File: train_GAN.py
Function: train_GAN at line 433

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   433                                           @profile
   434                                           def train_GAN(_batch_size, _training_iterations, global_size):
   435
   436
   437
   438         1     975530.0 975530.0      6.3      generator = Generator()
   439
   440         1         78.0     78.0      0.0      random_input = numpy.random.uniform(-1,1,[1,100]).astype(numpy.float32)
   441         1    1819987.0 1819987.0     11.8      generated_image = generator(random_input)
   442
   443
   444         1      17879.0  17879.0      0.1      discriminator = Discriminator()
   445         1      29422.0  29422.0      0.2      classification  = discriminator(generated_image)
   446
   447         1          3.0      3.0      0.0      models = {
   448         1          1.0      1.0      0.0          "generator" : generator,
   449         1          0.0      0.0      0.0          "discriminator" : discriminator
   450                                               }
   451
   452         1          1.0      1.0      0.0      opts = {
   453         1        239.0    239.0      0.0          "generator" : tf.keras.optimizers.Adam(0.001),
   454         1        148.0    148.0      0.0          "discriminator" : tf.keras.optimizers.RMSprop(0.0001)
   455
   456
   457                                               }
   458
   459         1          0.0      0.0      0.0      if global_size != 1:
   460                                                   hvd.broadcast_variables(generator.variables, root_rank=0)
   461                                                   hvd.broadcast_variables(discriminator.variables, root_rank=0)
   462                                                   hvd.broadcast_variables(opts['generator'].variables(), root_rank=0)
   463                                                   hvd.broadcast_variables(opts['discriminator'].variables(), root_rank=0)
   464
   465         1   12505979.0 12505979.0     81.1      train_loop(_batch_size, _training_iterations, models, opts, global_size)
   466
   467
   468                                               # Save the model:
   469         1      74740.0  74740.0      0.5      generator.save_weights("trained_GAN.h5")
```


Digging into the `train_loop` function, most of our time is spent in the `forward_pass` function so let's dig there too

```
   Total time: 12.5054 s
   File: train_GAN.py
   Function: train_loop at line 391

   Line #      Hits         Time  Per Hit   % Time  Line Contents
   ==============================================================
      391                                           @profile
      392                                           def train_loop(batch_size, n_training_iterations, models, opts, global_size):
      393
      394         1         14.0     14.0      0.0      logger = logging.getLogger()
      395
      396         1         34.0     34.0      0.0      rank = hvd.rank()
      397        21         22.0      1.0      0.0      for i in range(n_training_iterations):
      398
      399        20         22.0      1.1      0.0          start = time.time()
      400
      401        60         64.0      1.1      0.0          for network in ["generator", "discriminator"]:
      402
      403        40        975.0     24.4      0.0              with tf.GradientTape() as tape:
      404        80   11374986.0 142187.3     91.0                      loss, metrics, images = forward_pass(
      405        40         35.0      0.9      0.0                          models["generator"],
      406        40         20.0      0.5      0.0                          models["discriminator"],
      407        40         24.0      0.6      0.0                          _input_size = 100,
      408        40         26.0      0.7      0.0                          _batch_size = batch_size,
      409                                                               )
      410
      411
      412        40         37.0      0.9      0.0              if global_size != 1:
      413                                                           tape = hvd.DistributedGradientTape(tape)
      414
      415        40       3910.0     97.8      0.0              if loss["discriminator"] < 0.01:
      416                                                           break
      417
      418
      419        40       8287.0    207.2      0.1              trainable_vars = models[network].trainable_variables
      420
      421                                                       # Apply the update to the network (one at a time):
      422        40     840750.0  21018.8      6.7              grads = tape.gradient(loss[network], trainable_vars)
      423
      424        40     267638.0   6690.9      2.1              opts[network].apply_gradients(zip(grads, trainable_vars))
      425
      426        20         72.0      3.6      0.0          end = time.time()
      427
      428        20        363.0     18.1      0.0          images = batch_size*2*global_size
      429
      430        20       8074.0    403.7      0.1          logger.info(f"G Loss: {loss['generator']:.3f}, D Loss: {loss['discriminator']:.3f}, step_time: {end-start :.3f}, throughput: {images/(end-start):.3f} img/s
```


And in forward_pass, we see that almost all the time is fetching the real data!
```Total time: 11.372 s
File: train_GAN.py
Function: forward_pass at line 301

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   301                                           @profile
   302                                           def forward_pass(_generator, _discriminator, _batch_size, _input_size):
   303                                                   '''
   304                                                   This function takes the two models and runs a forward pass to the computation of the loss functions
   305                                                   '''
   306
   307                                                   # Fetch real data:
   308        40   10498888.0 262472.2     92.3          real_data = fetch_real_batch(_batch_size)
   309
   310
   311
   312                                                   # Use the generator to make fake images:
   313        40       3136.0     78.4      0.0          random_noise = numpy.random.uniform(-1, 1, size=_batch_size*_input_size).astype(numpy.float32)
   314        40         96.0      2.4      0.0          random_noise = random_noise.reshape([_batch_size, _input_size])
   315        40     343082.0   8577.0      3.0          fake_images  = _generator(random_noise)
   316
   317
   318                                                   # Use the discriminator to make a prediction on the REAL data:
   319        40     238890.0   5972.2      2.1          prediction_on_real_data = _discriminator(real_data)
   320                                                   # Use the discriminator to make a prediction on the FAKE data:
   321        40     211699.0   5292.5      1.9          prediction_on_fake_data = _discriminator(fake_images)
   322
   323
   324        40         51.0      1.3      0.0          soften = 0.1
   325        40        601.0     15.0      0.0          real_labels = numpy.zeros([_batch_size,1], dtype=numpy.float32) + soften
   326        40        707.0     17.7      0.0          fake_labels = numpy.ones([_batch_size,1],  dtype=numpy.float32) - soften
   327        40         71.0      1.8      0.0          gen_labels  = numpy.zeros([_batch_size,1], dtype=numpy.float32)
   328
   329
   330                                                   # Occasionally, we disrupt the discriminator (since it has an easier job)
   331
   332                                                   # Invert a few of the discriminator labels:
   333
   334        40         86.0      2.1      0.0          n_swap = int(_batch_size * 0.1)
   335
   336        40        132.0      3.3      0.0          real_labels [0:n_swap] = 1.
   337        40         70.0      1.8      0.0          fake_labels [0:n_swap] = 0.
   338
   339
   340                                                   # Compute the loss for the discriminator on the real images:
   341        80      22516.0    281.4      0.2          discriminator_real_loss = compute_loss(
   342        40         28.0      0.7      0.0              _logits  = prediction_on_real_data,
   343        40         29.0      0.7      0.0              _targets = real_labels)
   344
   345                                                   # Compute the loss for the discriminator on the fakse images:
   346        80      16348.0    204.3      0.1          discriminator_fake_loss = compute_loss(
   347        40         28.0      0.7      0.0              _logits  = prediction_on_fake_data,
   348        40         26.0      0.7      0.0              _targets = fake_labels)
   349
   350                                                   # The generator loss is based on the output of the discriminator.
   351                                                   # It wants the discriminator to pick the fake data as real
   352        40         72.0      1.8      0.0          generator_target_labels = [1] * _batch_size
   353
   354        80      16205.0    202.6      0.1          generator_loss = compute_loss(
   355        40         28.0      0.7      0.0              _logits  = prediction_on_fake_data,
   356        40         26.0      0.7      0.0              _targets = real_labels)
   357
   358                                                   # Average the discriminator loss:
   359        40       4395.0    109.9      0.0          discriminator_loss = 0.5*(discriminator_fake_loss  + discriminator_real_loss)
   360
   361                                                   # Calculate the predicted label (real or fake) to calculate the accuracy:
   362        40       3256.0     81.4      0.0          predicted_real_label = numpy.argmax(prediction_on_real_data.numpy(), axis=-1)
   363        40       1961.0     49.0      0.0          predicted_fake_label = numpy.argmax(prediction_on_fake_data.numpy(), axis=-1)
   364
   365        80       2742.0     34.3      0.0          discriminator_accuracy = 0.5 * numpy.mean(predicted_real_label == real_labels) + \
   366        40       1118.0     27.9      0.0              0.5 * numpy.mean(predicted_fake_label == fake_labels)
   367        40        984.0     24.6      0.0          generator_accuracy = 0.5 * numpy.mean(predicted_fake_label == generator_target_labels)
   368
   369
   370        40         31.0      0.8      0.0          metrics = {
   371        40         33.0      0.8      0.0              "discriminator" : discriminator_accuracy,
   372        40         30.0      0.8      0.0              "generator"    : generator_accuracy
   373                                                   }
   374
   375        40         35.0      0.9      0.0          loss = {
   376        40         26.0      0.7      0.0              "discriminator" : discriminator_loss,
   377        40         30.0      0.8      0.0              "generator"    : generator_loss
   378                                                   }
   379
   380        40         50.0      1.2      0.0          images = {
   381        40        141.0      3.5      0.0              "real" : real_data[0].reshape([28,28]),
   382        40       4290.0    107.2      0.0              "fake" : fake_images.numpy()[0].reshape([28,28])
   383                                                   }
   384
   385
   386        40         29.0      0.7      0.0          return loss, metrics, images
```


What's in that function?
```python
def fetch_real_batch(_batch_size):
    x_train, x_test = get_dataset()

    indexes = numpy.random.choice(a=x_train.shape[0], size=[_batch_size,])

    images = x_train[indexes].reshape(_batch_size, 28, 28, 1)

    return images
```

Well, we call `get_dataset` every time!  If we make the dataset a global, that we read from instead of reload, we should get a big improvement:

```2021-04-28 20:30:45,062 - INFO - G Loss: 0.668, D Loss: 0.709, step_time: 0.532, throughput: 240.770 img/s.
2021-04-28 20:30:45,198 - INFO - G Loss: 0.827, D Loss: 0.643, step_time: 0.136, throughput: 943.823 img/s.
2021-04-28 20:30:45,290 - INFO - G Loss: 0.855, D Loss: 0.610, step_time: 0.092, throughput: 1398.356 img/s.
2021-04-28 20:30:45,361 - INFO - G Loss: 0.891, D Loss: 0.594, step_time: 0.070, throughput: 1818.225 img/s.
2021-04-28 20:30:45,431 - INFO - G Loss: 0.892, D Loss: 0.580, step_time: 0.070, throughput: 1828.031 img/s.
2021-04-28 20:30:45,501 - INFO - G Loss: 0.880, D Loss: 0.583, step_time: 0.070, throughput: 1829.558 img/s.
2021-04-28 20:30:45,560 - INFO - G Loss: 0.859, D Loss: 0.584, step_time: 0.058, throughput: 2194.131 img/s.
2021-04-28 20:30:45,618 - INFO - G Loss: 0.811, D Loss: 0.610, step_time: 0.058, throughput: 2201.825 img/s.
2021-04-28 20:30:45,677 - INFO - G Loss: 0.715, D Loss: 0.628, step_time: 0.058, throughput: 2202.837 img/s.
2021-04-28 20:30:45,735 - INFO - G Loss: 0.685, D Loss: 0.630, step_time: 0.058, throughput: 2199.912 img/s.
2021-04-28 20:30:45,812 - INFO - G Loss: 0.672, D Loss: 0.618, step_time: 0.059, throughput: 2182.446 img/s.
2021-04-28 20:30:45,870 - INFO - G Loss: 0.660, D Loss: 0.621, step_time: 0.058, throughput: 2208.964 img/s.
2021-04-28 20:30:45,928 - INFO - G Loss: 0.653, D Loss: 0.630, step_time: 0.058, throughput: 2209.136 img/s.
2021-04-28 20:30:45,987 - INFO - G Loss: 0.646, D Loss: 0.628, step_time: 0.058, throughput: 2214.248 img/s.
2021-04-28 20:30:46,045 - INFO - G Loss: 0.650, D Loss: 0.632, step_time: 0.058, throughput: 2213.353 img/s.
2021-04-28 20:30:46,103 - INFO - G Loss: 0.645, D Loss: 0.641, step_time: 0.058, throughput: 2210.919 img/s.
2021-04-28 20:30:46,161 - INFO - G Loss: 0.648, D Loss: 0.643, step_time: 0.058, throughput: 2210.264 img/s.
2021-04-28 20:30:46,219 - INFO - G Loss: 0.658, D Loss: 0.646, step_time: 0.058, throughput: 2207.302 img/s.
2021-04-28 20:30:46,278 - INFO - G Loss: 0.680, D Loss: 0.652, step_time: 0.058, throughput: 2213.554 img/s.
2021-04-28 20:30:46,341 - INFO - G Loss: 0.671, D Loss: 0.655, step_time: 0.063, throughput: 2020.438 img/s.
Wrote profile results to train_GAN_iofix.py.lprof
```

Even with the profiler still running, we're at 2000 Img/s - a 10x improvement!

We'll pick up again in the `tf_function` folder.  But first, here's a report on the line-by-line profiling:

```
Total time: 0.722145 s
File: train_GAN_iofix.py
Function: forward_pass at line 297

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   297                                           @profile
   298                                           def forward_pass(_generator, _discriminator, _batch_size, _input_size):
   299                                                   '''
   300                                                   This function takes the two models and runs a forward pass to the computation of the loss functions
   301                                                   '''
   302
   303                                                   # Fetch real data:
   304        40       4309.0    107.7      0.6          real_data = fetch_real_batch(_batch_size)
   305
   306
   307
   308                                                   # Use the generator to make fake images:
   309        40       2625.0     65.6      0.4          random_noise = numpy.random.uniform(-1, 1, size=_batch_size*_input_size).astype(numpy.float32)
   310        40         76.0      1.9      0.0          random_noise = random_noise.reshape([_batch_size, _input_size])
   311        40     295528.0   7388.2     40.9          fake_images  = _generator(random_noise)
   312
   313
   314                                                   # Use the discriminator to make a prediction on the REAL data:
   315        40     180736.0   4518.4     25.0          prediction_on_real_data = _discriminator(real_data)
   316                                                   # Use the discriminator to make a prediction on the FAKE data:
   317        40     167606.0   4190.1     23.2          prediction_on_fake_data = _discriminator(fake_images)
   318
   319
   320        40         55.0      1.4      0.0          soften = 0.1
   321        40        487.0     12.2      0.1          real_labels = numpy.zeros([_batch_size,1], dtype=numpy.float32) + soften
   322        40        523.0     13.1      0.1          fake_labels = numpy.ones([_batch_size,1],  dtype=numpy.float32) - soften
   323        40         71.0      1.8      0.0          gen_labels  = numpy.zeros([_batch_size,1], dtype=numpy.float32)
   324
   325
   326                                                   # Occasionally, we disrupt the discriminator (since it has an easier job)
   327
   328                                                   # Invert a few of the discriminator labels:
   329
   330        40         77.0      1.9      0.0          n_swap = int(_batch_size * 0.1)
   331
   332        40        116.0      2.9      0.0          real_labels [0:n_swap] = 1.
   333        40         66.0      1.6      0.0          fake_labels [0:n_swap] = 0.
   334
   335
   336                                                   # Compute the loss for the discriminator on the real images:
   337        80      19489.0    243.6      2.7          discriminator_real_loss = compute_loss(
   338        40         31.0      0.8      0.0              _logits  = prediction_on_real_data,
   339        40         26.0      0.7      0.0              _targets = real_labels)
   340
   341                                                   # Compute the loss for the discriminator on the fakse images:
   342        80      16467.0    205.8      2.3          discriminator_fake_loss = compute_loss(
   343        40         29.0      0.7      0.0              _logits  = prediction_on_fake_data,
   344        40         32.0      0.8      0.0              _targets = fake_labels)
   345
   346                                                   # The generator loss is based on the output of the discriminator.
   347                                                   # It wants the discriminator to pick the fake data as real
   348        40         74.0      1.9      0.0          generator_target_labels = [1] * _batch_size
   349
   350        80      15979.0    199.7      2.2          generator_loss = compute_loss(
   351        40         24.0      0.6      0.0              _logits  = prediction_on_fake_data,
   352        40         31.0      0.8      0.0              _targets = real_labels)
   353
   354                                                   # Average the discriminator loss:
   355        40       4046.0    101.2      0.6          discriminator_loss = 0.5*(discriminator_fake_loss  + discriminator_real_loss)
   356
   357                                                   # Calculate the predicted label (real or fake) to calculate the accuracy:
   358        40       2880.0     72.0      0.4          predicted_real_label = numpy.argmax(prediction_on_real_data.numpy(), axis=-1)
   359        40       1876.0     46.9      0.3          predicted_fake_label = numpy.argmax(prediction_on_fake_data.numpy(), axis=-1)
   360
   361        80       2317.0     29.0      0.3          discriminator_accuracy = 0.5 * numpy.mean(predicted_real_label == real_labels) + \
   362        40       1107.0     27.7      0.2              0.5 * numpy.mean(predicted_fake_label == fake_labels)
   363        40        969.0     24.2      0.1          generator_accuracy = 0.5 * numpy.mean(predicted_fake_label == generator_target_labels)
   364
   365
   366        40         34.0      0.8      0.0          metrics = {
   367        40         32.0      0.8      0.0              "discriminator" : discriminator_accuracy,
   368        40         32.0      0.8      0.0              "generator"    : generator_accuracy
   369                                                   }
   370
   371        40         34.0      0.8      0.0          loss = {
   372        40         35.0      0.9      0.0              "discriminator" : discriminator_loss,
   373        40         30.0      0.8      0.0              "generator"    : generator_loss
   374                                                   }
   375
   376        40         47.0      1.2      0.0          images = {
   377        40        139.0      3.5      0.0              "real" : real_data[0].reshape([28,28]),
   378        40       4077.0    101.9      0.6              "fake" : fake_images.numpy()[0].reshape([28,28])
   379                                                   }
   380
   381
   382        40         33.0      0.8      0.0          return loss, metrics, images

Total time: 1.83035 s
File: train_GAN_iofix.py
Function: train_loop at line 387

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   387                                           @profile
   388                                           def train_loop(batch_size, n_training_iterations, models, opts, global_size):
   389
   390         1         13.0     13.0      0.0      logger = logging.getLogger()
   391
   392         1         30.0     30.0      0.0      rank = hvd.rank()
   393        21         25.0      1.2      0.0      for i in range(n_training_iterations):
   394
   395        20         23.0      1.1      0.0          start = time.time()
   396
   397        60         73.0      1.2      0.0          for network in ["generator", "discriminator"]:
   398
   399        40        857.0     21.4      0.0              with tf.GradientTape() as tape:
   400        80     724663.0   9058.3     39.6                      loss, metrics, images = forward_pass(
   401        40         33.0      0.8      0.0                          models["generator"],
   402        40         20.0      0.5      0.0                          models["discriminator"],
   403        40         26.0      0.7      0.0                          _input_size = 100,
   404        40         30.0      0.8      0.0                          _batch_size = batch_size,
   405                                                               )
   406
   407
   408        40         36.0      0.9      0.0              if global_size != 1:
   409                                                           tape = hvd.DistributedGradientTape(tape)
   410
   411        40       3634.0     90.8      0.2              if loss["discriminator"] < 0.01:
   412                                                           break
   413
   414
   415        40       7901.0    197.5      0.4              trainable_vars = models[network].trainable_variables
   416
   417                                                       # Apply the update to the network (one at a time):
   418        40     787618.0  19690.5     43.0              grads = tape.gradient(loss[network], trainable_vars)
   419
   420        40     261629.0   6540.7     14.3              opts[network].apply_gradients(zip(grads, trainable_vars))
   421
   422        20         42.0      2.1      0.0          end = time.time()
   423
   424        20        322.0     16.1      0.0          images = batch_size*2*global_size
   425
   426        20      43378.0   2168.9      2.4          logger.info(f"G Loss: {loss['generator']:.3f}, D Loss: {loss['discriminator']:.3f}, step_time: {end-start :.3f}, throughput: {images/(end-start):.3f} img/s.")
```

As you can see, the dominant calls are the generator, the discriminator (twice), and the gradient calculations.  We will see how to speed those up next.
