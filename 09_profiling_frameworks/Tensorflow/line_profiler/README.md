#Line by Line profiling

The first, and most easily accessible, profiling tool, is the line profiling tool.

Run the profiling tool using `kernprof` instead of python.  This is only for single-node performance.  For example:
```bash
kernprof -l train_GAN.py
```

This will dump the output for 3 functions, the biggest compute users, into a file `train_GAN.py.lprof`.  Let's dump out the line by line calls:

```bash
python -m line_profiler train_GAN.py.lprof
```

