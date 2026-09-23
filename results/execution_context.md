# Experiment execution context

All runs used PyTorch 2.2.2+cpu on the same Windows host. The model training protocol, datasets, test candidates and seeds are encoded in each `results/runs/full/*.json` file.

- The nine Grocery runs were sequential, with the host default of 14 PyTorch CPU threads.
- The three MovieLens RoTE runs used the host default of 14 threads.
- MovieLens SASRec and GRU4Rec runs set `OMP_NUM_THREADS=4` and `MKL_NUM_THREADS=4` and overlapped with RoTE runs. GRU4Rec seed 2 was launched in a separate process from seeds 0 and 1.
- As a result, wall-clock times include resource sharing and thread-count differences. They document the CPU cost incurred in this run but are not a controlled efficiency comparison. Do not rank algorithms by these timing figures.
- The reported accuracy metrics are computed from fixed datasets and candidate lists under the same model hyperparameters; each run uses its own stated seed.
