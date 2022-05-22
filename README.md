### Dependencies
This program is written in python 3.8 and uses PyTorch 1.8.1.

### Essential codes
- Naive and LSTD transfer-learning can be found at `funcs/transfer_linear_filter.py`.
- Naive meta-learning can be found at `funcs/meta_linear_filter.py`.
- LSTD meta-learning can be found at `funcs/meta_lstd_linear_filter.py`.
- Main file can be found at `main_offline.py`. Detailed usage can be found below.
- Channel dataset generation can be found in `channel_gen` folder.

### How to run the codes

#### Prerequisites (data generation)
- Run `channel_gen/5G_standard_SCM/main_custom.m` to generate 5G standard SCM channel data (default: multi-antenna frequency-selective channel)

#### Train and Test
- For conventional learning (naive), execute `runs/conven_naive.sh`
- For conventional learning (LSTD), execute `runs/conven_LSTD.sh`
- For transfer learning (naive), execute `runs/transfer_naive.sh`
- For transfer learning (LSTD), execute `runs/transfer_LSTD.sh`
- For meta-learning (naive), execute `runs/meta_naive.sh`
- For meta-learning (LSTD), execute `runs/meta_LSTD.sh`
