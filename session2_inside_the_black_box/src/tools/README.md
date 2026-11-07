# Tools

One-time scripts that build the datasets and pre-trained models in [`../data/`](../data/). Workshop
participants do not need them: the files they produce are already in the repository. Run them from the
`session2_inside_the_black_box/src` folder.

| Script | Builds | Used by |
|---|---|---|
| `prep_beans.py` | `data/beans128.npz` (bean-leaf images at 128×128) | `CNN.py`, `CNN_colab.ipynb`, `Autoencoder_colab.ipynb`, `Attention.py` |
| `pretrain_cnn.py` | `data/beans_cnn128.pt` (pre-trained bean-leaf CNN) | `CNN.py` |
| `prep_weather.py` | `data/chachapoyas_hourly.csv` (hourly NASA POWER weather) | `RNN.py`, `RNN_colab.ipynb`, `Autoencoder.py`, `Autoencoder_colab.ipynb` |
| `pretrain_lstm.py` | `data/lstm_chachapoyas.pt` (pre-trained weather LSTM) | `RNN.py` |
| `pretrain_ae.py` | `data/ae_chachapoyas.pt` (pre-trained autoencoders, bottleneck 2, 4 and 48) | `Autoencoder.py` |
| `pretrain_attention.py` | `data/dna_transformer.pt` (pre-trained DNA transformers and CNN) | `Attention.py` |
| `precompute_esm.py` | `data/esm2_ubiquitin.npz` (ESM-2 attention and contact maps; needs `pip install transformers`) | `Attention.py` |

Example:

```bash
python tools/prep_beans.py
python tools/pretrain_cnn.py --epochs 40
```

Each script's options are listed at the top of the file. Run the `prep_*` script before the matching
`pretrain_*` script.
