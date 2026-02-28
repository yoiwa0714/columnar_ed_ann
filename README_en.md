[**日本語**](README.md) | **English**

# Columnar ED Method (コラムED法)

The **Columnar ED Method** is a neural network implementation that introduces cortical column structure into the Error Diffusion Learning Algorithm (ED method) conceived by Isamu Kaneko. It learns entirely through biologically plausible amine diffusion mechanisms, without using backpropagation based on the chain rule of derivatives.

## Current Release Status

Currently, only the **Simple Version** is published. The simple version is designed to achieve high accuracy with minimal parameter configuration, making it ideal for understanding how the Columnar ED Method works.

**The Full Version will be released soon.**

## Documentation

| | Japanese | English |
|---|---|---|
| Simple Version Guide | [README_simple.md](README_simple.md) | **[README_simple_en.md](README_simple_en.md)** |
| Operational Flow | [コラムED法 動作の流れ](docs/ja/コラムED法_動作の流れ.md) | [Columnar ED Method Flow](docs/en/Columnar_ED_Method_Flow.md) |
| ED Method Explanation | [ED法 解説資料](docs/ja/ED法_解説資料.md) | [ED Method Explanation](docs/en/ED_Method_Explanation.md) |
| Custom Dataset | [CUSTOM_DATASET_GUIDE.md](CUSTOM_DATASET_GUIDE.md) | [CUSTOM_DATASET_GUIDE.md](CUSTOM_DATASET_GUIDE.md) |

## Quick Start

```bash
pip install numpy tensorflow PyYAML matplotlib tqdm

# 1-layer + Gabor features (default)
python columnar_ed_ann_simple.py --hidden 2048 --train 10000 --test 10000

# 2-layer + Gabor features (best accuracy)
python columnar_ed_ann_simple.py --hidden 2048,1024 --train 10000 --test 10000
```

See [README_simple_en.md](README_simple_en.md) for details.

## License

See [LICENSE](LICENSE) for details.
