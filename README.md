# Columnar ED Method (コラムED法)

**コラムED法**は、金子勇氏が考案した誤差拡散学習法（ED法）に大脳皮質のコラム構造を導入したニューラルネットワーク実装です。微分の連鎖律を用いた誤差逆伝播法（Backpropagation）を一切使用せず、生物学的に妥当なアミン拡散機構によって学習を行います。

The **Columnar ED Method** is a neural network implementation that introduces cortical column structure into the Error Diffusion Learning Algorithm (ED method) conceived by Isamu Kaneko. It learns entirely through biologically plausible amine diffusion mechanisms, without using backpropagation based on the chain rule of derivatives.

## 現在の公開状況 / Current Release Status

現在は**簡易版（Simple Version）**のみを公開しています。簡易版は最小限のパラメータ指定で高精度を実現するよう設計されており、コラムED法の動作原理を理解するのに最適です。

Currently, only the **Simple Version** is published. The simple version is designed to achieve high accuracy with minimal parameter configuration, making it ideal for understanding how the Columnar ED Method works.

**フルバージョン（Full Version）は近日公開予定です。** / **The Full Version will be released soon.**

## ドキュメント / Documentation

| | 日本語 | English |
|---|---|---|
| 簡易版ガイド / Simple Version Guide | **[README_simple.md](README_simple.md)** | **[README_simple_en.md](README_simple_en.md)** |
| 動作フロー解説 / Operational Flow | [コラムED法 動作の流れ](docs/ja/コラムED法_動作の流れ.md) | [Columnar ED Method Flow](docs/en/Columnar_ED_Method_Flow.md) |
| ED法解説 / ED Method Explanation | [ED法 解説資料](docs/ja/ED法_解説資料.md) | [ED Method Explanation](docs/en/ED_Method_Explanation.md) |
| カスタムデータ / Custom Dataset | [CUSTOM_DATASET_GUIDE.md](CUSTOM_DATASET_GUIDE.md) | [CUSTOM_DATASET_GUIDE.md](CUSTOM_DATASET_GUIDE.md) |

## クイックスタート / Quick Start

```bash
pip install numpy tensorflow PyYAML matplotlib tqdm

# 1層 + Gabor特徴（デフォルト）
python columnar_ed_ann_simple.py --hidden 2048 --train 10000 --test 10000

# 2層 + Gabor特徴（最高精度）
python columnar_ed_ann_simple.py --hidden 2048,1024 --train 10000 --test 10000
```

詳細は [README_simple.md](README_simple.md)（日本語）または [README_simple_en.md](README_simple_en.md)（English）を参照してください。

## ライセンス / License

[LICENSE](LICENSE) を参照してください。 / See [LICENSE](LICENSE) for details.
