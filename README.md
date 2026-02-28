**日本語** | [English](README_en.md)

# Columnar ED Method (コラムED法)

**コラムED法**は、金子勇氏が考案した誤差拡散学習法（ED法）に大脳皮質のコラム構造を導入したニューラルネットワーク実装です。微分の連鎖律を用いた誤差逆伝播法（Backpropagation）を一切使用せず、生物学的に妥当なアミン拡散機構によって学習を行います。

## 現在の公開状況

現在は**簡易版（Simple Version）**のみを公開しています。簡易版は最小限のパラメータ指定で高精度を実現するよう設計されており、コラムED法の動作原理を理解するのに最適です。

**フルバージョン（Full Version）は近日公開予定です。**

## ドキュメント

| | 日本語 | English |
|---|---|---|
| 簡易版ガイド | **[README_simple.md](README_simple.md)** | [README_simple_en.md](README_simple_en.md) |
| 動作フロー解説 | [コラムED法 動作の流れ](docs/ja/コラムED法_動作の流れ.md) | [Columnar ED Method Flow](docs/en/Columnar_ED_Method_Flow.md) |
| ED法解説 | [ED法 解説資料](docs/ja/ED法_解説資料.md) | [ED Method Explanation](docs/en/ED_Method_Explanation.md) |
| カスタムデータ | [CUSTOM_DATASET_GUIDE.md](CUSTOM_DATASET_GUIDE.md) | [CUSTOM_DATASET_GUIDE.md](CUSTOM_DATASET_GUIDE.md) |

## クイックスタート

```bash
# リポジトリのクローン
git clone https://github.com/yoiwa0714/columnar_ed_ann.git
cd columnar_ed_ann

# 仮想環境の作成（推奨）
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# 依存パッケージのインストール
pip install -r requirements.txt

# 1層 + Gabor特徴（デフォルト）
python columnar_ed_ann_simple.py --hidden 2048 --train 10000 --test 10000

# 2層 + Gabor特徴（最高精度）
python columnar_ed_ann_simple.py --hidden 2048,1024 --train 10000 --test 10000
```

詳細は [README_simple.md](README_simple.md)（日本語）または [README_simple_en.md](README_simple_en.md)（English）を参照してください。

## ライセンス

[LICENSE](LICENSE) を参照してください。
