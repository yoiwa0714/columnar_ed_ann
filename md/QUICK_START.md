# 実行ガイド - Columnar ED-ANN

## 基本的な実行方法

### 1. 最適パラメータで実行（デフォルト設定）

```bash
python columnar_ed_ann_v026_multiclass_multilayer.py
```

期待される結果:
- Train精度: 86.77% ± 1%
- Test精度: 83.80% ± 0.5%
- 実行時間: 約30-40分（CPU、hidden=512の場合）

### 2. パラメータをカスタマイズして実行

```bash
python columnar_ed_ann_v026_multiclass_multilayer.py \
  --train 3000 \
  --test 1000 \
  --epochs 100 \
  --hidden 128 \
  --lr 0.20 \
  --u1 0.5 \
  --lateral_lr 0.08 \
  --base_column_radius 1.0 \
  --participation_rate 1.0 \
  --seed 42
```

### 3. クイックテスト（5分で完了）

```bash
python columnar_ed_ann_v026_multiclass_multilayer.py \
  --train 500 \
  --test 200 \
  --epochs 10
```

期待される結果:
- Test精度: 40-45%
- 実行時間: 約5分

## コマンドライン引数例

| 引数 | デフォルト | 説明 |
|------|-----------|------|
| `--train` | 3000 | 訓練サンプル数 |
| `--test` | 1000 | テストサンプル数 |
| `--epochs` | 100 | エポック数 |
| `--hidden` | '512' | 隠れ層ニューロン数（例: '256,128'で2層） |
| `--lr` | 0.20 | 学習率 |
| `--u1` | 0.5 | アミン拡散係数 |
| `--lateral_lr` | 0.08 | 側方抑制学習率 |
| `--base_column_radius` | 1.0 | 基準コラム半径 |
| `--participation_rate` | 1.0 | コラム参加率 |
| `--seed` | 42 | 乱数シード（再現性確保） |
| `--no_realtime_plot` | False | リアルタイム可視化を無効化 |
| `--save_weights` | None | 重みを保存（ファイル名指定） |
| `--load_weights` | None | 重みを読み込み（ファイル名指定） |

## 実行例

### 例1: 隠れ層2層で実行

```bash
python columnar_ed_ann_v026_multiclass_multilayer.py \
  --hidden '256,128' \
  --epochs 100
```

### 例2: データ量を増やして実行

```bash
python columnar_ed_ann_v026_multiclass_multilayer.py \
  --train 5000 \
  --test 2000 \
  --epochs 100
```

注意: train=5000では過学習傾向が見られます（train=3000が最適）

### 例3: 異なる乱数シードで実行

```bash
# Seed 42（デフォルト）
python columnar_ed_ann_v026_multiclass_multilayer.py --seed 42

# Seed 123
python columnar_ed_ann_v026_multiclass_multilayer.py --seed 123

# ランダムシード
python columnar_ed_ann_v026_multiclass_multilayer.py --seed None
```

### 例4: 重みの保存と読み込み

```bash
# 学習して重みを保存
python columnar_ed_ann_v026_multiclass_multilayer.py \
  --save_weights best_model

# 保存した重みを読み込んで評価
python columnar_ed_ann_v026_multiclass_multilayer.py \
  --load_weights best_model \
  --epochs 0
```

## トラブルシューティング

### メモリ不足エラー

```bash
# サンプル数を減らす
python columnar_ed_ann_v026_multiclass_multilayer.py \
  --train 1000 \
  --test 500
```

### 実行時間が長い

```bash
# エポック数を減らす
python columnar_ed_ann_v026_multiclass_multilayer.py \
  --epochs 50
```

### リアルタイム可視化でエラー

```bash
# 可視化を無効化
python columnar_ed_ann_v026_multiclass_multilayer.py \
  --no_realtime_plot
```

## 期待される結果

### 隠れ層1層（[128]）の場合

| エポック数 | Train精度 | Test精度 | 実行時間 |
|-----------|----------|---------|---------|
| 10 | 60-65% | 40-45% | ~5分 |
| 50 | 75-80% | 65-70% | ~15分 |
| 100 | 84-85% | 78-79% | ~25分 |

### 隠れ層2層（[256, 128]）の場合

| エポック数 | Train精度 | Test精度 | 実行時間 |
|-----------|----------|---------|---------|
| 100 | 85-90% | 75-80% | ~40分 |

注意: 隠れ層を増やすと精度が向上する可能性がありますが、計算時間も増加します。

## 推奨設定

### 研究・開発用（精度重視）

```bash
python columnar_ed_ann_v026_multiclass_multilayer.py \
  --train 3000 \
  --test 1000 \
  --epochs 100 \
  --hidden 128 \
  --seed 42
```

### デモ・テスト用（速度重視）

```bash
python columnar_ed_ann_v026_multiclass_multilayer.py \
  --train 500 \
  --test 200 \
  --epochs 10 \
  --no_realtime_plot
```

### 論文実験用（再現性重視）

```bash
python columnar_ed_ann_v026_multiclass_multilayer.py \
  --train 3000 \
  --test 1000 \
  --epochs 100 \
  --hidden 128 \
  --seed 42 \
  --save_weights paper_results
```

## ログとレポート

実行後、以下のファイルが生成されます（`--save_weights`指定時）:

- `{name}_weights.npz`: 学習済み重み
- `{name}_report.txt`: 実験レポート（精度、パラメータ等）

## よくある質問

### Q1: v026とv027の違いは？

A: v026とv027は完全に同一のコードです。v027はPhase 2（Column構造最適化）の実行用にコピーされたものです。

### Q2: 最適パラメータは隠れ層1層だけ？

A: はい、現時点で確定しているのは**隠れ層1層（[128]）の場合の最適パラメータ**です。隠れ層2層以上の最適化は今後の課題です。

### Q3: なぜtrain=3000が最適？

A: グリッドサーチの結果、train=3000で最高精度（78.83%）が得られました。train=5000では過学習傾向（73.94%）が見られました。

### Q4: 異なるデータセット（MNIST等）でも動作する？

A: はい、コード内で`fashion_mnist`を`mnist`に変更すれば動作します。ただし、パラメータの再調整が必要な場合があります。

## サポート

問題が発生した場合は、以下を確認してください:

1. Pythonバージョン: 3.8以上
2. 依存パッケージ: `pip install -r requirements.txt`
3. メモリ: 最低4GB RAM推奨

詳細なドキュメントは[README.md](README.md)を参照してください。
