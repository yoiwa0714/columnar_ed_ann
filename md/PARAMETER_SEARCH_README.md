# パラメータ探索スクリプト - v026_modular_B_simplified

## 概要

v026_modular_B_simplified（均等ニューロン分配版）の最適なパラメータを自動探索するスクリプトです。

## 探索戦略

3段階の探索を実行します：

### Stage 1: 基本パラメータ探索（15通り）
- **learning_rate**: [0.1, 0.2, 0.3, 0.5, 0.7]
- **column_radius**: [2.0, 3.0, 5.0]
- 固定: train_samples=3000, epochs=30, lateral_lr=0.05

### Stage 2: スケールパラメータ探索（9通り）
- **epochs**: [30, 50, 100]
- **train_samples**: [3000, 5000, 10000]
- 固定: Stage 1のベストlr, radius, lateral_lr=0.05

### Stage 3: 微調整パラメータ探索（3通り）
- **lateral_lr**: [0.01, 0.05, 0.1]
- 固定: Stage 1, 2のベストパラメータ

**合計実験数**: 15 + 9 + 3 = **27回**

## 実行方法

### 方法1: シェルスクリプトで実行（推奨）

```bash
# 別ターミナルで実行
./run_parameter_search.sh
```

### 方法2: Pythonスクリプト直接実行

```bash
# 仮想環境をアクティベート
source .venv/bin/activate

# スクリプト実行
python3 parameter_search_v026_B_simplified.py
```

### バックグラウンド実行（長時間実行向け）

```bash
# nohupでバックグラウンド実行
nohup ./run_parameter_search.sh > search_output.log 2>&1 &

# 実行状態確認
tail -f search_output.log

# または、ログディレクトリを監視
tail -f parameter_search_logs/search_*.log
```

## 出力ファイル

探索結果は `parameter_search_logs/` ディレクトリに保存されます：

- `search_YYYYMMDD_HHMMSS.log`: 実行ログ（進捗、結果サマリー）
- `results_YYYYMMDD_HHMMSS.csv`: 全実験結果（CSV形式）

### CSV形式

```
train_samples,epochs,learning_rate,column_radius,lateral_lr,best_test_acc,best_epoch,final_test_acc,execution_time,status
3000,30,0.1,2.0,0.05,0.1234,15,0.1100,125.3,success
...
```

## 予想実行時間

- 1実験あたり: 約2-4分（train_samples=3000, epochs=30固定）
- Stage 1（25回）: 約50-100分
- Stage 2（5回）: 約10-20分
- **合計: 約1-2時間**

## 結果の確認

実行中でも結果を確認できます：

```bash
# 最新のログファイルを確認
ls -lt parameter_search_logs/search_*.log | head -1 | awk '{print $9}' | xargs tail -50

# CSVファイルを確認
ls -lt parameter_search_logs/results_*.csv | head -1 | awk '{print $9}' | xargs cat
```

## トラブルシューティング

### タイムアウトが発生する場合

スクリプト内の `timeout=600` を増やしてください（行93付近）：

```python
result = subprocess.run(
    cmd,
    capture_output=True,
    text=True,
    timeout=1200  # 20分に延長
)
```

### メモリ不足の場合

すでに最小構成（train_samples=3000, epochs=30）で実行しているため、これ以上の削減は推奨されません。

## 最終結果の活用

探索完了後、ログの最後に最適なパラメータが表示されます：

```
最終ベストパラメータ:
  train_samples: 3000 (固定)
  epochs: 30 (固定)
  learning_rate: 0.3
  column_radius: 3.0
  lateral_lr: 0.05

最高精度: 0.6523 (Epoch 28)
```

このパラメータを使って本番実行：

```bash
python3 columnar_ed_ann_v026_multiclass_multilayer_modular_B_simplified.py \
  --train_samples 3000 \
  --epochs 30 \
  --learning_rate 0.3 \
  --base_column_radius 3.0 \
  --lateral_lr 0.05
```

## 注意事項

- **固定条件**: train_samples=3000, epochs=30で全探索を実行
- 探索は約1-2時間かかるため、別ターミナルで実行することを推奨
- バックグラウンド実行する場合は nohup を使用
- ログファイルは自動的に保存されるため、途中で止めても結果は保持される
- CSV から途中結果を分析し、探索範囲を調整することも可能
