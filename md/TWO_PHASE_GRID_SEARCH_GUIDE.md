# 2段階グリッドサーチ実行ガイド

## 概要

学習率（lr）とlateral_lrの最適値を効率的に見つけるための2段階グリッドサーチシステムです。

- **Phase 1（粗探索）**: lrとlateral_lrの大まかな最良値を見つける（9実験、約1時間）
- **Phase 2（精密探索）**: Phase 1の結果を基に、u1・u2も含めて精密探索（81実験、約11時間）

## ファイル構成

```
grid_search_phase1.py              # Phase 1 実行スクリプト
grid_search_phase2.py              # Phase 2 実行スクリプト
run_two_phase_grid_search.py       # 統合実行スクリプト（推奨）
```

## クイックスタート

### 方法1: 統合実行（推奨）

Phase 1とPhase 2を自動で連続実行します。

```bash
python3 run_two_phase_grid_search.py
```

**出力ファイル**:
- `two_phase_grid_search_YYYYMMDD_HHMMSS.log` - 統合ログ
- `grid_search_phase1_results.csv` - Phase 1 詳細結果
- `grid_search_phase2_results.csv` - Phase 2 詳細結果
- 各実験のログファイル（`learning_result_*.log`）

**推定時間**: 約12時間（Phase 1: 1時間 + Phase 2: 11時間）

### 方法2: 個別実行

Phase 1とPhase 2を別々に実行したい場合。

#### Phase 1のみ実行

```bash
python3 grid_search_phase1.py
```

結果: `grid_search_phase1_results.csv`

#### Phase 2を手動実行

Phase 1の結果から最良のlr・lateral_lrを確認し、Phase 2に渡します。

```bash
# 例: Phase 1 で lr=0.05, lateral_lr=0.15 が最良だった場合
python3 grid_search_phase2.py --best_lr 0.05 --best_lateral_lr 0.15
```

結果: `grid_search_phase2_results.csv`

## 探索パラメータ

### Phase 1（粗探索）

| パラメータ | 探索値 |
|-----------|--------|
| lr | [0.02, 0.05, 0.08] |
| lateral_lr | [0.10, 0.15, 0.20] |
| u1 | 0.5（固定） |
| u2 | 0.8（固定） |
| epochs | 30 |

**総実験数**: 3×3 = 9

### Phase 2（精密探索）

| パラメータ | 探索値 |
|-----------|--------|
| lr | Phase 1最良値 ± 0.02（3値） |
| lateral_lr | Phase 1最良値 ± 0.03（3値） |
| u1 | [0.4, 0.5, 0.6] |
| u2 | [0.7, 0.8, 0.9] |
| epochs | 50 |

**総実験数**: 3×3×3×3 = 81

## 結果の見方

### CSV形式の結果ファイル

```csv
lr,lateral_lr,u1,u2,train_acc,test_acc,duration,log_file,status
0.05,0.15,0.5,0.8,0.6780,0.6010,480.5,learning_result_*.log,success
```

- **test_acc**: テスト精度（高いほど良い）
- **train_acc**: 訓練精度
- **duration**: 所要時間（秒）
- **status**: 実行状態（success/error）

### 統合ログファイル

`two_phase_grid_search_YYYYMMDD_HHMMSS.log` に以下が記録されます:

1. Phase 1の実行ログ
2. Phase 1の結果サマリー（上位3件）
3. Phase 1の最良パラメータ
4. Phase 2の実行ログ
5. Phase 2の結果サマリー（上位5件）
6. 最終的な最良パラメータ
7. 総所要時間

## バックグラウンド実行（推奨）

長時間実行のため、バックグラウンドで実行することを推奨します。

```bash
# nohup で実行（ターミナルを閉じても継続）
nohup python3 run_two_phase_grid_search.py > grid_search_output.log 2>&1 &

# 実行状況確認
tail -f grid_search_output.log

# または統合ログを確認
tail -f two_phase_grid_search_*.log
```

## 実行中の進捗確認

```bash
# 実行中の実験数を確認
ls learning_result_v027_3_fashion_*.log | wc -l

# 最新の実験ログを確認
ls -lt learning_result_v027_3_fashion_*.log | head -1

# Phase 1完了確認
ls grid_search_phase1_results.csv

# Phase 2完了確認
ls grid_search_phase2_results.csv
```

## 実験の中断と再開

### 中断

```bash
# プロセスIDを確認
ps aux | grep run_two_phase_grid_search.py

# 中断（Ctrl+C または kill）
kill <PID>
```

### 再開

- **Phase 1の途中で中断**: 最初からやり直し
- **Phase 2の途中で中断**: Phase 1の結果CSVから最良値を確認し、Phase 2を手動実行

```bash
# Phase 1の結果を確認
cat grid_search_phase1_results.csv | sort -t',' -k6 -nr | head -1

# Phase 2を再開
python3 grid_search_phase2.py --best_lr <値> --best_lateral_lr <値>
```

## 最良パラメータの適用

Phase 2完了後、最良パラメータで本番実行:

```bash
# 統合ログから最良パラメータを確認
grep "最終的な最良パラメータ" two_phase_grid_search_*.log -A 6

# 例: lr=0.05, lateral_lr=0.15, u1=0.5, u2=0.8 が最良の場合
python3 columnar_ed_ann_v027_3.py \
    --fashion \
    --hidden 512 256 \
    --lr 0.05 \
    --lateral_lr 0.15 \
    --u1 0.5 \
    --u2 0.8 \
    --epochs 100 \
    --train 5000 \
    --test 2000 \
    --seed 42
```

## トラブルシューティング

### エラー: "No such file or directory: columnar_ed_ann_v027_3.py"

→ ワークスペースのルートディレクトリで実行してください

### Phase 2が自動起動しない

→ Phase 1の結果CSVを確認し、手動でPhase 2を実行してください

### メモリ不足エラー

→ 他のプログラムを終了するか、サンプル数を減らしてください（各スクリプト内の`TRAIN_SAMPLES`/`TEST_SAMPLES`を変更）

### 実行時間が長すぎる

→ Phase 1のみ実行し、結果を見てからPhase 2の実行を判断してください

## カスタマイズ

探索範囲を変更したい場合は、各スクリプトの先頭部分を編集:

### grid_search_phase1.py
```python
LR_VALUES = [0.02, 0.05, 0.08]  # ← ここを変更
LATERAL_LR_VALUES = [0.10, 0.15, 0.20]  # ← ここを変更
EPOCHS = 30  # ← エポック数変更
```

### grid_search_phase2.py
```python
U1_VALUES = [0.4, 0.5, 0.6]  # ← ここを変更
U2_VALUES = [0.7, 0.8, 0.9]  # ← ここを変更
EPOCHS = 50  # ← エポック数変更
```

## 修正履歴

### 2025-12-21: 重要な修正
- **問題**: `--hidden`引数の渡し方が誤っており、全ての実験で精度が0になる
- **原因**: `--hidden`は文字列型引数（カンマ区切り）だが、スペース区切りで渡していた
- **修正**: `--hidden 512 256` → `--hidden 512,256` に修正
- **影響**: Phase 1、Phase 2 両方のスクリプトを修正済み

## 補足

- **データセット**: Fashion-MNIST固定（変更する場合はスクリプトを編集）
- **隠れ層**: [512, 256]固定
- **サンプル数**: 訓練2000、テスト2000
- **乱数シード**: 42（再現性確保）
- **1実験あたりの所要時間**: 約8分（環境により変動）
