# 実行ガイド: Phase 1 Extended & Phase 2 Column Optimization

## 概要

Phase 1グリッドサーチで特定された最良設定を50エポックで評価し、
さらにColumn構造パラメータを最適化して70%精度達成を目指します。

## 実装完了項目

✅ **Task 1**: `--seed`引数の実装（v026/v027）
- デフォルト: None（ランダム）
- `--seed 42`で再現性確保

✅ **Task 2**: grid_search_phase1.pyにシード固定機能追加
- 全実験で`--seed 42`を自動適用

✅ **Task 3**: Phase 1 Extended実行スクリプト作成
- `run_phase1_extended.py`
- 上位5設定を50エポックで評価
- 期待精度: 45-55%

✅ **Task 4**: v027ファイル作成
- `columnar_ed_ann_v027_column_optimization.py`（v026のコピー）

✅ **Task 5**: Phase 2パラメータ実装確認
- `base_column_radius`, `participation_rate`は既に実装済み

✅ **Task 6**: Phase 2グリッドサーチスクリプト作成
- `run_phase2_column_optimization.py`
- 36通りの組み合わせ探索
- 期待精度: 50-60%

---

## 実行手順

### ステップ1: Phase 1 Extended（上位5設定の50エポック評価）

**別ターミナルで実行:**

```bash
cd /home/yoichi/develop/ai/column_ed_snn
source .venv/bin/activate
python run_phase1_extended.py
```

**実行内容:**
- 5設定 × 50エポック
- 推定時間: 約15-20分（3-4分/設定）
- シード: 42（再現性確保）

**評価する設定:**
1. lr=0.20, u1=1.0, lateral_lr=0.05 (Phase 1: 37.90%)
2. lr=0.20, u1=0.8, lateral_lr=0.05 (Phase 1: 35.70%)
3. lr=0.20, u1=0.9, lateral_lr=0.15 (Phase 1: 35.60%)
4. lr=0.20, u1=0.6, lateral_lr=0.15 (Phase 1: 35.10%)
5. lr=0.20, u1=0.5, lateral_lr=0.10 (Phase 1: 35.00%)

**期待結果:**
- テスト精度: 45-55%（10エポック比+10-15%）
- 最良設定の特定

**出力ファイル:**
- `results/phase1_extended/execution_YYYYMMDD_HHMMSS.log`（進捗）
- `results/phase1_extended/config_N_YYYYMMDD_HHMMSS.log`（詳細、各設定）
- `results/phase1_extended/results_summary_YYYYMMDD_HHMMSS.json`（構造化データ）
- `results/phase1_extended/results_summary_YYYYMMDD_HHMMSS.csv`（テーブル形式）

---

### ステップ2: Phase 2 Column Optimization（Column構造最適化）

**Phase 1 Extended完了後、別ターミナルで実行:**

```bash
cd /home/yoichi/develop/ai/column_ed_snn
source .venv/bin/activate
python run_phase2_column_optimization.py
```

**実行内容:**
- 36通りの組み合わせ（6 × 6）
- 推定時間: 約1.5-2時間（3分/実験）
- シード: 42（再現性確保）

**固定パラメータ（Phase 1 Best）:**
- learning_rate: 0.20
- u1: 1.0
- lateral_lr: 0.05
- epochs: 50

**探索パラメータ:**
- base_column_radius: [0.8, 0.9, 1.0, 1.1, 1.2, 1.5]
  * 0.8-0.9: より密な重複、特徴の統合
  * 1.0: 現在のデフォルト
  * 1.1-1.5: より疎な重複、特徴の多様性

- participation_rate: [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
  * 0.5-0.7: 高スパース性、過学習抑制
  * 0.8-0.9: 中程度のスパース性
  * 1.0: 全参加、重複なし（デフォルト）

**期待結果:**
- テスト精度: 50-60%（Phase 1 Extendedから+5-10%）
- 最適Column構造の発見

**出力ファイル:**
- `results/phase2/execution_YYYYMMDD_HHMMSS.log`（進捗）
- `results/phase2/exp_XXX_of_036_YYYYMMDD_HHMMSS.log`（詳細、各実験）
- `results/phase2/results_summary_YYYYMMDD_HHMMSS.json`（構造化データ）
- `results/phase2/results_summary_YYYYMMDD_HHMMSS.csv`（テーブル形式）

---

## ステップ3: 結果の比較分析

**両方の実行完了後:**

```bash
python << 'EOF'
import pandas as pd
import numpy as np

# Phase 1 Extended結果読み込み
phase1_csv = "results/phase1_extended/results_summary_*.csv"  # 実際のファイル名に置換
phase2_csv = "results/phase2/results_summary_*.csv"  # 実際のファイル名に置換

# 最新のファイルを自動検出
import glob
phase1_files = sorted(glob.glob("results/phase1_extended/results_summary_*.csv"))
phase2_files = sorted(glob.glob("results/phase2/results_summary_*.csv"))

if not phase1_files or not phase2_files:
    print("❌ Result files not found. Run experiments first.")
    exit(1)

df_phase1 = pd.read_csv(phase1_files[-1])
df_phase2 = pd.read_csv(phase2_files[-1])

print("="*80)
print("PHASE 1 EXTENDED vs PHASE 2 COLUMN OPTIMIZATION")
print("="*80)

# Phase 1 Extended最良
phase1_best = df_phase1.loc[df_phase1['final_test_acc'].idxmax()]
print(f"\n【Phase 1 Extended Best】")
print(f"  Config: lr={phase1_best['learning_rate']:.2f}, u1={phase1_best['u1']:.1f}, lateral_lr={phase1_best['lateral_lr']:.2f}")
print(f"  Test Acc: {phase1_best['final_test_acc']:.4f} ({phase1_best['final_test_acc']*100:.2f}%)")
print(f"  Train Acc: {phase1_best['final_train_acc']:.4f}")

# Phase 2最良
phase2_best = df_phase2.loc[df_phase2['final_test_acc'].idxmax()]
print(f"\n【Phase 2 Column Optimization Best】")
print(f"  Column Params: base_radius={phase2_best['base_column_radius']:.1f}, participation_rate={phase2_best['participation_rate']:.1f}")
print(f"  Test Acc: {phase2_best['final_test_acc']:.4f} ({phase2_best['final_test_acc']*100:.2f}%)")
print(f"  Train Acc: {phase2_best['final_train_acc']:.4f}")

# 改善効果
improvement = (phase2_best['final_test_acc'] - phase1_best['final_test_acc']) * 100
print(f"\n【Improvement】")
print(f"  Column Optimization Effect: {improvement:+.2f}%")
if phase2_best['final_test_acc'] >= 0.70:
    print(f"  🎯 Target Achieved! 70%+ accuracy reached!")
elif phase2_best['final_test_acc'] >= 0.60:
    print(f"  ✅ Good progress! 60%+ achieved, approaching 70% target")
else:
    print(f"  ⚠️ Further optimization may be needed for 70% target")

# 統計
print(f"\n【Statistics】")
print(f"Phase 1 Extended:")
print(f"  Mean: {df_phase1['final_test_acc'].mean():.4f}, Max: {df_phase1['final_test_acc'].max():.4f}")
print(f"Phase 2 Column Opt:")
print(f"  Mean: {df_phase2['final_test_acc'].mean():.4f}, Max: {df_phase2['final_test_acc'].max():.4f}")

# 上位5件（Phase 2）
print(f"\n【Phase 2 Top 5 Configurations】")
top5_phase2 = df_phase2.nlargest(5, 'final_test_acc')
for i, (idx, row) in enumerate(top5_phase2.iterrows(), 1):
    print(f"  {i}. base_radius={row['base_column_radius']:.1f}, participation_rate={row['participation_rate']:.1f} "
          f"→ Test={row['final_test_acc']:.4f} ({row['final_test_acc']*100:.2f}%)")

print("\n" + "="*80)
EOF
```

---

## トラブルシューティング

### 実行中のエラー

**問題**: `ModuleNotFoundError`
```bash
# 仮想環境の確認
source .venv/bin/activate
pip list | grep tensorflow
```

**問題**: タイムアウト（30分超過）
- ログファイルで進捗確認
- エポック数を減らして再実行（`--epochs 30`）

### 結果が期待より低い

**Phase 1 Extended < 45%の場合:**
- シード固定の確認（`--seed 42`が有効か）
- 10エポック結果との比較（改善傾向の確認）
- データローダーの動作確認

**Phase 2 < 50%の場合:**
- Phase 1 Extendedの最良設定を確認
- Column構造パラメータの探索範囲を拡張
  * `base_column_radius`: [0.6, 0.7, ..., 2.0]
  * `participation_rate`: [0.3, 0.4, ..., 1.0]

---

## 次のステップ（70%未達成の場合）

### Phase 3: 追加の最適化

1. **u2パラメータの探索**
   - 現在固定: 0.8
   - 探索範囲: [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

2. **隠れ層構成の再検討**
   - 現在: [256, 128]
   - 候補: [512, 256], [256, 256], [384, 192]

3. **エポック数の延長**
   - 50エポック → 100エポック
   - 学習曲線の分析（早期停止の検討）

4. **データ拡張**
   - 訓練サンプル: 1000 → 5000 or 10000
   - アフィン変換、ノイズ付加

---

## ファイル構成

```
column_ed_snn/
├── columnar_ed_ann_v026_multiclass_multilayer.py  # Phase 1用（--seed追加）
├── columnar_ed_ann_v027_column_optimization.py    # Phase 2用（v026のコピー）
├── grid_search_phase1.py                          # Phase 1グリッドサーチ（--seed 42追加）
├── run_phase1_extended.py                         # Phase 1 Extended実行スクリプト ★
├── run_phase2_column_optimization.py              # Phase 2実行スクリプト ★
├── results/
│   ├── phase1/                                    # Phase 1結果（10エポック、210通り）
│   ├── phase1_extended/                           # Phase 1 Extended結果（50エポック、5設定）★
│   └── phase2/                                    # Phase 2結果（50エポック、36通り）★
```

**★ = 今回新規作成**

---

## 期待されるタイムライン

1. **Phase 1 Extended**: 15-20分
2. **Phase 2**: 1.5-2時間
3. **比較分析**: 5分

**合計**: 約2-2.5時間

---

## 成功基準

- ✅ **Phase 1 Extended**: テスト精度45%以上達成
- ✅ **Phase 2**: テスト精度50%以上達成
- 🎯 **最終目標**: テスト精度70%以上達成

---

## 実行開始

```bash
# Phase 1 Extended開始
python run_phase1_extended.py

# （完了後）Phase 2開始
python run_phase2_column_optimization.py

# （完了後）比較分析
# 上記のPythonスクリプトを実行
```

準備完了です。実行を開始してください！
