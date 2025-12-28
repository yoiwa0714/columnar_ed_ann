# Phase 1 グリッドサーチ実行ガイド

## 📋 概要

**目的**: コア学習パラメータの最適化  
**探索パラメータ**: learning_rate, u1, lateral_lr  
**探索空間**: 7 × 6 × 5 = **210通り**  
**エポック数**: 10（高速探索）  
**推定所要時間**: 約7時間

---

## 🚀 実行方法

### ステップ1: 別ターミナルを開く
VSCodeのターミナルではなく、**システムの別ターミナル**を使用してください。
（Keyboard Interrupt を避けるため）

### ステップ2: プロジェクトディレクトリに移動
```bash
cd /home/yoichi/develop/ai/column_ed_snn
```

### ステップ3: 仮想環境をアクティベート
```bash
source .venv/bin/activate
```

### ステップ4: グリッドサーチを実行
```bash
bash run_grid_search_phase1.sh
```

**確認プロンプトで `y` を入力して開始**

---

## 📊 出力ファイル

実行後、以下のファイルが `results/phase1/` に生成されます：

### 1. 実行ログ
```
results/phase1/execution_YYYYMMDD_HHMMSS.log
```
- グリッドサーチ全体の進捗ログ
- 各実験の簡易結果（1行サマリー）

### 2. 詳細ログ
```
results/phase1/grid_search_phase1_YYYYMMDD_HHMMSS.log
```
- 各実験の完全な出力
- パラメータ、エポック別精度、最終結果
- **分析時にAIに読み込ませる主要ファイル**

### 3. JSON結果
```
results/phase1/results_summary_YYYYMMDD_HHMMSS.json
```
- 全210実験の構造化データ
- プログラム的な分析に使用

### 4. CSV結果
```
results/phase1/results_summary_YYYYMMDD_HHMMSS.csv
```
- テーブル形式の結果
- Excelやスプレッドシートで確認可能
- ソートして上位設定を特定

---

## 📈 結果の見方

### CSVファイルの主要列

| 列名 | 説明 |
|------|------|
| `exp_id` | 実験ID（例: exp_001_of_210） |
| `learning_rate` | 学習率 |
| `u1` | アミン拡散係数（出力層→隠れ層） |
| `lateral_lr` | 側方抑制学習率 |
| `final_test_acc` | **最終テスト精度（重要）** |
| `final_train_acc` | 最終訓練精度 |
| `status` | 実行状態（success/failed/timeout） |

### 分析のポイント

1. **テスト精度の上位10件を確認**
   - CSVを `final_test_acc` でソート（降順）
   - 上位設定のパラメータ傾向を把握

2. **訓練精度 vs テスト精度の差**
   - 差が大きい → 過学習の可能性
   - 差が小さい → 汎化性能が良い

3. **パラメータの傾向分析**
   - learning_rate の最適範囲
   - u1 の最適範囲
   - lateral_lr の最適範囲

---

## 🔍 実行中の監視

### 別ターミナルで進捗確認
```bash
# リアルタイムで実行ログを監視
tail -f results/phase1/execution_*.log
```

### 現在の最良結果を確認
```bash
# 最新のCSVファイルを表示
ls -t results/phase1/results_summary_*.csv | head -1 | xargs cat | sort -t',' -k6 -rn | head -10
```

---

## ⚠️ 注意事項

### 1. 長時間実行
- 約7時間かかる見込み
- バックグラウンド実行を推奨

### 2. バックグラウンド実行方法
```bash
# nohup で実行（ログアウトしても継続）
nohup bash run_grid_search_phase1.sh > grid_search.out 2>&1 &

# 実行確認
ps aux | grep grid_search

# ログ監視
tail -f grid_search.out
```

### 3. 中断と再開
- 中断: `Ctrl+C`
- 再開: 現状では未実装（全実験を最初からやり直し）
- **将来改善**: チェックポイント機能の追加

---

## 📝 実行後のアクション

### 1. 結果ファイルを確認
```bash
ls -lh results/phase1/
```

### 2. CSVをソートして上位10件を表示
```bash
# ヘッダー付きでソート
(head -1 results/phase1/results_summary_*.csv && tail -n +2 results/phase1/results_summary_*.csv | sort -t',' -k6 -rn) | head -11
```

### 3. AIに分析依頼
詳細ログファイル（`grid_search_phase1_*.log`）をVSCodeで開き、内容をAIに共有：

**プロンプト例**:
```
以下のPhase 1グリッドサーチ結果を分析してください：
- 上位10設定のパラメータ傾向
- 各パラメータの影響度
- Phase 2への推奨設定
- 追加で30-50エポック実行すべき設定
```

---

## 🎯 次のステップ

### Phase 1完了後

1. **結果分析**（AIに依頼）
   - 上位設定の特定
   - パラメータ傾向の把握

2. **長時間学習（30-50エポック）**
   - 上位5-10設定で再実行
   - より正確な性能評価

3. **Phase 2準備**
   - Phase 1の最良設定を固定
   - コラム構造パラメータの探索開始

---

## 🛠️ トラブルシューティング

### 問題: 仮想環境がない
```bash
# 仮想環境作成
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # 必要なパッケージをインストール
```

### 問題: 実行中にエラー
```bash
# 詳細ログを確認
cat results/phase1/grid_search_phase1_*.log | grep "エラー"
```

### 問題: メモリ不足
```bash
# システムメモリを確認
free -h

# 必要に応じてスワップ拡張
```

---

## 📞 サポート

問題が発生した場合:
1. `results/phase1/execution_*.log` を確認
2. 詳細ログ（`grid_search_phase1_*.log`）を確認
3. エラーメッセージをAIに共有

---

**作成日**: 2025-12-06  
**バージョン**: v1.0  
**対応スクリプト**: grid_search_phase1.py, run_grid_search_phase1.sh
