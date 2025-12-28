# columnar_ed_ann_v007.py ED法実装検証レポート

## 検証日時
2025年11月30日

## 検証対象
`columnar_ed_ann_v007.py` のED法実装状況

---

## ED法の仕様（docs/ja/ED法_解説資料.mdより）

### 1. アミン濃度の定義（出力層）

**出力層における誤差信号**:
- `y - o^m > 0` の場合: `d^{m+} = y - o^m`, `d^{m-} = 0`
- `y - o^m < 0` の場合: `d^{m+} = 0`, `d^{m-} = o^m - y`

### 2. 情報拡散（アミン濃度の伝播）

アミン濃度が空間を介して**全ての層に拡散**:
- `d^{k+} = d^{m+}` （全ての層で同じ値）
- `d^{k-} = d^{m-}` （全ての層で同じ値）

### 3. 重み更新則

**興奮性細胞からの結合**:
```
Δw_{ij}^k = ε × d_j^{k+} × f'(o_j^k) × o_i^{k-1} × sign(w_{ij}^k)
```

**抑制性細胞からの結合**:
```
Δw_{ij}^k = ε × d_j^{k-} × f'(o_j^k) × o_i^{k-1} × sign(w_{ij}^k)
```

### 4. 重み制約条件

**同種細胞間（興奮性同士、抑制性同士）**:
```
w_{ij}^k > 0
```

**異種細胞間（興奮性と抑制性）**:
```
w_{ij}^k < 0
```

### 5. ED法の特徴

- **必ずウェイトの絶対値が増える方向に学習が進行**
- **シグモイド関数を入出力関数として使用**
- **概念的にはパーセプトロンに近い単純山登り法**

---

## columnar_ed_ann_v007.pyの実装検証

### Phase 1: 入力層ペア構造 ✅ 実装済み

**場所**: Line 1114-1115
```python
# 【Phase 5】第1層: ED法のペア構造入力を使用（C実装準拠で有効化）
z_input = create_excitatory_inhibitory_pairs(x)
```

**検証結果**: ✅ **正しく実装**
- 入力層で興奮性・抑制性のペアを作成
- 各ペアに同じ値が入力される

---

### Phase 2: 興奮性・抑制性フラグ ✅ 実装済み

**場所**: Line 650-656
```python
# 【Phase 3】第1層のみE/Iフラグを重みに適用（C実装準拠）
if i == 0:
    for k in range(n_out):
        for l in range(n_in):
            # 出力側は全て興奮性(+1)と仮定、入力側のE/Iフラグのみ適用
            w[k, l] *= self.excitatory_inhibitory[l]
```

**検証結果**: ✅ **正しく実装**
- 初期化時に第1層の重みにE/Iフラグを適用
- E/Iフラグは `self.excitatory_inhibitory` に格納

---

### Phase 3: 重み制約 ⚠️ 部分的に実装

**初期化時**: Line 650-656 で実装済み
**学習時**: Line 1153, 1200 で `sign(w)` を使用

**検証結果**: ⚠️ **部分的に実装、ただし仕様と異なる**

#### 問題点:

1. **重み制約の不完全性**:
   - 仕様では「同種細胞間: w > 0、異種細胞間: w < 0」
   - 実装では初期化時のみ制約を適用
   - 学習時は `sign(w)` による方向制御のみ（重みの符号を保証しない）

2. **学習時の制約欠如**:
   ```python
   # Line 1153, 1200
   w_sign_column = np.sign(w_column)
   w_sign_column[w_sign_column == 0] = 1
   ```
   - `sign(w)` を使用しているが、重み更新後に制約を再適用していない
   - 重みの符号が反転する可能性がある

**推奨される修正**:
- 学習後に重みの符号制約を再適用
- または、符号が反転しないように更新量を制限

---

### Phase 4: アミン濃度配列 ✅ 実装済み

**アミン濃度の計算**: Line 953-974
```python
# 出力層における誤差
error_output = y_true_vec - z_output

# アミン濃度の分離（ED法のPhase 4）
d_plus = np.maximum(0, error_output)   # d⁺: 出力を上げたい時
d_minus = np.maximum(0, -error_output) # d⁻: 出力を下げたい時

# アミン濃度を全層に拡散（次元を増やす）
self.amine_concentrations[y_true, :, 0] = d_plus  # d⁺を全層に拡散
self.amine_concentrations[y_true, :, 1] = d_minus # d⁻を全層に拡散
```

**検証結果**: ✅ **正しく実装**
- 出力層の誤差からd⁺とd⁻を計算
- 全層に同じ値を拡散

---

### Phase 5: 重み更新則 ❌ **重大な問題あり**

#### コラムニューロンの更新（Line 1129-1169）

**実装**:
```python
# Line 1140-1143
d_plus_output = self.amine_concentrations[y_true, 0, 0]  # 出力層の d⁺
d_minus_output = self.amine_concentrations[y_true, 0, 1]  # 出力層の d⁻

amine_effect_column_value = d_plus_output + d_minus_output
```

**問題点**: ❌ **ED法の仕様違反**

1. **興奮性・抑制性の区別がない**:
   - 仕様: 興奮性細胞は `d⁺` を使用、抑制性細胞は `d⁻` を使用
   - 実装: `d⁺ + d⁻` を使用（興奮性・抑制性の区別なし）

2. **理論的根拠の欠如**:
   - `d⁺ + d⁻` は仕様に存在しない
   - 誤差の方向情報が失われる

#### 非コラムニューロンの更新（Line 1172-1213）

**実装**:
```python
# Line 1179-1185
error_non_column = error_hidden[non_column_indices]  # [n_non_column]

# アミン濃度の分離（error > 0なら d⁺、error < 0なら d⁻）
d_plus_non_column = np.maximum(0, error_non_column)
d_minus_non_column = np.maximum(0, -error_non_column)

# 現在は全て興奮性ニューロンとみなす → d⁺を使用
amine_effect_non_column = d_plus_non_column  # [n_non_column]
```

**問題点**: ❌ **ED法の仕様違反**

1. **誤差逆伝播の使用**:
   - 仕様: 出力層のアミン濃度 `d^{m+}`, `d^{m-}` を全層に拡散
   - 実装: `error_hidden` から計算（誤差逆伝播に依存）

2. **情報拡散メカニズムの欠如**:
   - ED法の核心である「アミン濃度の空間的拡散」が実装されていない
   - 出力層のアミン濃度が非コラムニューロンに伝播していない

3. **生物学的妥当性の喪失**:
   - 誤差逆伝播に依存することで、ED法の生物学的妥当性が失われる

---

## 重み更新式の比較

### ED法の仕様（興奮性細胞）

```
Δw = ε × d^{k+} × f'(o) × o_prev × sign(w)
```

ここで:
- `d^{k+} = d^{m+}` （出力層から拡散）
- すべての層で同じ値

### columnar_ed_ann_v007.pyの実装

#### コラムニューロン（Line 1156-1157）

```python
amine_f_column = (self.alpha * amine_effect_column * f_prime_column * 0.5).reshape(-1, 1)
delta_w_column = amine_f_column * z_input.reshape(1, -1) * w_sign_column
```

展開すると:
```
Δw = α × (d⁺ + d⁻) × f'(o) × o_prev × sign(w) × 0.5
```

**問題**: `d⁺ + d⁻` は仕様に存在しない

#### 非コラムニューロン（Line 1203-1204）

```python
amine_f_non_column = (self.alpha * amine_effect_non_column * f_prime_non_column * self.non_column_coeff).reshape(-1, 1)
delta_w_non_column = amine_f_non_column * z_input.reshape(1, -1) * w_sign_non_column
```

展開すると:
```
Δw = α × d_error × f'(o) × o_prev × sign(w) × coeff
```

ここで `d_error = max(0, error_hidden)`

**問題**: 
1. `error_hidden` は誤差逆伝播から計算（ED法の仕様外）
2. 出力層のアミン濃度を使用していない

---

## 入力層ペア構造の使用状況

### Phase 1の実装（Line 1114-1115）

```python
# 【Phase 5】第1層: ED法のペア構造入力を使用（C実装準拠で有効化）
z_input = create_excitatory_inhibitory_pairs(x)
```

**検証結果**: ✅ **正しく実装**
- 第1層のみペア構造入力を使用
- 第2層以降は前の層の出力を使用

---

## シグモイド関数の使用

### 実装（Line 802-809）

```python
def tanh_scaled_sigmoid(self, u):
    """Tanhスケール版シグモイド"""
    return (1.0 + np.tanh(u / self.sig)) / 2.0

def tanh_scaled_derivative(self, a, z):
    """Tanhスケール版シグモイドの導関数"""
    return z * (1.0 - z) / self.sig
```

**検証結果**: ⚠️ **標準的なシグモイド関数ではない**

- ED法の仕様: `f(x) = 1 / (1 + exp(-2x/u0))`
- 実装: Tanhスケール版 `f(x) = (1 + tanh(x/sig)) / 2`

**注**: Tanhスケール版は標準シグモイドと数学的に等価だが、微分の計算方法が異なる。

---

## 総合評価

### ✅ 正しく実装されているPhase

1. **Phase 1**: 入力層ペア構造
2. **Phase 2**: 興奮性・抑制性フラグ
4. **Phase 4**: アミン濃度配列（計算と拡散）

### ⚠️ 部分的に実装されているPhase

3. **Phase 3**: 重み制約（初期化時のみ、学習時に再適用なし）

### ❌ 重大な問題があるPhase

5. **Phase 5**: 重み更新則
   - コラムニューロン: `d⁺ + d⁻` の使用（仕様外）
   - 非コラムニューロン: 誤差逆伝播の使用（ED法の仕様違反）

---

## 推奨される修正

### 優先度：高

1. **非コラムニューロンの更新則を修正**
   - `error_hidden` の使用を中止
   - 出力層のアミン濃度 `d^{m+}`, `d^{m-}` を使用
   - ED法の情報拡散メカニズムを実装

2. **コラムニューロンの更新則を修正**
   - `d⁺ + d⁻` の使用を中止
   - 興奮性ニューロン: `d⁺` を使用
   - 抑制性ニューロン: `d⁻` を使用（将来の拡張として）

### 優先度：中

3. **重み制約の強化**
   - 学習後に重みの符号制約を再適用
   - 同種細胞間: `w > 0`
   - 異種細胞間: `w < 0`

### 優先度：低

4. **シグモイド関数の確認**
   - Tanhスケール版と標準版の等価性を確認
   - 必要に応じて標準版に統一

---

## 結論

columnar_ed_ann_v007.pyのED法実装には、**Phase 5（重み更新則）に重大な問題**があります。特に：

1. **非コラムニューロン**: 誤差逆伝播を使用しており、ED法の核心である「アミン濃度の空間的拡散」が実装されていない
2. **コラムニューロン**: `d⁺ + d⁻` を使用しており、興奮性・抑制性の区別が失われている

これらの問題により、**現在の実装はED法の仕様に準拠していません**。修正が必要です。

---

## 参考資料

- `docs/ja/ED法_解説資料.md`
- 金子勇, "誤差拡散学習法のサンプルプログラム", 1999
