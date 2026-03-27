# コラムED法の動作原理

## 概要
本資料は、オリジナルED法の記述を土台に、実装済みコラムED法の学習則を論理式として定式化した草案である。

狙いは以下の3点である。

1. 連鎖律に依存しない局所可塑性則として記述する。
2. コラム構造（membership）とランク重み付けが学習へ与える役割を明示する。
3. 実装に対応する更新式を、論文に転記可能な形式で提示する。

---

## 1. 順伝播（実装対応）

入力を興奮/抑制ペア化して

$$
\tilde{\mathbf{x}} \in \mathbb{R}^{2d}, \qquad \mathbf{z}^{(0)} = \tilde{\mathbf{x}}
$$

とおく。

隠れ層 $\ell=1,\dots,L$ は

$$
\mathbf{a}^{(\ell)} = W^{(\ell)}\mathbf{z}^{(\ell-1)}, \qquad
\mathbf{z}^{(\ell)} = \phi\!\left(\mathbf{a}^{(\ell)}\right)
$$

で計算する（主設定は $\phi=\tanh$）。

出力層は

$$
\mathbf{a}^{(o)} = W^{(o)}\mathbf{z}^{(L)}, \qquad
\mathbf{p} = \mathrm{softmax}(\mathbf{a}^{(o)})
$$

とする。

---

## 2. 出力層のED型更新

教師 one-hot を $\mathbf{t}$ とし、出力誤差を

$$
\mathbf{e}^{(o)} = \mathbf{t} - \mathbf{p}
$$

と定義する。

出力飽和抑制項を

$$
\mathbf{s}^{(o)} = |\mathbf{p}| \odot (1 - |\mathbf{p}|)
$$

とし、出力重み更新を

$$
\Delta W^{(o)} = \eta_o\left(\mathbf{e}^{(o)} \odot \mathbf{s}^{(o)}\right)\mathbf{z}^{(L)\top}
$$

と書く。

この式は出力層の局所量のみで構成され、誤差逆伝播の連鎖律を使わない。

---

## 3. アミン生成（純粋ED設定）

正解クラスを $y$ とし、正解クラスのみアミンを生成する。

$$
A_{c,+}^{(o)} = \alpha_0\,(1-p_y)\,\mathbf{1}[c=y], \qquad
A_{c,-}^{(o)} = 0
$$

ここで $\alpha_0$ は初期アミン強度、$\mathbf{1}[\cdot]$ は指示関数である。

---

## 4. 層内拡散とコラム重み付け

層 $\ell$ の拡散係数を

$$
d_\ell =
\begin{cases}
u_1, & \ell=L \\
\nu_2, & \ell<L
\end{cases}
$$

とする（uniform_amine設定では全層 $d_\ell=\nu_1$）。

クラス $c$ に対するコラムmembershipを

$$
M_{c,j}^{(\ell)} \in \{0,1\}
$$

とし、同クラス内の活性ランクを $r_{c,j}^{(\ell)}$、ランクLUTを $g(\cdot)$ とする。

すると学習重みは

$$
\lambda_{c,j}^{(\ell)}=
\begin{cases}
 g\!\left(r_{c,j}^{(\ell)}\right), & M_{c,j}^{(\ell)}=1 \\
 \lambda^{\mathrm{NC}}_{c,j,\ell}, & M_{c,j}^{(\ell)}=0
\end{cases}
$$

で与えられる。

ここで $\lambda^{\mathrm{NC}}_{c,j,\ell}$ は設定に応じて

- 0（非コラム不学習）
- 最近傍クラス帰属NC強度
- 空間減衰拡散
- 均一微小値

のいずれかとなる。

以上より隠れ層のアミン信号は

$$
H_{c,\sigma,j}^{(\ell)} = d_\ell\,A_{c,\sigma}^{(o)}\,\lambda_{c,j}^{(\ell)}
$$

と表せる（$\sigma\in\{+, -\}$）。

---

## 5. 隠れ層更新（コラムED法の中核）

隠れ層の飽和抑制項を

$$
q_j^{(\ell)} = \max\!\left(|z_j^{(\ell)}|(1-|z_j^{(\ell)}|),\,\varepsilon\right)
$$

とし、クラス・符号を総和した局所信号を

$$
u_j^{(\ell)} = \eta_\ell\sum_{c,\sigma} H_{c,\sigma,j}^{(\ell)}\,q_j^{(\ell)}
$$

と定義する。

入力側活動を $\mathbf{z}^{(\ell-1)}$ とすると、基本更新は

$$
\Delta W_{j,:}^{(\ell)} = \nu_j^{(\ell)}\,\mathbf{z}^{(\ell-1)\top}
$$

で与えられる。

実装ではさらに

1. （中間層）符号整合マスク
2. ノルムクリップ
3. コラム行への学習率係数 $\beta_\ell\le 1$

を適用し、

$$
\Delta W_{j,:}^{(\ell)} \leftarrow
\mathrm{Clip}\!\left(\Gamma_j^{(\ell)}\odot\Delta W_{j,:}^{(\ell)}\right),
$$

$$
\Delta W_{j,:}^{(\ell)} \leftarrow
\begin{cases}
\beta_\ell\,\Delta W_{j,:}^{(\ell)}, & j\in\mathcal{C}_\ell \\
\Delta W_{j,:}^{(\ell)}, & j\notin\mathcal{C}_\ell
\end{cases}
$$

として最終的に

$$
W^{(\ell)} \leftarrow W^{(\ell)} + \Delta W^{(\ell)}
$$

で更新する。

---

## 6. コラムED法が学習できる理由（命題）

### 命題1: 正解クラス選択性

$$
A_{c,+}^{(o)} \propto \mathbf{1}[c=y]
$$

により、強化信号は正解クラスのコラム系へ集中する。これが多クラス干渉を低減する。

### 命題2: 局所可塑性の閉包性

各層の更新は

$$
\Delta W \propto
(\text{局所アミン})\times(\text{局所飽和})\times(\text{前層活動})
$$

の積で構成され、連鎖律なしに層ごと独立して定義できる。

### 命題3: 構造化正則化

membership、ランクLUT、コラム係数 $\beta_\ell$ は、

- どのニューロンを学習させるか
- どの程度学習させるか

を幾何学的・階層的に制御する。

この制御により、表現分化と安定性が両立し、実用精度まで学習が到達する。

---

## 実装コード対応表（式↔関数/行）

本節では、上記1〜6項で示した式が実装のどこで具体化されているかを対応付ける。

| 式・概念 | 数学表現（要約） | 実装箇所（関数/行） | 実装上の要点 | 検証ログとの対応（式→観測指標） |
|---|---|---|---|---|
| 順伝播（隠れ層） | $\mathbf{a}^{(\ell)}=W^{(\ell)}\mathbf{z}^{(\ell-1)},\ \mathbf{z}^{(\ell)}=\phi(\mathbf{a}^{(\ell)})$ | [modules_experiment/ed_network.py](../../modules_experiment/ed_network.py#L927), [modules_experiment/activation_functions.py](../../modules_experiment/activation_functions.py#L46) | `forward()` 内で `np.dot` 後に `tanh_activation`（またはleaky_relu）を適用 | エポック別ヒートマップの層活性分布、活性値レンジ（飽和/非飽和の比率） |
| 順伝播（出力層） | $\mathbf{p}=\mathrm{softmax}(W^{(o)}\mathbf{z}^{(L)})$ | [modules_experiment/ed_network.py](../../modules_experiment/ed_network.py#L927), [modules_experiment/activation_functions.py](../../modules_experiment/activation_functions.py#L63) | 最終隠れ層から出力層へ線形写像し `softmax` で確率化 | クラス別テスト正解率表、予測確率分布、勝者選択頻度 |
| E/Iペア化 | $\tilde{\mathbf{x}} = [\mathbf{x}, \mathbf{x}]$ | [modules_experiment/ed_network.py](../../modules_experiment/ed_network.py#L944), [modules_experiment/neuron_structure.py](../../modules_experiment/neuron_structure.py#L32) | `create_ei_pairs(x)` で入力を興奮/抑制ペアに変換 | — |
| 出力誤差 | $\mathbf{e}^{(o)}=\mathbf{t}-\mathbf{p}$ | [modules_experiment/ed_network.py](../../modules_experiment/ed_network.py#L1286) | one-hot `target_probs` と `z_output` の差分で定義 | 学習初期の誤分類率、クラス別誤差推移（正解クラスの取りこぼし） |
| 出力飽和項 | $\mathbf{s}^{(o)}=|\mathbf{p}|\odot(1-|\mathbf{p}|)$ | [modules_experiment/ed_network.py](../../modules_experiment/ed_network.py#L1288) | ED法の飽和抑制項を出力層にも適用 | 出力確率が0/1近傍に張り付く割合、学習後半の更新量減衰 |
| 出力更新 | $\Delta W^{(o)}=\eta_o(\mathbf{e}^{(o)}\odot\mathbf{s}^{(o)})\mathbf{z}^{(L)\top}$ | [modules_experiment/ed_network.py](../../modules_experiment/ed_network.py#L1296) | `np.outer` で直接更新量を計算（連鎖律なし） | Train/Test曲線の立ち上がり速度、クラス別最終精度の収束速度 |
| 正解クラスのみアミン生成 | $A_{c,+}^{(o)}=\alpha_0(1-p_y)\mathbf{1}[c=y],\ A_{c,-}^{(o)}=0$ | [modules_experiment/ed_network.py](../../modules_experiment/ed_network.py#L1305), [modules_experiment/ed_network.py](../../modules_experiment/ed_network.py#L1311) | `error_correct = 1 - z_output[y_true]` を正解クラスの正チャネルにのみ注入 | 「純粋ED」設定時の全体精度、誤差拡散方式比較（有/無） |
| 層別拡散係数 | $d_\ell\in\{u_1,u_2\}$（uniform時は全層$u_1$） | [modules_experiment/ed_network.py](../../modules_experiment/ed_network.py#L1326) | 最終隠れ層とそれ以前で拡散係数を切替 | 層別活性ヒートマップの変化、u1/u2スイープ時のbest/final精度差 |
| コラムmembership | $M_{c,j}^{(\ell)}\in\{0,1\}$ | [modules_experiment/column_structure.py](../../modules_experiment/column_structure.py#L44), [modules_experiment/ed_network.py](../../modules_experiment/ed_network.py#L1348) | `create_column_membership()` で生成し、層ごとの学習マスクとして利用 | コラム診断ログ（クラスごとの割当数）、クラス別精度の偏り |
| ランクLUT重み | $\lambda_{c,j}^{(\ell)}=g(r_{c,j}^{(\ell)})$（コラム内） | [modules_experiment/ed_network.py](../../modules_experiment/ed_network.py#L1373) | クラス内活性ランクを計算し `_learning_weight_lut` で重み付け | top-k設定やLUTモード変更時のbest精度、早期エポックの安定性 |
| NC分岐（非コラム） | $\lambda^{NC}_{c,j,\ell}$ | [modules_experiment/ed_network.py](../../modules_experiment/ed_network.py#L1383), [modules_experiment/ed_network.py](../../modules_experiment/ed_network.py#L1407) | 最近傍帰属・空間拡散・均一微小値など設定に応じて切替 | NC学習ON/OFF比較、`nc_amine_strength` スイープの山型応答 |
| 隠れ層飽和項 | $q_j^{(\ell)}=\max(|z_j^{(\ell)}|(1-|z_j^{(\ell)}|),\varepsilon)$ | [modules_experiment/ed_network.py](../../modules_experiment/ed_network.py#L1466) | 数値安定のため下限 `1e-3` を付与 | 更新停滞の有無、活性値飽和率と精度低下の相関 |
| 隠れ層局所信号 | $u_j^{(\ell)}=\eta_\ell\sum_{c,\sigma}H_{c,\sigma,j}^{(\ell)}q_j^{(\ell)}$ | [modules_experiment/ed_network.py](../../modules_experiment/ed_network.py#L1492), [modules_experiment/ed_network.py](../../modules_experiment/ed_network.py#L1493) | 3D信号をクラス・符号で総和し、入力との外積で更新量を形成 | エポックごとのTrain改善量、層別更新強度（デバッグ統計） |
| 勾配クリップ | $\Delta W\leftarrow\mathrm{Clip}(\Delta W)$ | [modules_experiment/ed_network.py](../../modules_experiment/ed_network.py#L1502) | 行ノルム単位で `gradient_clip` 上限を適用 | `gradient_clip` 値ごとの収束安定性、best-finalギャップ |
| コラム学習率係数 | $\Delta W_{j,:}\leftarrow\beta_\ell\Delta W_{j,:}\ (j\in\mathcal{C}_\ell)$ | [modules_experiment/ed_network.py](../../modules_experiment/ed_network.py#L1514) | コラム行のみ `column_lr_factors` で抑制 | `column_lr_factors` スイープ時の過学習抑制効果、クラス間バランス |
| 重み適用 | $W^{(\ell)}\leftarrow W^{(\ell)}+\Delta W^{(\ell)}$ | [modules_experiment/ed_network.py](../../modules_experiment/ed_network.py#L1530), [modules_experiment/ed_network.py](../../modules_experiment/ed_network.py#L1572) | 出力層・隠れ層を順次加算更新 | 各エポックのTrain/Test推移、最終精度・ベスト到達エポック |
| 可視化 | — | [modules_experiment/visualization_manager.py](../../modules_experiment/visualization_manager.py#L158) | `VisualizationManager` によるリアルタイム学習曲線・ヒートマップ | 可視化出力 |

> **注記**: メイン版（`modules/`）の対応表は、[コラムED法の動作原理詳細.md](コラムED法の動作原理詳細.md) を参照してください。

注記:

- 行番号は現時点の実験版実装に基づく対応であり、コード更新により変動する可能性がある。
- 上表は「連鎖律不使用」の実装上の局所更新構造を明示することを目的としている。

---

## 最小検証プロトコル（3実験で主要命題を検証）

本節では、命題1〜3を最小本数で検証するための実験計画を示す。
各実験は「固定条件」「比較条件」「判定指標」を明確に分離し、再現しやすい形で記述する。

### 共通設定（原則固定）

- データセット: MNIST（必要に応じてFashion-MNISTで追試）
- seed: 42（本検証）
- 学習データ数/テストデータ数: 10000/10000（各エポックで同条件）
- 隠れ層構成: まずは既存の高精度設定を採用
- 学習エポック: 10（必要時のみ20で追試）
- 評価: best_test, final_test, best_epoch, クラス別正解率, 勝者選択頻度

注記:

- コラムED法のテスト正解率は学習データ数の影響を強く受ける。
- 10000サンプル/epoch と 3000サンプル/epoch では、同一設定でも最終精度に有意な差が出るため、比較実験ではデータ数を固定する。

---

### 実験1: 正解クラス選択性（命題1）

目的:
正解クラスのみアミン生成する設定が、多クラス干渉を抑え、精度を向上/維持するか検証する。

比較条件:

1. 純粋ED（現行）: 正解クラスのみアミン生成
2. 対照条件: 誤答クラス側にも学習信号を流す方式（可能な範囲で過去実装を再現）

固定条件:

- ネットワーク構造、lr、init_scales、column_lr_factors、gradient_clip を同一に固定
- seed固定（まず42、可能なら追加seedで再確認）

主要観測指標:

- 全体: best_test, final_test, best-final gap
- 局所: クラス別正解率（特に過去に崩れやすかったクラス）
- 補助: 勝者選択頻度の偏り

判定基準（最小）:

- 純粋EDが対照条件以上の best_test を示し、かつクラス別崩壊（極端な0%近傍）を減らす

実行コマンド雛形:

```bash
# Exp1-A: 純粋ED（現行基準）
python columnar_ed_ann_experiment.py \
	--dataset mnist --train 5000 --test 5000 --epochs 10 --seed 42 \
	--hidden 2048,1024 --column_neurons 10 \
	--init_method he --init_scales 0.7,1.8,0.8 \
	--gradient_clip 0.03 --gabor_features \
	--output_lr 0.15 --non_column_lr 0.15,0.15 --column_lr 0.00075,0.00045 \
	> logs/exp1_pure_ed.log 2>&1

# Exp1-B: 対照条件（誤答クラス側にも学習信号を流す実装ブランチ向け）
# 注: 実験版標準CLIにはこの切替オプションがないため、実装済みブランチの
#     対照オプションに置き換えて実行する。
python columnar_ed_ann_experiment.py \
	--dataset mnist --train 5000 --test 5000 --epochs 10 --seed 42 \
	--hidden 2048,1024 --column_neurons 10 \
	--init_method he --init_scales 0.7,1.8,0.8 \
	--gradient_clip 0.03 --gabor_features \
	--output_lr 0.15 --non_column_lr 0.15,0.15 --column_lr 0.00075,0.00045 \
	<対照条件オプション> \
	> logs/exp1_control.log 2>&1
```

---

### 実験2: 局所可塑性の閉包性（命題2）

目的:
連鎖律なしの局所更新則で、安定した収束が得られるかを検証する。

比較条件:

1. 基準: 現行設定（飽和項あり、clipあり）
2. 変法A: gradient_clip を弱める/外す
3. 変法B: 飽和項の下限設定（epsilon）を変更

固定条件:

- それ以外のハイパーパラメータは完全固定
- 同一seedで開始

主要観測指標:

- 収束速度: best_epoch
- 安定性: best-final gap, 学習終盤の精度振動幅
- 異常検知: 途中急落、NaN/Inf、極端なクラス偏り

判定基準（最小）:

- 基準条件が最も安定（小さいgap）で、かつ同等以上の best_test を示す

実行コマンド雛形:

```bash
# Exp2-A: 基準（clipあり）
python columnar_ed_ann_experiment.py \
	--dataset mnist --train 5000 --test 5000 --epochs 10 --seed 42 \
	--hidden 2048,1024 --column_neurons 10 \
	--init_method he --init_scales 0.7,1.8,0.8 \
	--gradient_clip 0.03 --gabor_features \
	> logs/exp2_baseline_clip003.log 2>&1

# Exp2-B: 変法A（clip弱化/無効）
python columnar_ed_ann_experiment.py \
	--dataset mnist --train 5000 --test 5000 --epochs 10 --seed 42 \
	--hidden 2048,1024 --column_neurons 10 \
	--init_method he --init_scales 0.7,1.8,0.8 \
	--gradient_clip 0.0 --gabor_features \
	> logs/exp2_variant_no_clip.log 2>&1

# Exp2-C: 変法B（飽和項epsilon変更のパッチ版スクリプトを実行）
# 注: 実験版標準CLIにepsilon指定がないため、別名スクリプトで実行する。
python tmp/columnar_ed_ann_experiment_eps_patch.py \
	--dataset mnist --train 5000 --test 5000 --epochs 10 --seed 42 \
	--hidden 2048,1024 --column_neurons 10 \
	--init_method he --init_scales 0.7,1.8,0.8 \
	--gradient_clip 0.03 --gabor_features \
	> logs/exp2_variant_eps_patch.log 2>&1
```

---

### 実験3: 構造化正則化（命題3）

目的:
membership + ランクLUT + コラム学習率係数が、干渉抑制と汎化に寄与するかを検証する。

比較条件（段階的アブレーション）:

1. Full: 現行（membership + LUT + column_lr_factors）
2. Ablation-1: column_lr_factors を弱抑制化/無効化
3. Ablation-2: LUTの差を縮小（可能なら equal系へ）
4. Ablation-3: コラム構造効果を弱める設定（例: column_neurons/participationの変更）

固定条件:

- lr, init_scales, gradient_clip, epochs, seed を固定

主要観測指標:

- 汎化: best_test, final_test, best-final gap
- バランス: クラス別正解率の分散（標準偏差）
- 学習挙動: 勝者選択頻度の偏り、コラム/非コラム寄与の偏在

判定基準（最小）:

- Full条件が、アブレーション群に対して
	- 同等以上の best_test
	- 小さい best-final gap
	- 小さいクラス間ばらつき
	を同時に満たす

実行コマンド雛形:

```bash
# Exp3-A: Full（membership + LUT + column学習率抑制）
python columnar_ed_ann_experiment.py \
	--dataset mnist --train 5000 --test 5000 --epochs 10 --seed 42 \
	--hidden 2048,1024 --column_neurons 10 --participation_rate 0.1 \
	--init_method he --init_scales 0.7,1.8,0.8 \
	--gradient_clip 0.03 --gabor_features \
	--output_lr 0.15 --non_column_lr 0.15,0.15 --column_lr 0.00075,0.00045 \
	> logs/exp3_full.log 2>&1

# Exp3-B: Ablation-1（column学習率抑制を解除）
python columnar_ed_ann_experiment.py \
	--dataset mnist --train 5000 --test 5000 --epochs 10 --seed 42 \
	--hidden 2048,1024 --column_neurons 10 --participation_rate 0.1 \
	--init_method he --init_scales 0.7,1.8,0.8 \
	--gradient_clip 0.03 --gabor_features \
	--output_lr 0.15 --non_column_lr 0.15,0.15 --column_lr 0.15,0.15 \
	> logs/exp3_ablation1_no_column_suppression.log 2>&1

# Exp3-C: Ablation-2（LUT差を縮小する実装ブランチ向け）
# 注: 実験版標準CLIにLUTモード指定がないため、実装済みブランチの
#     LUT切替オプションに置き換えて実行する。
python columnar_ed_ann_experiment.py \
	--dataset mnist --train 5000 --test 5000 --epochs 10 --seed 42 \
	--hidden 2048,1024 --column_neurons 10 --participation_rate 0.1 \
	--init_method he --init_scales 0.7,1.8,0.8 \
	--gradient_clip 0.03 --gabor_features \
	--output_lr 0.15 --non_column_lr 0.15,0.15 --column_lr 0.00075,0.00045 \
	<LUT差縮小オプション> \
	> logs/exp3_ablation2_lut_equal.log 2>&1

# Exp3-D: Ablation-3（コラム構造効果を弱める）
python columnar_ed_ann_experiment.py \
	--dataset mnist --train 5000 --test 5000 --epochs 10 --seed 42 \
	--hidden 2048,1024 --column_neurons 1 --participation_rate 1.0 \
	--init_method he --init_scales 0.7,1.8,0.8 \
	--gradient_clip 0.03 --gabor_features \
	--output_lr 0.15 --non_column_lr 0.15,0.15 --column_lr 0.00075,0.00045 \
	> logs/exp3_ablation3_weak_column_structure.log 2>&1
```

---

### 実行後の集計コマンド雛形（best_test, final_test, gap, クラス別CSV化）

目的:
実験1〜3で生成したログから、比較に必要な最小指標を機械的にCSV化する。

#### 1) best_test / final_test / gap のサマリーCSV

```bash
cd /home/yoichi/develop/ai/columnar_ed_ann

mkdir -p tmp
out="tmp/protocol_summary_$(date +%Y%m%d_%H%M%S).csv"
echo "label,best_test,best_epoch,final_test,final_epoch,gap,log" > "$out"

for l in logs/exp1_*.log logs/exp2_*.log logs/exp3_*.log; do
	[[ -f "$l" ]] || continue
	label=$(basename "$l" .log)

	best_line=$(grep -E 'ベスト精度: Test=' "$l" | tail -n1)
	best_test=$(echo "$best_line" | sed -E 's/.*Test=([0-9]+\.[0-9]+)%.*/\1/')
	best_epoch=$(echo "$best_line" | sed -E 's/.*\(Epoch ([0-9]+)\).*/\1/')

	# エポック表の最終行（形式: epoch train% test% ...）を優先して取得
	final_line=$(grep -E '^[[:space:]]*[0-9]+[[:space:]]+[0-9]+\.[0-9]+%[[:space:]]+[0-9]+\.[0-9]+%' "$l" | tail -n1)
	if [[ -n "$final_line" ]]; then
		final_epoch=$(echo "$final_line" | awk '{print $1}')
		final_test=$(echo "$final_line" | awk '{print $3}' | tr -d '%')
	else
		# フォールバック: 最終精度行から抽出
		fline=$(grep -E '最終精度: Test=' "$l" | tail -n1)
		final_test=$(echo "$fline" | sed -E 's/.*Test=([0-9]+\.[0-9]+)%.*/\1/')
		final_epoch=""
	fi

	gap=$(awk -v b="$best_test" -v f="$final_test" 'BEGIN{if(b==""||f==""){print ""}else{printf "%.4f", b-f}}')
	echo "$label,$best_test,$best_epoch,$final_test,$final_epoch,$gap,$l" >> "$out"
done

echo "SUMMARY=$out"
cat "$out"
```

#### 2) クラス別CSV（best epoch / final epoch の比較）

```bash
cd /home/yoichi/develop/ai/columnar_ed_ann

mkdir -p tmp
class_out="tmp/protocol_class_compare_$(date +%Y%m%d_%H%M%S).csv"
echo "label,best_epoch,best_C2,best_C6,best_C7,best_C9,best_avg,final_epoch,final_C2,final_C6,final_C7,final_C9,final_avg,log" > "$class_out"

for l in logs/exp1_*.log logs/exp2_*.log logs/exp3_*.log; do
	[[ -f "$l" ]] || continue
	label=$(basename "$l" .log)

	be=$(grep -E 'ベスト精度: Test=' "$l" | sed -E 's/.*\(Epoch ([0-9]+)\).*/\1/' | tail -n1)
	fe=$(grep -E '^[[:space:]]*[0-9]+[[:space:]]+[0-9]+\.[0-9]+%[[:space:]]+[0-9]+\.[0-9]+%' "$l" | tail -n1 | awk '{print $1}')

	bline=$(grep -E "^[[:space:]]*${be}[[:space:]]+[0-9]+\.[0-9]+%" "$l" | tail -n1)
	fline=$(grep -E "^[[:space:]]*${fe}[[:space:]]+[0-9]+\.[0-9]+%" "$l" | tail -n1)

	bnorm=$(echo "$bline" | tr -d '%' | xargs)
	fnorm=$(echo "$fline" | tr -d '%' | xargs)

	# 現行ログの列構成（C2/C6/C7/C9/Avg）に合わせた抽出
	bC2=$(echo "$bnorm" | awk '{print $4}')
	bC6=$(echo "$bnorm" | awk '{print $8}')
	bC7=$(echo "$bnorm" | awk '{print $9}')
	bC9=$(echo "$bnorm" | awk '{print $11}')
	bAvg=$(echo "$bnorm" | awk '{print $12}')

	fC2=$(echo "$fnorm" | awk '{print $4}')
	fC6=$(echo "$fnorm" | awk '{print $8}')
	fC7=$(echo "$fnorm" | awk '{print $9}')
	fC9=$(echo "$fnorm" | awk '{print $11}')
	fAvg=$(echo "$fnorm" | awk '{print $12}')

	echo "$label,$be,$bC2,$bC6,$bC7,$bC9,$bAvg,$fe,$fC2,$fC6,$fC7,$fC9,$fAvg,$l" >> "$class_out"
done

echo "CLASS=$class_out"
cat "$class_out"
```

注記:

- 上記2)は、現行ログの列並び（C2/C6/C7/C9/Avg）を前提にしている。
- 全クラス（C0〜C9）を出力する形式に変更した場合は、`awk '{print $N}'` の列番号を対応更新する。

---

### 3実験を一括実行→集計まで自動化するラッパースクリプト雛形

目的:
実験1〜3の実行と、サマリーCSV/クラス別CSVの生成を1本のシェルで自動化する。

使い方:

1. 下記内容を `scripts/run_protocol_3experiments.sh` として保存
2. `chmod +x scripts/run_protocol_3experiments.sh`
3. `bash scripts/run_protocol_3experiments.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

cd /home/yoichi/develop/ai/columnar_ed_ann
mkdir -p logs tmp

ts=$(date +%Y%m%d_%H%M%S)
master_log="logs/protocol_3exp_${ts}.log"

echo "[INFO] master_log=${master_log}" | tee -a "$master_log"

# --- 実験実行 ---
echo "[RUN] Exp1-A pure ED" | tee -a "$master_log"
python columnar_ed_ann_experiment.py \
	--dataset mnist --train 10000 --test 10000 --epochs 10 --seed 42 \
	--hidden 2048,1024 --column_neurons 10 \
	--init_method he --init_scales 0.7,1.8,0.8 \
	--gradient_clip 0.03 --gabor_features \
	--output_lr 0.15 --non_column_lr 0.15,0.15 --column_lr 0.00075,0.00045 \
	> "logs/exp1_pure_ed_${ts}.log" 2>&1

echo "[RUN] Exp2-A baseline clip=0.03" | tee -a "$master_log"
python columnar_ed_ann_experiment.py \
	--dataset mnist --train 10000 --test 10000 --epochs 10 --seed 42 \
	--hidden 2048,1024 --column_neurons 10 \
	--init_method he --init_scales 0.7,1.8,0.8 \
	--gradient_clip 0.03 --gabor_features \
	> "logs/exp2_baseline_clip003_${ts}.log" 2>&1

echo "[RUN] Exp3-A full structured regularization" | tee -a "$master_log"
python columnar_ed_ann_experiment.py \
	--dataset mnist --train 10000 --test 10000 --epochs 10 --seed 42 \
	--hidden 2048,1024 --column_neurons 10 --participation_rate 0.1 \
	--init_method he --init_scales 0.7,1.8,0.8 \
	--gradient_clip 0.03 --gabor_features \
	--output_lr 0.15 --non_column_lr 0.15,0.15 --column_lr 0.00075,0.00045 \
	> "logs/exp3_full_${ts}.log" 2>&1

# --- 集計1: best/final/gap ---
summary_csv="tmp/protocol_summary_${ts}.csv"
echo "label,best_test,best_epoch,final_test,final_epoch,gap,log" > "$summary_csv"

for l in "logs/exp1_pure_ed_${ts}.log" "logs/exp2_baseline_clip003_${ts}.log" "logs/exp3_full_${ts}.log"; do
	label=$(basename "$l" .log)

	best_line=$(grep -E 'ベスト精度: Test=' "$l" | tail -n1 || true)
	best_test=$(echo "$best_line" | sed -E 's/.*Test=([0-9]+\.[0-9]+)%.*/\1/' || true)
	best_epoch=$(echo "$best_line" | sed -E 's/.*\(Epoch ([0-9]+)\).*/\1/' || true)

	final_line=$(grep -E '^[[:space:]]*[0-9]+[[:space:]]+[0-9]+\.[0-9]+%[[:space:]]+[0-9]+\.[0-9]+%' "$l" | tail -n1 || true)
	if [[ -n "$final_line" ]]; then
		final_epoch=$(echo "$final_line" | awk '{print $1}')
		final_test=$(echo "$final_line" | awk '{print $3}' | tr -d '%')
	else
		fline=$(grep -E '最終精度: Test=' "$l" | tail -n1 || true)
		final_test=$(echo "$fline" | sed -E 's/.*Test=([0-9]+\.[0-9]+)%.*/\1/' || true)
		final_epoch=""
	fi

	gap=$(awk -v b="$best_test" -v f="$final_test" 'BEGIN{if(b==""||f==""){print ""}else{printf "%.4f", b-f}}')
	echo "$label,$best_test,$best_epoch,$final_test,$final_epoch,$gap,$l" >> "$summary_csv"
done

# --- 集計2: クラス別（best/final） ---
class_csv="tmp/protocol_class_compare_${ts}.csv"
echo "label,best_epoch,best_C2,best_C6,best_C7,best_C9,best_avg,final_epoch,final_C2,final_C6,final_C7,final_C9,final_avg,log" > "$class_csv"

for l in "logs/exp1_pure_ed_${ts}.log" "logs/exp2_baseline_clip003_${ts}.log" "logs/exp3_full_${ts}.log"; do
	label=$(basename "$l" .log)

	be=$(grep -E 'ベスト精度: Test=' "$l" | sed -E 's/.*\(Epoch ([0-9]+)\).*/\1/' | tail -n1 || true)
	fe=$(grep -E '^[[:space:]]*[0-9]+[[:space:]]+[0-9]+\.[0-9]+%[[:space:]]+[0-9]+\.[0-9]+%' "$l" | tail -n1 | awk '{print $1}' || true)

	bline=$(grep -E "^[[:space:]]*${be}[[:space:]]+[0-9]+\.[0-9]+%" "$l" | tail -n1 || true)
	fline=$(grep -E "^[[:space:]]*${fe}[[:space:]]+[0-9]+\.[0-9]+%" "$l" | tail -n1 || true)

	bnorm=$(echo "$bline" | tr -d '%' | xargs)
	fnorm=$(echo "$fline" | tr -d '%' | xargs)

	bC2=$(echo "$bnorm" | awk '{print $4}')
	bC6=$(echo "$bnorm" | awk '{print $8}')
	bC7=$(echo "$bnorm" | awk '{print $9}')
	bC9=$(echo "$bnorm" | awk '{print $11}')
	bAvg=$(echo "$bnorm" | awk '{print $12}')

	fC2=$(echo "$fnorm" | awk '{print $4}')
	fC6=$(echo "$fnorm" | awk '{print $8}')
	fC7=$(echo "$fnorm" | awk '{print $9}')
	fC9=$(echo "$fnorm" | awk '{print $11}')
	fAvg=$(echo "$fnorm" | awk '{print $12}')

	echo "$label,$be,$bC2,$bC6,$bC7,$bC9,$bAvg,$fe,$fC2,$fC6,$fC7,$fC9,$fAvg,$l" >> "$class_csv"
done

echo "[DONE] SUMMARY=${summary_csv}" | tee -a "$master_log"
echo "[DONE] CLASS=${class_csv}" | tee -a "$master_log"
cat "$summary_csv"
cat "$class_csv"
```

注記:

- 本雛形は「最小3実験（Exp1-A / Exp2-A / Exp3-A）」を対象にしている。
- 対照群（Exp1-B, Exp2-B/C, Exp3-B/C/D）を追加する場合は、同形式で `[RUN]` ブロックと集計対象ログ配列を増やす。
- 実験中の進捗確認は `tail -n 40 logs/protocol_3exp_*.log` を使用する。

---

### 実施順（推奨）

1. 実験1（命題1）で学習信号の向きの妥当性を先に確認
2. 実験2（命題2）で更新則の安定動作を確認
3. 実験3（命題3）で構造化正則化の寄与を最終確認

この順に実施すると、因果の切り分け（信号設計→更新安定性→構造効果）が最も明確になる。

---

## 備考

本定式化は、実装コードに一致する操作を優先して記述した「実装整合型モデル」である。

今後は以下を加えることで、論文としての完成度を高められる。

1. 各命題に対する補題（単調性・有界性・固定点近傍挙動）
2. NC学習分岐（最近傍帰属・空間拡散）の統一表記
3. 収束観測量（best/final gap、クラス別寄与）と式の対応付け
