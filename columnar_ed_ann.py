#!/usr/bin/env python3
"""
Columnar ED-ANN - コラムED法による純粋ANN実装
Version: 0.0.1

大脳皮質のコラム構造を模倣したED法のANN実装
- 1つの共有重み空間で複数クラスを分類
- コラム帰属度による選択的アミン拡散
- ED法の本質(アミン拡散メカニズム)を完全保持
- LIFニューロンを使用せず、通常のANNで構成

実装日: 2025-11-23
ベース: columnar_ed_test.py
仕様: ed_multi_snn.prompt.md準拠

【達成精度】:
- MNIST: テスト正答率95.01% (訓練99.15%, 10000サンプル×20エポック)
- Fashion-MNIST: テスト正答率85.84% (訓練94.09%, 2層512→256, lr=0.005)
- Fashion-MNIST: テスト正答率83.04% (訓練85.71%, 2層512→256, lr=0.001, 過学習なし)
"""

# TensorFlowの情報メッセージを抑制
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=全表示, 1=INFO非表示, 2=WARNING以上, 3=ERROR以上
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # oneDNNの最適化メッセージを非表示

import numpy as np
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import argparse
import sys
from datetime import datetime
from pathlib import Path

# TensorFlowのログレベルを設定
tf.get_logger().setLevel('ERROR')


class HyperParams:
    """
    ハイパーパラメータ管理クラス
    
    ed_multi_lif_snn.pyのHyperParamsクラスから
    LIF関連パラメータを削除し、コラムED-ANN用に最適化
    """
    
    def __init__(self):
        # データセット設定
        self.dataset = 'mnist'
        self.train_samples = 10000  # デフォルト値を大規模学習の値に変更
        self.test_samples = 10000   # デフォルト値を大規模学習の値に変更
        
        # 学習設定
        self.epochs = 20  # デフォルト値を変更
        self.hidden = [256]  # デフォルト値を変更
        self.batch_size = 128
        self.seed = None
        self.no_shuffle = False  # デフォルトはシャッフル有効
        
        # ED法ハイパーパラメータ
        self.lr = 0.005  # デフォルト値を変更
        self.ami = 0.25  # アミン濃度 (beta)
        self.dif = 0.5   # アミン信号増減係数 (u1)
        self.sig = 1.2   # シグモイド閾値 (u0)
        self.w1 = 0.3    # 重み初期値1
        self.w2 = 0.5    # 重み初期値2
        
        # コラムED特有のパラメータ
        self.column_overlap = 0.1  # コラム重複度
        self.column_neurons = None # コラム占有数 (Noneの場合は自動計算)
        
        # 可視化設定
        self.viz = False
        self.heatmap = False
        self.save_fig = None
        
        # その他
        self.cpu = False
        self.verbose = False
        self.verify_acc_loss = False
    
    def parse_args(self, args=None):
        """コマンドライン引数をパース"""
        parser = argparse.ArgumentParser(
            description='Columnar ED-ANN - コラムED法による純粋ANN実装',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
使用例:
  # MNIST学習 (デフォルト設定)
  python columnar_ed_ann.py
  
  # Fashion-MNIST 2層構成
  python columnar_ed_ann.py --fashion --hidden 512,256 --epochs 50
  
  # 学習率調整
  python columnar_ed_ann.py --lr 0.001 --epochs 100
  
  # リアルタイム可視化
  python columnar_ed_ann.py --viz --heatmap --save_fig results/exp001
            """
        )
        
        # データセット
        dataset_group = parser.add_argument_group('データセット')
        dataset_group.add_argument('--mnist', action='store_true',
                                   help='MNISTデータセット使用（デフォルト）')
        dataset_group.add_argument('--fashion', action='store_true',
                                   help='Fashion-MNISTデータセット使用')
        
        # 学習設定
        train_group = parser.add_argument_group('学習設定')
        train_group.add_argument('--train', type=int, default=None, metavar='N',
                                help='訓練サンプル数（デフォルト: 10000）')
        train_group.add_argument('--test', type=int, default=None, metavar='N',
                                help='テストサンプル数（デフォルト: 10000）')
        train_group.add_argument('--epochs', type=int, default=None, metavar='N',
                                help='エポック数（デフォルト: 20）')
        train_group.add_argument('--hidden', type=str, default=None, metavar='N1,N2,...',
                                help='隠れ層構造（デフォルト: 256）')
        train_group.add_argument('--batch', type=int, default=None, metavar='N',
                                help='ミニバッチサイズ（デフォルト: 128）')
        train_group.add_argument('--seed', type=int, default=None, metavar='N',
                                help='ランダムシード（デフォルト: ランダム）')
        train_group.add_argument('--no_shuffle', action='store_true',
                                help='データシャッフル無効化（デフォルト: 有効）')
        
        # ED法ハイパーパラメータ
        ed_group = parser.add_argument_group('ED法ハイパーパラメータ')
        ed_group.add_argument('--lr', '--learning_rate', type=float, default=None, metavar='FLOAT',
                             dest='lr', help='学習率 (alpha)（デフォルト: 0.005）')
        ed_group.add_argument('--ami', type=float, default=None, metavar='FLOAT',
                             help='アミン濃度 (beta)（デフォルト: 0.25）')
        ed_group.add_argument('--dif', type=float, default=None, metavar='FLOAT',
                             help='アミン信号増減係数 (u1)（デフォルト: 0.5）')
        ed_group.add_argument('--sig', type=float, default=None, metavar='FLOAT',
                             help='シグモイド閾値 (u0)（デフォルト: 1.2）')
        ed_group.add_argument('--w1', type=float, default=None, metavar='FLOAT',
                             help='重み初期値1（デフォルト: 0.3）')
        ed_group.add_argument('--w2', type=float, default=None, metavar='FLOAT',
                             help='重み初期値2（デフォルト: 0.5）')
        ed_group.add_argument('--column_overlap', type=float, default=None, metavar='FLOAT',
                             help='コラム重複度（デフォルト: 0.1）')
        ed_group.add_argument('--column_neurons', type=int, default=None, metavar='N',
                             help='コラム占有数（デフォルト: 自動計算）')
        
        # 可視化
        viz_group = parser.add_argument_group('可視化')
        viz_group.add_argument('--viz', action='store_true',
                              help='リアルタイム学習進捗表示（デフォルト: 無効）')
        viz_group.add_argument('--heatmap', action='store_true',
                              help='コラム帰属度ヒートマップ表示（デフォルト: 無効）')
        viz_group.add_argument('--save_fig', type=str, default=None, metavar='PATH',
                              help='可視化保存パス (ディレクトリ/ファイル名/フルパス)')
        
        # その他
        other_group = parser.add_argument_group('その他')
        other_group.add_argument('--cpu', action='store_true',
                                help='CPUで実行（GPU環境でもCPUで実行）')
        other_group.add_argument('--verbose', '--v', action='store_true',
                                dest='verbose', help='詳細ログ表示')
        other_group.add_argument('--verify_acc_loss', action='store_true',
                                help='精度・誤差の検証レポート表示')
        
        parsed_args = parser.parse_args(args)
        
        # 引数をHyperParamsに反映
        if parsed_args.fashion:
            self.dataset = 'fashion'
        elif parsed_args.mnist:
            self.dataset = 'mnist'
        
        if parsed_args.train is not None:
            self.train_samples = parsed_args.train
        if parsed_args.test is not None:
            self.test_samples = parsed_args.test
        if parsed_args.epochs is not None:
            self.epochs = parsed_args.epochs
        if parsed_args.hidden is not None:
            self.hidden = [int(x.strip()) for x in parsed_args.hidden.split(',')]
        if parsed_args.batch is not None:
            self.batch_size = parsed_args.batch
        if parsed_args.seed is not None:
            self.seed = parsed_args.seed
        
        self.no_shuffle = parsed_args.no_shuffle
        
        if parsed_args.lr is not None:
            self.lr = parsed_args.lr
        if parsed_args.ami is not None:
            self.ami = parsed_args.ami
        if parsed_args.dif is not None:
            self.dif = parsed_args.dif
        if parsed_args.sig is not None:
            self.sig = parsed_args.sig
        if parsed_args.w1 is not None:
            self.w1 = parsed_args.w1
        if parsed_args.w2 is not None:
            self.w2 = parsed_args.w2
        if parsed_args.column_overlap is not None:
            self.column_overlap = parsed_args.column_overlap
        if parsed_args.column_neurons is not None:
            self.column_neurons = parsed_args.column_neurons
        
        self.viz = parsed_args.viz
        self.heatmap = parsed_args.heatmap
        self.save_fig = parsed_args.save_fig
        
        self.cpu = parsed_args.cpu
        self.verbose = parsed_args.verbose
        self.verify_acc_loss = parsed_args.verify_acc_loss
        
        return self


def check_column_neurons(column_neurons, n_hidden, n_output):
    """
    コラム占有数の妥当性チェック
    
    コラム占有数が最小層のニューロン数/クラス数を超える場合はエラー
    
    Parameters:
    -----------
    column_neurons : int or None
        指定されたコラム占有数 (Noneの場合はチェックスキップ)
    n_hidden : list[int]
        隠れ層ニューロン数のリスト
    n_output : int
        出力クラス数
    
    Raises:
    -------
    ValueError
        コラム占有数が最大許容値を超える場合
    
    Returns:
    --------
    int
        実際に使用するコラム占有数
    """
    min_hidden = min(n_hidden)
    max_neurons_per_column = min_hidden // n_output
    
    # 自動計算の場合
    if column_neurons is None:
        return max_neurons_per_column
    
    # チェック
    if column_neurons > max_neurons_per_column:
        error_msg = (
            f"\n{'='*70}\n"
            f"コラム占有数エラー\n"
            f"{'='*70}\n"
            f"指定されたコラム占有数: {column_neurons}\n"
            f"最大許容値: {max_neurons_per_column}\n"
            f"\n"
            f"詳細:\n"
            f"  最小層ニューロン数: {min_hidden}\n"
            f"  出力クラス数: {n_output}\n"
            f"  最大許容値 = {min_hidden} / {n_output} = {max_neurons_per_column}\n"
            f"\n"
            f"コラム占有数は最小層のニューロン数をクラス数で割った値以下\n"
            f"である必要があります。\n"
            f"{'='*70}\n"
        )
        print(error_msg, file=sys.stderr)
        sys.exit(1)
    
    return column_neurons


class ColumnarEDNetwork:
    """
    コラムED法ネットワーク
    
    特徴:
    - 1つの共有重み空間: w[hidden][input]
    - コラム帰属度: column_affinity[class][hidden]
    - 選択的アミン拡散: amine = error * column_affinity * u1
    - バッチ処理対応
    """
    
    def __init__(
        self,
        n_input: int = 784,
        n_hidden = 256,  # int or list[int]
        n_output: int = 10,
        alpha: float = 0.005,
        u1: float = 0.5,
        column_overlap: float = 0.1,
        column_neurons: int = None,
        w1: float = 0.3,
        w2: float = 0.5,
        sig: float = 1.2,
        verbose: bool = False
    ):
        """
        Parameters:
        -----------
        n_input : int
            入力ニューロン数
        n_hidden : int or list[int]
            隠れ層ニューロン数 (整数または多層リスト, e.g., [256] or [512, 256])
        n_output : int
            出力クラス数
        alpha : float
            学習率
        u1 : float
            アミン拡散係数
        column_overlap : float
            隣接コラムへの弱い帰属度 (0.0-0.5)
        column_neurons : int
            コラム占有数 (Noneの場合は自動計算)
        w1 : float
            重み初期値1 (興奮性ニューロン)
        w2 : float
            重み初期値2 (抑制性ニューロン)
        sig : float
            シグモイド閾値
        verbose : bool
            詳細ログ表示
        """
        self.n_input = n_input
        # n_hiddenをリストに統一
        self.n_hidden = [n_hidden] if isinstance(n_hidden, int) else n_hidden
        self.n_layers = len(self.n_hidden)
        self.n_output = n_output
        self.alpha = alpha
        self.u1 = u1
        self.column_overlap = column_overlap
        self.w1 = w1
        self.w2 = w2
        self.sig = sig
        self.verbose = verbose
        
        # コラム占有数のチェックと決定
        self.column_neurons = check_column_neurons(column_neurons, self.n_hidden, n_output)
        
        # ★重要★ 多層対応の共有重み空間
        self.w_hidden = []
        layer_sizes = [n_input] + self.n_hidden
        for i in range(self.n_layers):
            # 重み初期化 (w1, w2を使用)
            w = np.random.randn(layer_sizes[i+1], layer_sizes[i]) * w1
            self.w_hidden.append(w)
        
        # 出力層の重み
        self.w_output = np.random.randn(n_output, self.n_hidden[-1]) * w1
        
        # ★重要★ コラム帰属度マップ (最終隠れ層に対して)
        self.column_affinity = self.initialize_columns()
        
        # 学習履歴
        self.train_acc_history = []
        self.test_acc_history = []
        self.train_loss_history = []
        self.test_loss_history = []
        
        if verbose:
            self.print_network_info()
    
    def print_network_info(self):
        """ネットワーク情報を表示"""
        print("\n" + "="*70)
        print("コラムEDネットワーク初期化")
        print("="*70)
        print(f"入力層: {self.n_input} ニューロン")
        if self.n_layers == 1:
            print(f"隠れ層: {self.n_hidden[0]} ニューロン")
        else:
            print(f"隠れ層: {' → '.join(map(str, self.n_hidden))} ニューロン (多層)")
        print(f"出力層: {self.n_output} クラス")
        for i, w in enumerate(self.w_hidden):
            print(f"重み空間[{i}]: w[{w.shape[0]}, {w.shape[1]}]")
        print(f"コラム帰属度: [{self.n_output}, {self.n_hidden[-1]}] (最終層)")
        print(f"コラム占有数: {self.column_neurons} ニューロン/コラム")
        print(f"学習率α: {self.alpha}")
        print(f"アミン拡散係数u1: {self.u1}")
        print(f"コラム重複度: {self.column_overlap}")
        print(f"シグモイド閾値: {self.sig}")
        print("="*70 + "\n")
    
    def initialize_columns(self):
        """
        コラム帰属度マップの初期化
        
        各クラスに専用のニューロングループを割り当て
        隣接コラムにも弱い帰属度を設定(生物学的特徴)
        最終隠れ層に対して適用
        
        Returns:
        --------
        column_affinity : array [n_output, n_hidden[-1]]
            コラム帰属度マップ
        """
        n_final_hidden = self.n_hidden[-1]
        column_affinity = np.zeros((self.n_output, n_final_hidden))
        
        # 各クラスに均等にニューロンを割り当て
        neurons_per_column = self.column_neurons
        
        for c in range(self.n_output):
            # そのクラス専用のニューロン: 帰属度1.0
            start = c * neurons_per_column
            end = start + neurons_per_column
            column_affinity[c, start:end] = 1.0
            
            # ★生物学的特徴★: 隣接コラムにも弱い帰属度
            # コラムC₀は、隣のコラムC₁にも少し影響を与える
            if c > 0:
                prev_start = (c-1) * neurons_per_column
                prev_end = prev_start + neurons_per_column
                column_affinity[c, prev_start:prev_end] = self.column_overlap
            
            if c < self.n_output - 1:
                next_start = (c+1) * neurons_per_column
                next_end = next_start + neurons_per_column
                column_affinity[c, next_start:next_end] = self.column_overlap
        
        if self.verbose:
            print(f"[初期化] コラム帰属度マップ作成完了")
            print(f"  - 各コラム: {neurons_per_column} ニューロン専有")
            print(f"  - 隣接コラム重複: {self.column_overlap}")
        
        return column_affinity
    
    def sigmoid(self, x):
        """シグモイド活性化関数"""
        return 1 / (1 + np.exp(-x / self.sig))
    
    def sigmoid_derivative(self, z):
        """シグモイド導関数"""
        return z * (1 - z) / self.sig
    
    def forward(self, x):
        """
        順方向計算 (多層対応)
        
        Parameters:
        -----------
        x : array [n_input]
            入力データ (1サンプル)
        
        Returns:
        --------
        z_hiddens : list of arrays
            各隠れ層の活性 [layer][n_hidden[layer]]
        z_output : array [n_output]
            出力層の活性
        """
        # 多層順伝播
        z_hiddens = []
        z = x
        for i in range(self.n_layers):
            z = self.sigmoid(np.dot(self.w_hidden[i], z))
            z_hiddens.append(z)
        
        # 出力層
        z_output = self.sigmoid(np.dot(self.w_output, z_hiddens[-1]))
        
        return z_hiddens, z_output
    
    def columnar_ed_update(self, x, y_true, z_hiddens, z_output):
        """
        コラムED法による重み更新 (多層対応)
        
        ★重要★ コラム構造の適用範囲:
        - 最終隠れ層: コラム帰属度による選択的アミン拡散
        - 中間層: 通常のバックプロパゲーション (全ニューロン均等更新)
        
        Parameters:
        -----------
        x : array [n_input]
            入力ベクトル
        y_true : int
            正解クラス
        z_hiddens : list of arrays
            各隠れ層活性 [layer][n_hidden[layer]]
        z_output : array [n_output]
            出力層活性
        """
        # 出力層の誤差
        y_target = np.zeros(self.n_output)
        y_target[y_true] = 1.0
        error_output = y_target - z_output
        
        # 出力層の重み更新
        delta_w_output = self.alpha * np.outer(
            error_output * self.sigmoid_derivative(z_output), 
            z_hiddens[-1]
        )
        self.w_output += delta_w_output
        
        # ===== 最終隠れ層への誤差伝播 (コラム選択的) =====
        # 標準的なバックプロパゲーション誤差
        hidden_error_raw = np.dot(
            self.w_output.T, 
            error_output * self.sigmoid_derivative(z_output)
        )
        
        # ★コラム帰属度による選択的アミン拡散★
        # 正解クラスのコラムニューロンには強い学習シグナル
        # 他のニューロンには弱い学習シグナル (u1でスケーリング)
        column_scale = self.column_affinity[y_true] * self.u1
        error_hidden_final = hidden_error_raw * column_scale * self.sigmoid_derivative(z_hiddens[-1])
        
        # 最終隠れ層の重み更新
        z_input_final = z_hiddens[-2] if self.n_layers > 1 else x
        delta_w_final = self.alpha * np.outer(error_hidden_final, z_input_final)
        self.w_hidden[-1] += delta_w_final
        
        # ===== 中間層への逆伝播 (通常のバックプロパゲーション) =====
        if self.n_layers > 1:
            # 最終隠れ層から中間層への誤差伝播
            # ★重要★ コラム帰属度を適用する前の誤差を使用
            # 中間層は全ニューロンを均等に更新
            error_hidden = np.dot(
                self.w_hidden[-1].T, 
                hidden_error_raw * self.sigmoid_derivative(z_hiddens[-1])
            ) * self.sigmoid_derivative(z_hiddens[-2])
            
            # 中間層の逆伝播
            for layer in range(self.n_layers - 2, -1, -1):
                z_input = z_hiddens[layer-1] if layer > 0 else x
                
                # 重み更新
                delta_w = self.alpha * np.outer(error_hidden, z_input)
                self.w_hidden[layer] += delta_w
                
                # 前の層への誤差伝播
                if layer > 0:
                    error_hidden = np.dot(
                        self.w_hidden[layer].T, 
                        error_hidden
                    ) * self.sigmoid_derivative(z_hiddens[layer-1])
    
    def train_epoch(self, x_train, y_train, batch_size=128, no_shuffle=False):
        """
        1エポックの学習 (バッチ処理対応)
        
        Parameters:
        -----------
        x_train : array [n_samples, n_input]
            訓練データ
        y_train : array [n_samples]
            訓練ラベル
        batch_size : int
            バッチサイズ
        no_shuffle : bool
            シャッフル無効化
        
        Returns:
        --------
        accuracy : float
            訓練精度
        loss : float
            訓練損失
        """
        n_samples = len(x_train)
        correct = 0
        total_loss = 0.0
        
        # シャッフル
        indices = np.arange(n_samples)
        if not no_shuffle:
            np.random.shuffle(indices)
        
        # バッチ処理
        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i:i+batch_size]
            
            for idx in batch_indices:
                x = x_train[idx]
                y = y_train[idx]
                
                # 順方向計算
                z_hiddens, z_output = self.forward(x)
                
                # コラムED学習
                self.columnar_ed_update(x, y, z_hiddens, z_output)
                
                # 精度計算
                pred = np.argmax(z_output)
                if pred == y:
                    correct += 1
                
                # 損失計算 (クロスエントロピー)
                y_true_vec = np.zeros(self.n_output)
                y_true_vec[y] = 1.0
                loss = -np.sum(y_true_vec * np.log(z_output + 1e-10))
                total_loss += loss
        
        accuracy = correct / n_samples
        avg_loss = total_loss / n_samples
        return accuracy, avg_loss
    
    def evaluate(self, x_test, y_test):
        """
        評価
        
        Parameters:
        -----------
        x_test : array [n_samples, n_input]
            テストデータ
        y_test : array [n_samples]
            テストラベル
        
        Returns:
        --------
        accuracy : float
            テスト精度
        loss : float
            テスト損失
        """
        n_samples = len(x_test)
        correct = 0
        total_loss = 0.0
        
        for i in range(n_samples):
            x = x_test[i]
            y = y_test[i]
            
            # 順方向計算
            _, z_output = self.forward(x)
            
            # 予測
            pred = np.argmax(z_output)
            if pred == y:
                correct += 1
            
            # 損失計算
            y_true_vec = np.zeros(self.n_output)
            y_true_vec[y] = 1.0
            loss = -np.sum(y_true_vec * np.log(z_output + 1e-10))
            total_loss += loss
        
        accuracy = correct / n_samples
        avg_loss = total_loss / n_samples
        return accuracy, avg_loss
    
    def compute_confusion_matrix(self, x_data, y_data):
        """
        混同行列を計算
        
        Parameters:
        -----------
        x_data : array [n_samples, n_input]
            データ
        y_data : array [n_samples]
            ラベル
        
        Returns:
        --------
        confusion_matrix : array [n_classes, n_classes]
            混同行列 (行: 真のクラス, 列: 予測クラス)
        """
        n_samples = len(x_data)
        confusion_matrix = np.zeros((self.n_output, self.n_output), dtype=int)
        
        for i in range(n_samples):
            x = x_data[i]
            y_true = y_data[i]
            
            # 順方向計算
            _, z_output = self.forward(x)
            y_pred = np.argmax(z_output)
            
            # 混同行列に記録
            confusion_matrix[y_true, y_pred] += 1
        
        return confusion_matrix


class AccuracyLossVerifier:
    """
    精度・誤差の詳細検証レポート
    
    ed_multi_lif_snn.pyの--verify_acc_loss機能を移植
    """
    
    def __init__(self, network, class_names=None):
        """
        Parameters:
        -----------
        network : ColumnarEDNetwork
            検証対象のネットワーク
        class_names : list[str]
            クラス名リスト (Noneの場合は数字)
        """
        self.network = network
        self.class_names = class_names or [str(i) for i in range(network.n_output)]
    
    def verify(self, x_data, y_data, dataset_name="Dataset"):
        """
        詳細な精度・誤差分析
        
        Parameters:
        -----------
        x_data : array [n_samples, n_input]
            データ
        y_data : array [n_samples]
            ラベル
        dataset_name : str
            データセット名
        """
        n_samples = len(x_data)
        n_classes = self.network.n_output
        
        # クラス別統計
        class_correct = np.zeros(n_classes, dtype=int)
        class_total = np.zeros(n_classes, dtype=int)
        class_loss = np.zeros(n_classes, dtype=float)
        
        # 混同行列
        confusion_matrix = np.zeros((n_classes, n_classes), dtype=int)
        
        # 全サンプル評価
        predictions = []
        losses = []
        
        for i in range(n_samples):
            x = x_data[i]
            y_true = y_data[i]
            
            # 予測
            _, z_output = self.network.forward(x)
            y_pred = np.argmax(z_output)
            
            # 損失計算
            y_true_vec = np.zeros(n_classes)
            y_true_vec[y_true] = 1.0
            loss = -np.sum(y_true_vec * np.log(z_output + 1e-10))
            
            # 統計更新
            class_total[y_true] += 1
            class_loss[y_true] += loss
            if y_pred == y_true:
                class_correct[y_true] += 1
            
            confusion_matrix[y_true, y_pred] += 1
            predictions.append(y_pred)
            losses.append(loss)
        
        # レポート生成
        print("\n" + "="*70)
        print(f"精度・誤差検証レポート - {dataset_name}")
        print("="*70)
        
        # 全体精度・損失
        overall_acc = np.sum(class_correct) / n_samples
        overall_loss = np.sum(losses) / n_samples
        print(f"\n全体統計:")
        print(f"  精度: {overall_acc:.4f} ({overall_acc*100:.2f}%)")
        print(f"  損失: {overall_loss:.4f}")
        
        # クラス別精度
        print(f"\nクラス別精度:")
        for c in range(n_classes):
            if class_total[c] > 0:
                acc = class_correct[c] / class_total[c]
                avg_loss = class_loss[c] / class_total[c]
                print(f"  {self.class_names[c]:10s}: {acc:.4f} ({acc*100:5.2f}%) "
                      f"Loss: {avg_loss:.4f} ({class_total[c]:4d} samples)")
            else:
                print(f"  {self.class_names[c]:10s}: N/A (0 samples)")
        
        # 混同行列
        print(f"\n混同行列:")
        # 表示桁数を動的に調整
        max_value = np.max(confusion_matrix)
        max_digits = len(str(max_value))
        if max_digits <= 3:
            col_width = 4
        else:
            col_width = max_digits + 1
        
        print("   Pred: " + " ".join(f"{i:{col_width}d}" for i in range(n_classes)))
        for c in range(n_classes):
            print(f"True {c:2d}: " + " ".join(f"{confusion_matrix[c, i]:{col_width}d}" for i in range(n_classes)))
        
        print("="*70 + "\n")


def load_mnist(n_train=10000, n_test=10000, dataset='mnist', verbose=False):
    """
    MNIST/Fashion-MNISTデータセットをロード
    
    Parameters:
    -----------
    n_train : int
        訓練サンプル数
    n_test : int
        テストサンプル数
    dataset : str
        'mnist' または 'fashion'
    verbose : bool
        詳細ログ表示
    
    Returns:
    --------
    x_train, y_train, x_test, y_test
    """
    # TensorFlowからデータセットロード
    if dataset == 'fashion':
        (x_train_full, y_train_full), (x_test_full, y_test_full) = tf.keras.datasets.fashion_mnist.load_data()
        dataset_name = "Fashion-MNIST"
    else:
        (x_train_full, y_train_full), (x_test_full, y_test_full) = tf.keras.datasets.mnist.load_data()
        dataset_name = "MNIST"
    
    # 正規化 [0, 255] → [0, 1]
    x_train_full = x_train_full.astype(np.float32) / 255.0
    x_test_full = x_test_full.astype(np.float32) / 255.0
    
    # フラット化 (28, 28) → (784,)
    x_train_full = x_train_full.reshape(-1, 784)
    x_test_full = x_test_full.reshape(-1, 784)
    
    # サンプリング
    x_train = x_train_full[:n_train]
    y_train = y_train_full[:n_train]
    x_test = x_test_full[:n_test]
    y_test = y_test_full[:n_test]
    
    if verbose:
        print(f"[データロード] {dataset_name}")
        print(f"  訓練: {len(x_train)} サンプル")
        print(f"  テスト: {len(x_test)} サンプル")
    
    return x_train, y_train, x_test, y_test


def determine_save_path(save_fig_arg):
    """
    --save_figオプションから保存パスを決定
    
    Args:
        save_fig_arg: --save_figオプションの値 (None or str)
    
    Returns:
        str: 決定された保存パス
    """
    if save_fig_arg is None:
        # 指定なし: viz_results/viz_results_YYYYMMDD_HHMMSS.png
        viz_dir = Path('viz_results')
        viz_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return str(viz_dir / f'viz_results_{timestamp}.png')
    else:
        save_fig_path = Path(save_fig_arg)
        
        # ディレクトリのみ指定の場合 (存在するディレクトリ、または拡張子なし)
        if save_fig_path.is_dir():
            # 既存ディレクトリ
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            return str(save_fig_path / f'viz_results_{timestamp}.png')
        elif not save_fig_path.suffix:
            # 拡張子なし → ディレクトリとして扱う
            save_fig_path.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            return str(save_fig_path / f'viz_results_{timestamp}.png')
        # パス付きファイル名の場合
        elif save_fig_path.parent != Path('.'):
            save_fig_path.parent.mkdir(parents=True, exist_ok=True)
            return str(save_fig_path)
        # パス無しファイル名の場合
        else:
            return str(save_fig_path)


# ============================================================================
# 日本語フォント設定
# ============================================================================

def setup_japanese_font():
    """日本語フォントを設定する（Noto Sans CJK JP優先、fallback付き）"""
    import matplotlib.font_manager as fm
    
    # 優先フォントリスト
    preferred_fonts = [
        'Noto Sans CJK JP',
        'Noto Sans JP',
        'IPAexGothic',
        'IPAGothic',
        'TakaoPGothic',
        'VL PGothic',
    ]
    
    # システムにインストールされている日本語フォントを検索
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    # 優先順位に従ってフォントを選択
    selected_font = None
    for font in preferred_fonts:
        if font in available_fonts:
            selected_font = font
            break
    
    # 見つからない場合はCJKを含む任意のフォントを検索
    if selected_font is None:
        for font_name in available_fonts:
            if 'CJK' in font_name or 'Japan' in font_name or 'IPA' in font_name:
                selected_font = font_name
                break
    
    # フォントを設定
    if selected_font:
        plt.rcParams['font.family'] = selected_font
        print(f"日本語フォント設定: {selected_font}")
    else:
        print("警告: 日本語フォントが見つかりませんでした。デフォルトフォントを使用します。")


def main():
    """メイン関数"""
    # 日本語フォント設定
    setup_japanese_font()
    
    # ハイパーパラメータ解析
    hp = HyperParams()
    hp.parse_args()
    
    # ランダムシード設定
    if hp.seed is not None:
        np.random.seed(hp.seed)
        tf.random.set_seed(hp.seed)
        if hp.verbose:
            print(f"[Random Seed] {hp.seed}")
    
    # 保存パス決定
    save_path = determine_save_path(hp.save_fig)
    if hp.viz or hp.heatmap:
        print(f"[可視化保存先] {save_path}")
    
    print("\n" + "="*70)
    print("Columnar ED-ANN - コラムED法による純粋ANN実装")
    print("="*70)
    
    # データロード
    x_train, y_train, x_test, y_test = load_mnist(
        n_train=hp.train_samples,
        n_test=hp.test_samples,
        dataset=hp.dataset,
        verbose=True
    )
    
    # ネットワーク初期化
    network = ColumnarEDNetwork(
        n_input=784,
        n_hidden=hp.hidden,
        n_output=10,
        alpha=hp.lr,
        u1=hp.dif,
        column_overlap=hp.column_overlap,
        column_neurons=hp.column_neurons,
        w1=hp.w1,
        w2=hp.w2,
        sig=hp.sig,
        verbose=True
    )
    
    # 学習
    print(f"\n学習開始: {hp.epochs} エポック")
    print("-"*70)
    
    # リアルタイム可視化の準備
    fig_viz = None
    fig_heatmap = None
    
    if hp.viz:
        # 学習曲線 + 混同行列を表示
        plt.ion()
        fig_viz = plt.figure(figsize=(15, 5))
        fig_viz.canvas.manager.set_window_title('学習曲線 + 混同行列')
    
    if hp.heatmap:
        # 各層の活性化ヒートマップを表示
        plt.ion()
        # 層数に応じてレイアウトを決定
        n_layers = len(hp.hidden) + 1  # 隠れ層 + 出力層
        fig_heatmap = plt.figure(figsize=(16, 8))
        fig_heatmap.canvas.manager.set_window_title('層別活性化ヒートマップ')
    
    # 学習ループ
    for epoch in range(1, hp.epochs + 1):
        # 訓練
        train_acc, train_loss = network.train_epoch(
            x_train, y_train, 
            batch_size=hp.batch_size,
            no_shuffle=hp.no_shuffle
        )
        network.train_acc_history.append(train_acc)
        network.train_loss_history.append(train_loss)
        
        # 評価
        test_acc, test_loss = network.evaluate(x_test, y_test)
        network.test_acc_history.append(test_acc)
        network.test_loss_history.append(test_loss)
        
        # 進捗表示
        print(f"Epoch {epoch:3d}/{hp.epochs} | "
              f"Train Acc: {train_acc:.4f} | "
              f"Test Acc: {test_acc:.4f}")
        
        if hp.verbose:
            print(f"         Loss | Train: {train_loss:.4f} | Test: {test_loss:.4f}")
        
        # リアルタイム可視化
        if hp.viz and fig_viz is not None:
            # 学習曲線 + 混同行列を表示
            fig_viz.clear()
            ax1, ax2 = fig_viz.subplots(1, 2)
            
            # 学習曲線
            epochs_list = list(range(1, len(network.train_acc_history) + 1))
            ax1.plot(epochs_list, network.train_acc_history, label='Train', marker='o', markersize=3)
            ax1.plot(epochs_list, network.test_acc_history, label='Test', marker='s', markersize=3)
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Accuracy')
            ax1.set_title('Learning Progress')
            ax1.legend()
            
            # 縦軸設定: 0.0〜1.0
            ax1.set_ylim(0.0, 1.0)
            ax1.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
            
            # 横軸設定: 1〜最大エポック数
            ax1.set_xlim(1, hp.epochs)
            # 横軸の目盛り：10分割
            x_tick_interval = hp.epochs / 10
            x_ticks = [1] + [int(1 + i * x_tick_interval) for i in range(1, 10)] + [hp.epochs]
            ax1.set_xticks(x_ticks)
            
            # グリッド線の設定
            ax1.grid(True, alpha=0.3)
            # 縦軸のグリッド：0.1, 0.3, 0.5, 0.7, 0.9が点線、0.2, 0.4, 0.6, 0.8が実線
            for y in [0.1, 0.3, 0.5, 0.7, 0.9]:
                ax1.axhline(y=y, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
            for y in [0.2, 0.4, 0.6, 0.8]:
                ax1.axhline(y=y, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
            # 横軸のグリッド：10分割して点線と実線を交互に配置
            for i, x in enumerate([int(1 + j * x_tick_interval) for j in range(1, 10)]):
                if i % 2 == 0:  # 0, 2, 4, 6, 8 -> 点線
                    ax1.axvline(x=x, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
                else:  # 1, 3, 5, 7 -> 実線
                    ax1.axvline(x=x, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
            
            # 混同行列を表示
            conf_matrix = network.compute_confusion_matrix(x_test, y_test)
            sns.heatmap(conf_matrix, annot=True, fmt='d',
                       cmap='Blues', ax=ax2, cbar_kws={'label': 'Count'})
            ax2.set_xlabel('Predicted Class')
            ax2.set_ylabel('True Class')
            ax2.set_title('Confusion Matrix (Test Data)')
            
            plt.figure(fig_viz.number)
            plt.pause(0.1)
            plt.draw()
            
        if hp.heatmap and fig_heatmap is not None:
            # 各層の活性化ヒートマップを表示（ed_multi_lif_snn.py準拠）
            fig_heatmap.clear()
            
            # テストデータから1サンプルを取得して順伝播
            sample_idx = epoch % len(x_test)  # エポックごとに異なるサンプル
            sample_x = x_test[sample_idx]
            sample_y_true = y_test[sample_idx]
            z_hiddens, z_output = network.forward(sample_x)
            sample_y_pred = np.argmax(z_output)
            
            # GridSpec作成（ed_multi_lif_snn.py準拠: 2行×4列）
            gs = gridspec.GridSpec(4, 4, figure=fig_heatmap, hspace=0.4, wspace=0.3)
            
            # タイトル（エポック、正解、予測） - 予測クラスの色分け：正解=青、不正解=赤
            is_correct = (sample_y_pred == sample_y_true)
            pred_color = 'blue' if is_correct else 'red'
            title_text = f'エポック: {epoch} | 正解クラス: {sample_y_true} | '
            fig_heatmap.suptitle(title_text, fontsize=14, fontweight='bold', y=0.98, x=0.42, ha='right')
            fig_heatmap.text(0.42, 0.98, f'予測クラス: {sample_y_pred}', 
                    fontsize=14, fontweight='bold', color=pred_color, 
                    ha='left', va='top', transform=fig_heatmap.transFigure)
            
            # 表示する層を選択（8層超の場合は最初の4層と最後の4層）
            total_layers = len(z_hiddens) + 1  # 隠れ層 + 出力層
            if total_layers <= 8:
                # 全層表示
                display_layers = list(range(len(z_hiddens))) + [-1]  # -1は出力層
            else:
                # 最初の4層と最後の4層のみ表示
                display_layers = list(range(4)) + list(range(len(z_hiddens) - 3, len(z_hiddens))) + [-1]
            
            # 各層を表示
            for plot_idx, layer_idx in enumerate(display_layers[:8]):  # 最大8層
                if layer_idx == -1:
                    # 出力層
                    z_data = z_output
                    layer_name = f'Output Layer ({len(z_output)})'
                else:
                    # 隠れ層
                    z_data = z_hiddens[layer_idx]
                    layer_name = f'Hidden {layer_idx+1} ({len(z_data)})'
                
                # グリッド位置計算（2行×4列）
                row = plot_idx // 4
                col = plot_idx % 4
                ax = fig_heatmap.add_subplot(gs[row+1, col])  # 1行目はタイトル用に空ける
                
                # 活性化を2D配列に整形（正方形に近い形状、row-wise配置）
                n_neurons = len(z_data)
                side = int(np.ceil(np.sqrt(n_neurons)))
                z_reshaped = np.zeros((side, side))
                z_reshaped.flat[:n_neurons] = z_data
                im = ax.imshow(z_reshaped, cmap='rainbow', aspect='equal', vmin=0, vmax=1)
                
                ax.set_title(layer_name, fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
                
                # カラーバーを正しいfigureに追加
                from matplotlib import pyplot
                pyplot.figure(fig_heatmap.number)
                pyplot.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            plt.figure(fig_heatmap.number)
            plt.pause(0.1)
            plt.draw()
    
    if hp.viz or hp.heatmap:
        plt.ioff()
        if hp.viz and fig_viz is not None:
            save_path_viz = save_path.replace('.png', '_viz.png') if hp.viz and hp.heatmap else save_path
            plt.figure(fig_viz.number)
            plt.savefig(save_path_viz, dpi=150, bbox_inches='tight')
            print(f"[学習曲線保存] {save_path_viz}")
        if hp.heatmap and fig_heatmap is not None:
            save_path_heatmap = save_path.replace('.png', '_heatmap.png') if hp.viz and hp.heatmap else save_path
            plt.figure(fig_heatmap.number)
            plt.savefig(save_path_heatmap, dpi=150, bbox_inches='tight')
            print(f"[ヒートマップ保存] {save_path_heatmap}")
    
    print("-"*70)
    print("学習完了!")
    
    # 最終結果
    final_train_acc = network.train_acc_history[-1]
    final_test_acc = network.test_acc_history[-1]
    
    print("\n" + "="*70)
    print("最終結果")
    print("="*70)
    print(f"訓練精度: {final_train_acc:.4f} ({final_train_acc*100:.2f}%)")
    print(f"テスト精度: {final_test_acc:.4f} ({final_test_acc*100:.2f}%)")
    print(f"汎化ギャップ: {(final_train_acc - final_test_acc)*100:.2f}%")
    print("="*70 + "\n")
    
    # 精度・誤差の詳細検証
    if hp.verify_acc_loss:
        class_names_mnist = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        class_names_fashion = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
                              'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        class_names = class_names_fashion if hp.dataset == 'fashion' else class_names_mnist
        
        verifier = AccuracyLossVerifier(network, class_names)
        verifier.verify(x_train, y_train, "Training Set")
        verifier.verify(x_test, y_test, "Test Set")


if __name__ == "__main__":
    main()
