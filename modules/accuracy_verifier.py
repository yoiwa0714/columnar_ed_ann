#!/usr/bin/env python3
"""
精度・誤差検証モジュール - クラス別精度・混同行列の詳細レポート

columnar_ed_ann.pyからAccuracyLossVerifierクラスを抽出したモジュール
- クラス別精度・損失の計算
- 混同行列の生成・表示
- 詳細な検証レポート出力

実装日: 2025-12-06
"""

import numpy as np


class AccuracyLossVerifier:
    """
    精度・誤差の詳細検証レポート
    
    機能:
    - 全体精度・損失の計算
    - クラス別精度・損失の計算
    - 混同行列の生成・表示
    """
    
    def __init__(self, network, class_names=None):
        """
        Parameters:
        -----------
        network : object
            検証対象のネットワーク (forwardメソッドを持つ)
        class_names : list[str] or None
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
            
            # 予測（forwardは3つの値を返す）
            _, z_output, _ = self.network.forward(x)
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
