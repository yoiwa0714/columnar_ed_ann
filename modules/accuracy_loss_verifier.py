#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ED-SNN 精度・誤差検証モジュール v1.0
ed_multi_snn.prompt.md準拠・検証システム

目的:
1. 学習結果の精度・誤差を独立して再計算
2. 実装の信頼性を検証
3. デバッグ・開発時の詳細確認

使用方法:
    from modules.accuracy_loss_verifier import AccuracyLossVerifier
    
    verifier = AccuracyLossVerifier()
    verifier.verify_and_report(
        train_results_log=train_log,
        test_results_log=test_log,
        train_accuracies=train_acc,
        test_accuracies=test_acc,
        losses=train_loss,
        epochs=num_epochs
    )

特徴:
- ed_multi_snn.prompt.md準拠の検証ロジック
- 全エポックの詳細統計表示
- クラス別精度確認
- サンプル単位の詳細表示
"""

from typing import List, Dict, Any


class AccuracyLossVerifier:
    """
    精度・誤差の検証システム
    
    学習過程で記録された結果から、精度と誤差を独立して再計算し、
    現在の実装値と比較することで、計算の正確性を検証します。
    
    Attributes:
        None（ステートレス設計）
    """
    
    def __init__(self):
        """初期化（ステートレス設計のため何もしない）"""
        pass
    
    def verify_and_report(
        self,
        train_results_log: List[List[Dict[str, Any]]],
        test_results_log: List[List[Dict[str, Any]]],
        train_accuracies: List[float],
        test_accuracies: List[float],
        losses: List[float],
        epochs: int,
        show_sample_details: bool = True
    ) -> Dict[str, Any]:
        """
        精度・誤差を検証し、詳細レポートを表示
        
        Args:
            train_results_log: 訓練データの結果ログ（エポックごとのリスト）
            test_results_log: テストデータの結果ログ（エポックごとのリスト）
            train_accuracies: 記録された訓練正答率のリスト
            test_accuracies: 記録されたテスト正答率のリスト
            losses: 記録された訓練誤差のリスト
            epochs: エポック数
            show_sample_details: サンプル詳細表示のON/OFF
        
        Returns:
            検証結果の辞書（最終エポックの検証値を含む）
        """
        print("\n" + "="*80)
        print("🔍 精度・誤差計算の検証レポート（ed_multi_snn.prompt.md準拠）")
        print("="*80)
        
        # 最終エポックの検証値を保存
        final_verified_train_accuracy = 0.0
        final_verified_test_accuracy = 0.0
        
        for epoch_idx in range(epochs):
            # 訓練結果の検証
            train_log = train_results_log[epoch_idx]
            verified_train_correct = sum(
                1 for r in train_log if r['true_label'] == r['pred_label']
            )
            verified_train_accuracy = (verified_train_correct / len(train_log)) * 100
            verified_train_error = sum(r['error'] for r in train_log) / len(train_log)
            
            # テスト結果の検証
            test_log = test_results_log[epoch_idx]
            verified_test_correct = sum(
                1 for r in test_log if r['true_label'] == r['pred_label']
            )
            verified_test_accuracy = (verified_test_correct / len(test_log)) * 100
            verified_test_error = sum(r['error'] for r in test_log) / len(test_log)
            
            # 最終エポックの値を保存
            if epoch_idx == epochs - 1:
                final_verified_train_accuracy = verified_train_accuracy
                final_verified_test_accuracy = verified_test_accuracy
            
            # 現在の実装の値
            current_train_acc = train_accuracies[epoch_idx]
            current_test_acc = test_accuracies[epoch_idx]
            current_train_err = losses[epoch_idx]
            
            # エポックごとの検証結果表示
            print(f"\nエポック {epoch_idx + 1}:")
            print(f"  📊 訓練データ ({len(train_log)}サンプル):")
            print(f"    現在の実装: 精度={current_train_acc:.2f}%, 誤差={current_train_err:.4f}")
            print(f"    検証結果:   精度={verified_train_accuracy:.2f}%, 誤差={verified_train_error:.4f}")
            print(f"    ✅ 精度差: {abs(current_train_acc - verified_train_accuracy):.4f}%")
            print(f"    ✅ 誤差差: {abs(current_train_err - verified_train_error):.6f}")
            
            print(f"  📊 テストデータ ({len(test_log)}サンプル):")
            print(f"    現在の実装: 精度={current_test_acc:.2f}%")
            print(f"    検証結果:   精度={verified_test_accuracy:.2f}%")
            print(f"    ✅ 精度差: {abs(current_test_acc - verified_test_accuracy):.4f}%")
            
            # 詳細統計（最終エポックのみ）
            if show_sample_details and epoch_idx == epochs - 1:
                self._show_detailed_statistics(
                    train_log, test_log,
                    verified_train_correct, verified_test_correct
                )
        
        print("\n" + "="*80)
        
        # 検証結果を返す
        return {
            'final_train_accuracy': final_verified_train_accuracy,
            'final_test_accuracy': final_verified_test_accuracy
        }
    
    def _show_detailed_statistics(
        self,
        train_log: List[Dict[str, Any]],
        test_log: List[Dict[str, Any]],
        verified_train_correct: int,
        verified_test_correct: int
    ) -> None:
        """
        最終エポックの詳細統計を表示
        
        Args:
            train_log: 訓練データの結果ログ
            test_log: テストデータの結果ログ
            verified_train_correct: 訓練データの正解数
            verified_test_correct: テストデータの正解数
        """
        print(f"\n  📋 最終エポック詳細統計:")
        print(f"    訓練: 正解={verified_train_correct}/{len(train_log)}")
        print(f"    テスト: 正解={verified_test_correct}/{len(test_log)}")
        
        # 訓練データ先頭10サンプル
        print(f"\n  🔍 訓練データ先頭10サンプル:")
        for i, r in enumerate(train_log[:10]):
            match_mark = "✓" if r['true_label'] == r['pred_label'] else "✗"
            print(
                f"    [{i}] 正解={r['true_label']}, "
                f"予測={r['pred_label']}, "
                f"誤差={r['error']:.4f} {match_mark}"
            )
        
        # テストデータ先頭10サンプル
        print(f"\n  🔍 テストデータ先頭10サンプル:")
        for i, r in enumerate(test_log[:10]):
            match_mark = "✓" if r['true_label'] == r['pred_label'] else "✗"
            print(
                f"    [{i}] 正解={r['true_label']}, "
                f"予測={r['pred_label']}, "
                f"誤差={r['error']:.4f} {match_mark}"
            )


# 便利関数
def create_verifier() -> AccuracyLossVerifier:
    """
    AccuracyLossVerifierインスタンスを作成
    
    Returns:
        AccuracyLossVerifierインスタンス
    """
    return AccuracyLossVerifier()
