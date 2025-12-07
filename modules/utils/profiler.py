"""
ED-SNN性能分析・デバッグモジュール

学習時間のボトルネック特定とed_multi_snn.prompt.md準拠の最適化

作成者: ED-SNN開発チーム
作成日: 2025年9月28日
"""

import time
import psutil
import numpy as np
from typing import Dict, Any, List
from functools import wraps
import matplotlib.pyplot as plt

class EDSNNProfiler:
    """
    ED-SNN性能分析クラス
    
    各処理段階の実行時間とメモリ使用量を追跡
    """
    
    def __init__(self):
        self.timings = {}
        self.memory_usage = {}
        self.call_counts = {}
        self.cumulative_times = {}
        
    def profile_method(self, method_name: str):
        """メソッド実行時間測定デコレータ"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # 実行前
                start_time = time.time()
                start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                
                # 実行
                result = func(*args, **kwargs)
                
                # 実行後
                end_time = time.time()
                end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                
                # 統計更新
                execution_time = end_time - start_time
                memory_delta = end_memory - start_memory
                
                if method_name not in self.timings:
                    self.timings[method_name] = []
                    self.memory_usage[method_name] = []
                    self.call_counts[method_name] = 0
                    self.cumulative_times[method_name] = 0.0
                
                self.timings[method_name].append(execution_time)
                self.memory_usage[method_name].append(memory_delta)
                self.call_counts[method_name] += 1
                self.cumulative_times[method_name] += execution_time
                
                return result
            return wrapper
        return decorator
    
    def get_performance_report(self) -> str:
        """性能分析レポート生成"""
        report = "\n" + "="*70 + "\n"
        report += "           ED-SNN 性能分析レポート\n"
        report += "="*70 + "\n"
        
        # 時間順でソート
        sorted_methods = sorted(
            self.cumulative_times.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        report += f"{'メソッド名':<25} {'呼出回数':<8} {'累積時間':<12} {'平均時間':<12} {'メモリ':<10}\n"
        report += "-" * 70 + "\n"
        
        for method, total_time in sorted_methods:
            avg_time = total_time / self.call_counts[method]
            avg_memory = np.mean(self.memory_usage[method]) if method in self.memory_usage and self.memory_usage[method] else 0
            
            report += f"{method:<25} {self.call_counts[method]:<8} "
            report += f"{total_time:<12.4f}s {avg_time:<12.6f}s {avg_memory:<10.2f}MB\n"
        
        report += "="*70 + "\n"
        
        # ボトルネック特定
        if sorted_methods:
            bottleneck = sorted_methods[0]
            report += f"\n🔍 ボトルネック: {bottleneck[0]} (累積時間: {bottleneck[1]:.4f}s)\n"
            
            # 最適化提案
            report += self._get_optimization_suggestions(bottleneck[0])
        
        return report
    
    def _get_optimization_suggestions(self, bottleneck_method: str) -> str:
        """ボトルネックに応じた最適化提案"""
        suggestions = "\n💡 最適化提案:\n"
        
        if "snn_dynamics" in bottleneck_method.lower():
            suggestions += "  - LIFニューロンの並列計算最適化\n"
            suggestions += "  - スパイクパターンの効率的な行列演算\n"
            suggestions += "  - シミュレーション時間の短縮検討\n"
            
        elif "ed_learning" in bottleneck_method.lower() or "weight" in bottleneck_method.lower():
            suggestions += "  - ED法重み更新のNumPy最適化\n"
            suggestions += "  - 3D配列演算の効率化\n"
            suggestions += "  - アミン濃度計算の最適化\n"
            
        elif "encode" in bottleneck_method.lower():
            suggestions += "  - スパイクエンコーディングの前計算\n"
            suggestions += "  - バッチ処理の導入\n"
            suggestions += "  - エンコーディングタイプの最適化\n"
            
        elif "data" in bottleneck_method.lower():
            suggestions += "  - データローダーの並列処理\n"
            suggestions += "  - バッチサイズの最適化\n"
            suggestions += "  - メモリ効率的なデータ読み込み\n"
            
        else:
            suggestions += "  - 該当メソッドの詳細プロファイリング推奨\n"
        
        return suggestions
    
    def visualize_performance(self, save_path: str = "performance_analysis.png"):
        """性能分析結果の可視化"""
        if not self.cumulative_times:
            print("⚠️ プロファイリングデータがありません")
            return
            
        # 日本語フォント設定
        from modules.utils.font_config import ensure_japanese_font
        ensure_japanese_font()
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 累積実行時間
        methods = list(self.cumulative_times.keys())
        times = list(self.cumulative_times.values())
        
        ax1.bar(range(len(methods)), times, color='skyblue')
        ax1.set_title('メソッド別累積実行時間')
        ax1.set_xlabel('メソッド')
        ax1.set_ylabel('時間 (秒)')
        ax1.set_xticks(range(len(methods)))
        ax1.set_xticklabels([m[:15] + '...' if len(m) > 15 else m for m in methods], 
                           rotation=45, ha='right')
        
        # 2. 呼び出し回数
        counts = [self.call_counts[m] for m in methods]
        ax2.bar(range(len(methods)), counts, color='lightcoral')
        ax2.set_title('メソッド別呼び出し回数')
        ax2.set_xlabel('メソッド')
        ax2.set_ylabel('回数')
        ax2.set_xticks(range(len(methods)))
        ax2.set_xticklabels([m[:15] + '...' if len(m) > 15 else m for m in methods], 
                           rotation=45, ha='right')
        
        # 3. 平均実行時間
        avg_times = [times[i] / counts[i] for i in range(len(times))]
        ax3.bar(range(len(methods)), avg_times, color='lightgreen')
        ax3.set_title('メソッド別平均実行時間')
        ax3.set_xlabel('メソッド')
        ax3.set_ylabel('時間 (秒)')
        ax3.set_xticks(range(len(methods)))
        ax3.set_xticklabels([m[:15] + '...' if len(m) > 15 else m for m in methods], 
                           rotation=45, ha='right')
        
        # 4. 実行時間分布（上位5メソッド）
        if len(self.cumulative_times) >= 1:
            top_method = max(self.cumulative_times.keys(), key=lambda k: self.cumulative_times[k])
            if self.timings[top_method]:
                ax4.hist(self.timings[top_method], bins=20, alpha=0.7, color='orange')
                ax4.set_title(f'実行時間分布: {top_method[:20]}')
                ax4.set_xlabel('実行時間 (秒)')
                ax4.set_ylabel('頻度')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 性能分析グラフを保存: {save_path}")
        plt.close()
    
    def reset(self):
        """統計データリセット"""
        self.timings.clear()
        self.memory_usage.clear()
        self.call_counts.clear()
        self.cumulative_times.clear()


# グローバルプロファイラ
profiler = EDSNNProfiler()


def profile_function(name: str):
    """関数プロファイリング用デコレータ"""
    return profiler.profile_method(name)


class TimingContext:
    """with文で使用するタイミング測定コンテキスト"""
    
    def __init__(self, name: str, profiler_instance: EDSNNProfiler = None):
        self.name = name
        self.profiler = profiler_instance or profiler
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        execution_time = time.time() - self.start_time
        
        if self.name not in self.profiler.cumulative_times:
            self.profiler.cumulative_times[self.name] = 0.0
            self.profiler.call_counts[self.name] = 0
            self.profiler.timings[self.name] = []
            self.profiler.memory_usage[self.name] = []
            
        self.profiler.cumulative_times[self.name] += execution_time
        self.profiler.call_counts[self.name] += 1
        self.profiler.timings[self.name].append(execution_time)


def benchmark_ed_snn_components():
    """ED-SNNコンポーネント個別ベンチマーク"""
    print("🔍 ED-SNNコンポーネント個別性能測定")
    print("=" * 50)
    
    from modules.snn.snn_network import EDSpikingNeuralNetwork
    import numpy as np
    
    # 小規模ネットワークでテスト
    network = EDSpikingNeuralNetwork([28, 16, 4], simulation_time=10.0)
    
    # テストデータ
    test_input = np.random.rand(28) * 0.8
    test_target = np.array([0, 1, 0, 0])
    
    # 1. スパイクエンコーディング性能
    with TimingContext("spike_encoding"):
        for _ in range(100):
            network.encode_input_to_spikes(test_input)
    
    # 2. SNNダイナミクス性能
    spike_input = network.encode_input_to_spikes(test_input)
    with TimingContext("snn_dynamics"):
        for _ in range(10):
            network.simulate_snn_dynamics(spike_input)
    
    # 3. ED法学習性能
    with TimingContext("ed_learning"):
        for _ in range(10):
            network.train_step(test_input, test_target)
    
    print(profiler.get_performance_report())
    profiler.visualize_performance("images/component_benchmark.png")