#!/usr/bin/env python3
"""
Module: comparator.py
Mục tiêu: So sánh kết quả benchmark giữa các model.
Tạo bảng so sánh, xếp hạng, và biểu đồ.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from ..evaluation.benchmark_report import ModelRanking
from ..evaluation.plots import plot_metrics_comparison, save_figure

logger = logging.getLogger(__name__)

COLORS = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22', '#34495e']


class BenchmarkComparator:
    """
    So sánh kết quả benchmark giữa các model.
    
    Usage:
        comparator = BenchmarkComparator(results)
        comparator.print_comparison()
        comparator.plot_all('outputs/figures')
    """
    
    def __init__(self, results: Dict[str, Dict]):
        """
        Args:
            results: Dict từ BenchmarkRunner.results
        """
        self.results = results
        self.models = [m for m, r in results.items() if 'error' not in r]
        self.error_models = [m for m, r in results.items() if 'error' in r]
        
        # Trích xuất metrics chính
        self.comparison_df = self._create_comparison_df()
        
        # Ranking
        self.ranking = ModelRanking(self._extract_metrics_for_ranking())
        self.ranking.compute_scores()
    
    def _create_comparison_df(self) -> pd.DataFrame:
        """Tạo DataFrame so sánh."""
        rows = []
        
        for model_name in self.models:
            result = self.results[model_name]
            metrics = result['metrics']['overall']
            
            row = {
                'Model': model_name,
                'Accuracy': metrics.get('accuracy', 0),
                'F1 (Weighted)': metrics.get('f1_weighted', 0),
                'F1 (Macro)': metrics.get('f1_macro', 0),
                'Balanced Acc': metrics.get('balanced_accuracy', 0),
                'Precision': metrics.get('precision_weighted', 0),
                'Recall': metrics.get('recall_weighted', 0),
                'ROC AUC': metrics.get('roc_auc_ovr', np.nan),
                'Cohen Kappa': metrics.get('cohen_kappa', 0),
                'MCC': metrics.get('mcc', 0),
                'Train Time (s)': result.get('training_time_seconds', 0),
                'Inference (ms)': result.get('inference_time_per_sample', 0) * 1000,
                'Params': result.get('total_params', 0),
                'Size (MB)': result.get('model_size_mb', 0),
                'Best Val Loss': result.get('best_val_loss', 0),
                'Best Val Acc': result.get('best_val_acc', 0),
                'Best Epoch': result.get('best_epoch', 0),
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        if len(df) > 0:
            df.set_index('Model', inplace=True)
        else:
            return pd.DataFrame()
        
        return df
    
    def _extract_metrics_for_ranking(self) -> Dict[str, Dict]:
        """Trích xuất metrics cho ModelRanking."""
        ranking_data = {}
        
        for model_name in self.models:
            result = self.results[model_name]
            metrics = result['metrics']['overall']
            
            ranking_data[model_name] = {
                'accuracy': metrics.get('accuracy', 0),
                'f1_weighted': metrics.get('f1_weighted', 0),
                'balanced_accuracy': metrics.get('balanced_accuracy', 0),
                'inference_speed': 1.0 / (result.get('inference_time_per_sample', 0.001) * 1000 + 1e-9),
                'model_size_mb': result.get('model_size_mb', 0),
                'params': result.get('total_params', 0),
                'training_time': result.get('training_time_seconds', 0),
            }
        
        return ranking_data
    
    def print_comparison(self) -> None:
        """In bảng so sánh."""
        print("\n" + "=" * 100)
        print("BENCHMARK COMPARISON")
        print("=" * 100)
        print(self.comparison_df.to_string())
        print("=" * 100)
        
        # In ranking
        self.ranking.print_table()
        
        # In model lỗi
        if self.error_models:
            print(f"\n⚠ Models with errors: {self.error_models}")
    
    def get_best_model(self) -> str:
        """Trả về model tốt nhất."""
        return self.ranking.get_best_model()
    
    def plot_accuracy_comparison(
        self,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Vẽ biểu đồ so sánh accuracy."""
        return plot_metrics_comparison(
            {m: {'accuracy': self.comparison_df.loc[m, 'Accuracy']}
             for m in self.models},
            metric_name='accuracy',
            save_path=save_path,
            title='Model Accuracy Comparison',
        )
    
    def plot_f1_comparison(
        self,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Vẽ biểu đồ so sánh F1."""
        return plot_metrics_comparison(
            {m: {'f1_weighted': self.comparison_df.loc[m, 'F1 (Weighted)']}
             for m in self.models},
            metric_name='f1_weighted',
            save_path=save_path,
            title='Model F1 Score Comparison',
        )
    
    def plot_training_time_comparison(
        self,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Vẽ biểu đồ so sánh thời gian training."""
        return plot_metrics_comparison(
            {m: {'training_time': self.comparison_df.loc[m, 'Train Time (s)']}
             for m in self.models},
            metric_name='training_time',
            save_path=save_path,
            title='Training Time Comparison',
        )
    
    def plot_params_comparison(
        self,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Vẽ biểu đồ so sánh số tham số."""
        return plot_metrics_comparison(
            {m: {'params': self.comparison_df.loc[m, 'Params']}
             for m in self.models},
            metric_name='params',
            save_path=save_path,
            title='Model Parameters Comparison',
        )
    
    def plot_accuracy_vs_size(
        self,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Vẽ Accuracy vs Model Size."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for i, model in enumerate(self.models):
            x = self.comparison_df.loc[model, 'Size (MB)']
            y = self.comparison_df.loc[model, 'Accuracy']
            s = self.comparison_df.loc[model, 'F1 (Weighted)'] * 300
            
            ax.scatter(
                x, y, s=s,
                color=COLORS[i % len(COLORS)],
                alpha=0.7,
                edgecolors='black',
                linewidth=1,
            )
            ax.annotate(model, (x, y), xytext=(5, 5),
                       textcoords='offset points', fontsize=10)
        
        ax.set_xlabel('Model Size (MB)', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Accuracy vs Model Size (Bubble = F1 Score)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            save_figure(fig, save_path)
        
        return fig
    
    def plot_all(self, output_dir: str) -> None:
        """
        Vẽ tất cả biểu đồ so sánh.
        
        Args:
            output_dir: Thư mục output.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.plot_accuracy_comparison(str(output_dir / 'accuracy_comparison.png'))
        self.plot_f1_comparison(str(output_dir / 'f1_comparison.png'))
        self.plot_training_time_comparison(str(output_dir / 'training_time.png'))
        self.plot_params_comparison(str(output_dir / 'params_comparison.png'))
        self.plot_accuracy_vs_size(str(output_dir / 'accuracy_vs_size.png'))
        
        logger.info(f"All comparison plots saved to {output_dir}")
    
    def export_csv(self, path: str) -> None:
        """Xuất bảng so sánh ra CSV."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.comparison_df.to_csv(path)
        logger.info(f"Comparison CSV saved to {path}")
    
    def export_excel(self, path: str) -> None:
        """Xuất bảng so sánh ra Excel."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(path, engine='openpyxl') as writer:
            self.comparison_df.to_excel(writer, sheet_name='Comparison')
            if self.ranking.ranking_table is not None:
                self.ranking.ranking_table.to_excel(writer, sheet_name='Ranking')
        logger.info(f"Comparison Excel saved to {path}")