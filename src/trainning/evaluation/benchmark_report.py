#!/usr/bin/env python3
"""
Module: benchmark_report.py
Mục tiêu: Tạo báo cáo benchmark so sánh nhiều model.
Bao gồm: ranking table, radar chart, bubble chart, export multi-format.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

logger = logging.getLogger(__name__)

COLORS = [
    '#2ecc71', '#3498db', '#e74c3c', '#f39c12',
    '#9b59b6', '#1abc9c', '#e67e22', '#34495e',
    '#e91e63', '#00bcd4', '#ff5722', '#607d8b',
]

class ModelRanking:
    """
    Xếp hạng model dựa trên nhiều tiêu chí.
    
    Usage:
        ranking = ModelRanking(results)
        ranking.compute_scores()
        ranking.print_table()
        ranking.export_csv('ranking.csv')
    """
    
    # Trọng số cho từng metric (tổng = 1.0)
    DEFAULT_WEIGHTS = {
        'accuracy': 0.20,
        'f1_weighted': 0.20,
        'balanced_accuracy': 0.15,
        'inference_speed': 0.15,
        'model_size_mb': 0.10,
        'params': 0.10,
        'training_time': 0.10,
    }
    
    def __init__(
        self,
        results: Dict[str, Dict],
        weights: Optional[Dict[str, float]] = None,
    ):
        """
        Args:
            results: Dict {model_name: {metric_name: value, ...}}
            weights: Trọng số cho từng metric.
        """
        self.results = results
        self.weights = weights or self.DEFAULT_WEIGHTS
        self.models = list(results.keys())
        self.scores = {}
        self.ranks = {}
        self.ranking_table = None
    
    def compute_scores(self) -> pd.DataFrame:
        """
        Tính điểm tổng hợp và xếp hạng.
        
        Returns:
            DataFrame ranking.
        """
        if not self.results:
            logger.warning("No results to rank.")
            return pd.DataFrame()
        
        # Tạo DataFrame từ results
        rows = []
        for model_name, metrics in self.results.items():
            row = {'Model': model_name}
            row.update(metrics)
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.set_index('Model', inplace=True)
        
        # Chuẩn hóa từng metric về [0, 1]
        normalized = pd.DataFrame(index=df.index)
        
        for col in df.columns:
            if col in self.weights:
                min_val = df[col].min()
                max_val = df[col].max()
                
                if max_val > min_val:
                    # Metrics cao hơn = tốt hơn (accuracy, f1...)
                    if col in ['accuracy', 'f1_weighted', 'balanced_accuracy', 'f1_macro']:
                        normalized[col] = (df[col] - min_val) / (max_val - min_val)
                    # Metrics thấp hơn = tốt hơn (loss, time, size...)
                    else:
                        normalized[col] = (max_val - df[col]) / (max_val - min_val)
                else:
                    normalized[col] = 1.0
        
        # Tính điểm tổng hợp
        total_score = pd.Series(0.0, index=normalized.index)
        for col, weight in self.weights.items():
            if col in normalized.columns:
                total_score += normalized[col] * weight
        
        # Chuẩn hóa về [0, 100]
        total_score = (total_score / total_score.max()) * 100
        
        # Tạo ranking
        df['Score'] = total_score.round(2)
        df = df.sort_values('Score', ascending=False)
        df['Rank'] = range(1, len(df) + 1)
        
        # Sắp xếp cột
        cols = ['Rank', 'Score'] + [c for c in df.columns if c not in ['Rank', 'Score']]
        df = df[cols]
        
        self.ranking_table = df
        return df
    
    def print_table(self) -> None:
        """In bảng xếp hạng."""
        if self.ranking_table is None:
            self.compute_scores()
        
        print("\n" + "=" * 100)
        print("MODEL RANKING")
        print("=" * 100)
        print(self.ranking_table.to_string())
        print("=" * 100)
        
        # Top 3
        top3 = self.ranking_table.head(3)
        print("\n🏆 Top 3 Models:")
        for i, (model, row) in enumerate(top3.iterrows(), 1):
            print(f"  {i}. {model} (Score: {row['Score']:.1f})")
    
    def get_best_model(self) -> str:
        """Trả về tên model tốt nhất."""
        if self.ranking_table is None:
            self.compute_scores()
        
        return self.ranking_table.index[0]
    
    def export_csv(self, path: str) -> None:
        """Xuất ra CSV."""
        if self.ranking_table is None:
            self.compute_scores()
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.ranking_table.to_csv(path)
        logger.info(f"Ranking exported to {path}")
    
    def export_json(self, path: str) -> None:
        """Xuất ra JSON."""
        if self.ranking_table is None:
            self.compute_scores()
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.ranking_table.to_json(path, orient='index', indent=2)
        logger.info(f"Ranking exported to {path}")
    
    def export_excel(self, path: str) -> None:
        """Xuất ra Excel."""
        if self.ranking_table is None:
            self.compute_scores()
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.ranking_table.to_excel(path, sheet_name='Model Ranking')
        logger.info(f"Ranking exported to {path}")


class BenchmarkReport:
    """
    Tạo báo cáo benchmark hoàn chỉnh.
    
    Usage:
        report = BenchmarkReport(results, output_dir='outputs/benchmark')
        report.generate()
    """
    
    def __init__(
        self,
        results: Dict[str, Dict],
        output_dir: str = 'outputs/benchmark',
    ):
        """
        Args:
            results: Dict {model_name: {metrics, training_time, inference_time, ...}}
            output_dir: Thư mục output.
        """
        self.results = results
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.ranking = ModelRanking(results)
        self.ranking.compute_scores()
    
    def generate_radar_chart(
        self,
        metrics: List[str] = None,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8),
    ) -> plt.Figure:
        """
        Vẽ radar chart so sánh các model.
        
        Args:
            metrics: Các metrics cần so sánh.
            save_path: Đường dẫn lưu ảnh.
            figsize: Kích thước figure.
            
        Returns:
            Matplotlib figure.
        """
        if metrics is None:
            metrics = ['accuracy', 'f1_weighted', 'balanced_accuracy', 'precision_weighted', 'recall_weighted']
        
        models = list(self.results.keys())
        n_metrics = len(metrics)
        
        # Tính góc cho từng trục
        angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
        angles += angles[:1]  # Đóng vòng
        
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
        
        for i, model in enumerate(models):
            values = [self.results[model].get(m, 0) for m in metrics]
            values += values[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, label=model, color=COLORS[i % len(COLORS)])
            ax.fill(angles, values, alpha=0.1)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics], fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_title('Model Comparison - Radar Chart', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
        ax.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def generate_bubble_chart(
        self,
        x_metric: str = 'model_size_mb',
        y_metric: str = 'accuracy',
        size_metric: str = 'f1_weighted',
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 8),
    ) -> plt.Figure:
        """
        Vẽ bubble chart (Accuracy vs Model Size).
        
        Args:
            x_metric: Metric cho trục X.
            y_metric: Metric cho trục Y.
            size_metric: Metric cho kích thước bubble.
            save_path: Đường dẫn lưu ảnh.
            figsize: Kích thước figure.
            
        Returns:
            Matplotlib figure.
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        models = list(self.results.keys())
        
        for i, model in enumerate(models):
            x = self.results[model].get(x_metric, 0)
            y = self.results[model].get(y_metric, 0)
            s = self.results[model].get(size_metric, 1) * 500
            
            ax.scatter(
                x, y, s=s,
                color=COLORS[i % len(COLORS)],
                alpha=0.6,
                edgecolors='black',
                linewidth=1,
                label=model,
            )
            ax.annotate(
                model,
                (x, y),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=9,
            )
        
        ax.set_xlabel(x_metric.replace('_', ' ').title(), fontsize=12)
        ax.set_ylabel(y_metric.replace('_', ' ').title(), fontsize=12)
        ax.set_title(
            f'{y_metric.replace("_", " ").title()} vs {x_metric.replace("_", " ").title()}',
            fontsize=14, fontweight='bold'
        )
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def generate_markdown_report(self, save_path: Optional[str] = None) -> str:
        """
        Tạo báo cáo Markdown.
        
        Args:
            save_path: Đường dẫn lưu file.
            
        Returns:
            Nội dung Markdown.
        """
        lines = []
        lines.append("# Benchmark Report - FuelSentinel-AI")
        lines.append("")
        lines.append(f"**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"**Number of models:** {len(self.results)}")
        lines.append("")
        
        # Model Ranking Table
        lines.append("## Model Ranking")
        lines.append("")
        
        if self.ranking.ranking_table is not None:
            df = self.ranking.ranking_table
            # Format
            lines.append("| Rank | Model | Score | Accuracy | F1 (Weighted) | Bal. Accuracy | Params | Size (MB) | Train Time (s) |")
            lines.append("|------|-------|-------|----------|---------------|---------------|--------|-----------|----------------|")
            
            for model, row in df.iterrows():
                lines.append(
                    f"| {int(row['Rank'])} | {model} | {row['Score']:.1f} | "
                    f"{row.get('accuracy', 'N/A'):.4f} | "
                    f"{row.get('f1_weighted', 'N/A'):.4f} | "
                    f"{row.get('balanced_accuracy', 'N/A'):.4f} | "
                    f"{row.get('params', 'N/A')} | "
                    f"{row.get('model_size_mb', 'N/A')} | "
                    f"{row.get('training_time', 'N/A')} |"
                )
        lines.append("")
        
        # Recommendation
        best_model = self.ranking.get_best_model()
        lines.append("## Recommendation")
        lines.append(f"**Best Model:** {best_model}")
        lines.append("")
        lines.append("This model was selected based on the weighted score considering:")
        for metric, weight in self.ranking.weights.items():
            lines.append(f"- {metric}: {weight:.0%}")
        lines.append("")
        
        # Per-model details
        lines.append("## Model Details")
        lines.append("")
        
        for model_name, metrics in self.results.items():
            lines.append(f"### {model_name}")
            lines.append("")
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            for k, v in metrics.items():
                if isinstance(v, float):
                    lines.append(f"| {k} | {v:.4f} |")
                else:
                    lines.append(f"| {k} | {v} |")
            lines.append("")
        
        report = "\n".join(lines)
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"Markdown report saved to {save_path}")
        
        return report
    
    def generate_all(self) -> None:
        """Tạo tất cả báo cáo và biểu đồ."""
        logger.info("Generating benchmark reports...")
        
        # Tạo thư mục
        figures_dir = self.output_dir / 'figures'
        reports_dir = self.output_dir / 'reports'
        
        # Ranking CSV
        self.ranking.export_csv(str(reports_dir / 'model_ranking.csv'))
        
        # Ranking JSON
        self.ranking.export_json(str(reports_dir / 'model_ranking.json'))
        
        # Ranking Excel
        try:
            self.ranking.export_excel(str(reports_dir / 'model_ranking.xlsx'))
        except Exception:
            logger.warning("Excel export failed (openpyxl may not be installed)")
        
        # Radar chart
        self.generate_radar_chart(save_path=str(figures_dir / 'radar_chart.png'))
        
        # Bubble chart
        self.generate_bubble_chart(save_path=str(figures_dir / 'bubble_chart.png'))
        
        # Markdown report
        self.generate_markdown_report(save_path=str(reports_dir / 'benchmark_report.md'))
        
        # In ranking
        self.ranking.print_table()
        
        logger.info(f"All reports generated in {self.output_dir}")


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    # Tạo dữ liệu giả
    results = {
        'BiLSTM': {
            'accuracy': 0.85,
            'f1_weighted': 0.84,
            'balanced_accuracy': 0.78,
            'precision_weighted': 0.86,
            'recall_weighted': 0.85,
            'params': 500000,
            'model_size_mb': 2.0,
            'training_time': 120.5,
            'inference_time': 0.015,
        },
        'GRU': {
            'accuracy': 0.83,
            'f1_weighted': 0.82,
            'balanced_accuracy': 0.76,
            'precision_weighted': 0.84,
            'recall_weighted': 0.83,
            'params': 380000,
            'model_size_mb': 1.5,
            'training_time': 100.3,
            'inference_time': 0.012,
        },
        'Transformer': {
            'accuracy': 0.87,
            'f1_weighted': 0.86,
            'balanced_accuracy': 0.80,
            'precision_weighted': 0.88,
            'recall_weighted': 0.87,
            'params': 1200000,
            'model_size_mb': 4.8,
            'training_time': 180.2,
            'inference_time': 0.025,
        },
        'TCN': {
            'accuracy': 0.82,
            'f1_weighted': 0.81,
            'balanced_accuracy': 0.75,
            'precision_weighted': 0.83,
            'recall_weighted': 0.82,
            'params': 350000,
            'model_size_mb': 1.4,
            'training_time': 95.8,
            'inference_time': 0.010,
        },
    }
    
    report = BenchmarkReport(results, output_dir='outputs/test/benchmark')
    report.generate_all()
    
    print("\nBenchmark report test passed!")