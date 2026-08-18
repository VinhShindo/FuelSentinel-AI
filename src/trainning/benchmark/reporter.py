#!/usr/bin/env python3
"""
Module: reporter.py
Mục tiêu: Tạo báo cáo benchmark tổng hợp.
Export ra nhiều định dạng: JSON, CSV, Excel, Markdown, HTML.
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

from .comparator import BenchmarkComparator
from ..evaluation.benchmark_report import BenchmarkReport
from ..evaluation.confusion import ConfusionMatrixAnalyzer
from ..evaluation.roc_pr import ROCPRCurves

logger = logging.getLogger(__name__)


class BenchmarkReporter:
    """
    Tạo báo cáo benchmark hoàn chỉnh.
    
    Usage:
        reporter = BenchmarkReporter(results, output_dir='outputs/benchmark')
        reporter.generate_all()
    """
    
    def __init__(
        self,
        results: Dict[str, Dict],
        output_dir: str = 'outputs/benchmark',
    ):
        """
        Args:
            results: Dict từ BenchmarkRunner.
            output_dir: Thư mục output.
        """
        self.results = results
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.comparator = BenchmarkComparator(results)
        self.report = BenchmarkReport(
            self.comparator._extract_metrics_for_ranking(),
            output_dir=str(self.output_dir),
        )
    
    def generate_all(self) -> str:
        """
        Tạo tất cả báo cáo.
        
        Returns:
            Tên model tốt nhất.
        """
        logger.info("Generating benchmark reports...")
        
        # Tạo thư mục
        figures_dir = self.output_dir / 'figures'
        reports_dir = self.output_dir / 'reports'
        csv_dir = self.output_dir / 'csv'
        excel_dir = self.output_dir / 'excel'
        json_dir = self.output_dir / 'json'
        markdown_dir = self.output_dir / 'markdown'
        
        for d in [figures_dir, reports_dir, csv_dir, excel_dir, json_dir, markdown_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # 1. In comparison
        self.comparator.print_comparison()
        
        # 2. Vẽ tất cả biểu đồ so sánh
        self.comparator.plot_all(str(figures_dir))
        
        # 3. Benchmark report (radar, bubble, markdown)
        self.report.generate_all()
        
        # 4. Per-model evaluation
        for model_name in self.comparator.models:
            result = self.results[model_name]
            self._generate_per_model_report(model_name, result)
        
        # 5. Export data
        # CSV
        self.comparator.export_csv(str(csv_dir / 'benchmark_comparison.csv'))
        
        # Excel
        try:
            self.comparator.export_excel(str(excel_dir / 'benchmark_comparison.xlsx'))
        except Exception:
            logger.warning("Excel export failed")
        
        # JSON
        with open(json_dir / 'benchmark_results.json', 'w', encoding='utf-8') as f:
            serializable = {}
            for m, r in self.results.items():
                if 'error' in r:
                    serializable[m] = r
                else:
                    serializable[m] = {
                        'metrics': {
                            'overall': {
                                k: float(v) if isinstance(v, (np.floating, float)) else v
                                for k, v in r['metrics']['overall'].items()
                            }
                        },
                        'training_time_seconds': r['training_time_seconds'],
                        'inference_time_per_sample': r['inference_time_per_sample'],
                        'total_params': r['total_params'],
                        'model_size_mb': r['model_size_mb'],
                    }
            json.dump(serializable, f, indent=2, ensure_ascii=False)
        
        # Markdown
        markdown_content = self.report.generate_markdown_report(
            save_path=str(markdown_dir / 'benchmark_report.md')
        )
        
        # Best model
        best_model = self.comparator.get_best_model()
        logger.info(f"\n{'='*50}")
        logger.info(f"BEST MODEL: {best_model}")
        logger.info(f"{'='*50}")
        
        return best_model
    
    def _generate_per_model_report(
        self,
        model_name: str,
        result: Dict,
    ) -> None:
        """
        Tạo báo cáo riêng cho từng model.
        
        Args:
            model_name: Tên model.
            result: Kết quả benchmark.
        """
        model_dir = self.output_dir / 'models' / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Lưu metrics
        with open(model_dir / 'metrics.json', 'w', encoding='utf-8') as f:
            metrics_copy = result.get('metrics', {})
            json.dump(metrics_copy, f, indent=2, ensure_ascii=False, default=str)
        
        # Lưu history
        if 'history' in result:
            history_df = pd.DataFrame(result['history'])
            history_df.to_csv(model_dir / 'training_history.csv', index=False)
        
        logger.debug(f"Per-model report saved for {model_name}")