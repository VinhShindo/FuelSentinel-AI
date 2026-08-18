#!/usr/bin/env python3
"""
benchmark.py - Entry point cho Phase 1: Model Benchmarking.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Tắt TẤT CẢ TensorFlow logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' 

try:
    import absl.logging
    absl.logging.set_verbosity('error')
    absl.logging.set_stderrthreshold('error')
except:
    pass

# Tắt oneDNN logs
import logging as tf_logging
tf_logging.getLogger('tensorflow').setLevel(tf_logging.ERROR)
tf_logging.getLogger('absl').setLevel(tf_logging.ERROR)

import argparse
import sys
import json
import logging
from pathlib import Path

# Thêm src vào path
sys.path.insert(0, str(Path(__file__).parent))

from trainning.utils.config import Config
from trainning.utils.seed import set_seed
from trainning.utils.device import get_device, print_device_info
from trainning.utils.logger import setup_logger
from trainning.datasets.dataset_builder import build_datasets
from trainning.models import all_models
from trainning.benchmark.runner import BenchmarkRunner
from trainning.benchmark.reporter import BenchmarkReporter

logger = logging.getLogger(__name__)


def run_benchmark(config: Config) -> str:
    """Chạy toàn bộ pipeline benchmark."""
    # Setup
    set_seed(config.seed, config.deterministic)
    device = get_device(config.get('device', 'auto'))
    print_device_info(device)
    
    # Build datasets
    logger.info("Building datasets...")
    train_ds, val_ds, test_ds = build_datasets(
        train_path=config.data.train_path,
        val_path=config.data.val_path,
        test_path=config.data.test_path,
        config={
            'min_sequence_length': config.get('data.min_sequence_length', 2),
            'fillna_strategy': config.get('data.fillna_strategy', 'median'),
        }
    )
    
    logger.info(f"Train samples: {len(train_ds)}")
    logger.info(f"Val samples: {len(val_ds)}")
    logger.info(f"Test samples: {len(test_ds)}")
    
    # Run benchmark
    runner = BenchmarkRunner(config, train_ds, val_ds, test_ds)
    results = runner.run()
    
    # Save results
    output_dir = Path(config.output.base_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runner.save_results(str(output_dir / 'results.json'))
    
    # Generate reports
    if any('error' not in r for r in results.values()):
        reporter = BenchmarkReporter(results, output_dir=str(output_dir))
        best_model = reporter.generate_all()
    else:
        logger.error("All models failed! Cannot generate reports.")
        return None
        
    # Save best model info
    with open(output_dir / 'best_model.json', 'w') as f:
        json.dump({
            'best_model': best_model,
            'config': config.to_dict(),
        }, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\nBenchmark complete! Best model: {best_model}")
    return best_model


def main():
    parser = argparse.ArgumentParser(description='FuelSentinel-AI Model Benchmark')
    parser.add_argument('--config', type=str, default='src/configs/benchmark.yaml')
    parser.add_argument('--models', type=str, nargs='+')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--device', type=str, choices=['auto', 'cuda', 'cpu'])
    
    args = parser.parse_args()
    
    config = Config.from_yaml(args.config)
    
    if args.models:
        config.set('models_to_benchmark', args.models)
    if args.epochs:
        config.set('training.epochs', args.epochs)
    if args.device:
        config.set('device', args.device)
    
    # Setup logger
    log_dir = Path(config.output.base_dir) / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    global logger
    logger = setup_logger(name='Benchmark', log_dir=str(log_dir))
    
    best_model = run_benchmark(config)
    print(f"\n✅ Benchmark completed! Best model: {best_model}")


if __name__ == '__main__':
    main()