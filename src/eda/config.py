"""
config.py – Cấu hình cho module EDA (Step2)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional

DEFAULT_ROOT_OUTPUT: Path = Path("docs/output")
    
# Đường dẫn file Excel gốc
INPUT_CSV: Path = Path(r"D:\pyhton\FuelSentinel-AI\data\raw\CarFuelHistory.xlsx")

# Các cột tín hiệu
SIGNAL_COLUMNS: List[str] = ["fuel", "speed", "latitude", "longitude"]

# Các cột dùng cho heatmap
HEATMAP_COLS: List[str] = ["fuel", "speed", "latitude", "longitude"]

# Cặp scatter
SCATTER_PAIRS: List[Dict[str, str]] = [
    {"x": "fuel", "y": "speed"},
    {"x": "latitude", "y": "longitude"},
]

# Ngưỡng kiểm tra chất lượng
MAX_FUEL_LITERS: float = 500.0
MAX_SPEED_KMH: float = 200.0

# Danh sách các xe (sẽ được cập nhật tự động từ dữ liệu nếu không chỉ định)
CAR_IDS: Optional[List[str]] = None  # None -> lấy từ cột car_id trong dữ liệu


@dataclass(frozen=True)
class EDAConfig:
    """Cấu hình cho bước EDA."""
    output_dir: Path = DEFAULT_ROOT_OUTPUT / "eda"
    figures_dir: Path = field(init=False)
    summary_path: Path = field(init=False)

    SIGNAL_COLUMNS: List[str] = field(default_factory=lambda: SIGNAL_COLUMNS)
    HEATMAP_COLS: List[str] = field(default_factory=lambda: HEATMAP_COLS)

    FIGURE_DPI: int = 150
    HIST_BINS: int = 40
    ROLLING_WINDOW: int = 100
    PAIRPLOT_SAMPLE_SIZE: int = 3000

    def __post_init__(self) -> None:
        object.__setattr__(self, "figures_dir", self.output_dir / "figures")
        object.__setattr__(self, "summary_path", self.output_dir / "eda_summary.md")