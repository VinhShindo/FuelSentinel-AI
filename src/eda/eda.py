"""
run_step2.py – Điểm chạy độc lập cho Bước 2: EDA.
Đọc tất cả sheet từ Excel, thêm CarID, gộp lại, sau đó chạy EDA.
Sử dụng: python src/pipeline/step2_eda/run_step2.py
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

if __name__ == "__main__" and __package__ is None:
    package_root = Path(__file__).resolve().parents[2]
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))

import pandas as pd

from src.eda.config import INPUT_CSV, EDAConfig, SIGNAL_COLUMNS, HEATMAP_COLS, CAR_IDS
from src.eda.utils import (
    get_logger, ensure_dir, describe_missing, compute_sampling_stats,
    describe_extended, describe_by_car, check_data_quality, generate_auto_comments
)
from src.eda.visualization import generate_all_plots

logger = get_logger(__name__)

def _convert_excel_to_csv(input_path: Path) -> Path:
    input_path = input_path.resolve()
    ext = input_path.suffix.lower()
    if ext == ".csv":
        return input_path
    if ext in (".xlsx", ".xls"):
        csv_path = input_path.with_suffix(".csv")
        logger.info("Đang đọc Excel: %s", input_path)
        xls = pd.ExcelFile(input_path)
        sheet_names = xls.sheet_names
        logger.info("Tìm thấy %d sheet: %s", len(sheet_names), sheet_names)

        all_dfs = []
        for sheet in sheet_names:
            df_sheet = pd.read_excel(input_path, sheet_name=sheet)
            df_sheet["car_id"] = sheet  # thêm cột car_id
            all_dfs.append(df_sheet)
            logger.info("Sheet '%s': %d dòng", sheet, len(df_sheet))

        df_all = pd.concat(all_dfs, ignore_index=True)
        logger.info("Tổng số dòng sau khi gộp: %d", len(df_all))
        df_all.to_csv(csv_path, index=False, encoding="utf-8")
        logger.info("Đã ghi CSV: %s", csv_path)
        return csv_path
    else:
        raise ValueError(f"Định dạng file không hỗ trợ: {ext}")


def load_data(csv_path: Path) -> pd.DataFrame:
    logger.info("Đọc CSV: %s", csv_path)
    df = pd.read_csv(csv_path)
    logger.info("Số dòng đọc từ CSV: %d", len(df))

    rename_map = {
        "FuelTime": "timestamp",
        "FuelLevel": "fuel",
        "Lat": "latitude",
        "Lng": "longitude",
        "Speed": "speed"
    }
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

    required = ["timestamp", "fuel", "latitude", "longitude", "speed"]
    missing_cols = set(required) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Thiếu các cột bắt buộc: {missing_cols}")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.sort_values(["car_id", "timestamp"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def build_summary_md(df: pd.DataFrame, figures: dict, output_dir: Path) -> str:
    n_rows, n_cols = df.shape
    time_min = df["timestamp"].min()
    time_max = df["timestamp"].max()

    car_info_df = describe_by_car(df)  # Thống kê theo xe
    lines = [
        "# EDA Summary",
        "",
        "## Dataset Overview",
        f"- Số dòng: {n_rows}",
        f"- Số cột: {n_cols}",
        f"- Số xe (car_id): {len(car_info_df)}",
        car_info_df[["car_id", "rows", "time_start", "time_end"]].to_markdown(index=False),
        f"- Thời gian bắt đầu chung: {time_min}",
        f"- Thời gian kết thúc chung: {time_max}",
        ""
    ]

    lines.append("## Data Sample")
    lines.append("### 5 dòng đầu")
    lines.append(df.head(5).to_markdown(index=False))
    lines.append("")
    lines.append("### 5 dòng cuối")
    lines.append(df.tail(5).to_markdown(index=False))
    lines.append("")

    # Missing
    missing_df = describe_missing(df)
    lines.append("## Missing Values")
    if missing_df.empty:
        lines.append("- Không có giá trị thiếu.")
    else:
        lines.append(missing_df.to_markdown())
    lines.append("")

    dup_count = df.duplicated().sum()
    lines.append("## Duplicate Rows")
    lines.append(f"- Số dòng trùng lặp: {dup_count}")
    lines.append("")

    # Sampling chung
    sampling_stats = compute_sampling_stats(df)
    if sampling_stats:
        lines.append("## Sampling Statistics (toàn bộ)")
        lines.append(f"- min: {sampling_stats['min_sec']:.2f}s")
        lines.append(f"- max: {sampling_stats['max_sec']:.2f}s")
        lines.append(f"- mean: {sampling_stats['mean_sec']:.2f}s")
        lines.append(f"- median: {sampling_stats['median_sec']:.2f}s")
        lines.append(f"- mode: {sampling_stats.get('mode_sec', 'N/A')}")
        lines.append("")

    # Thống kê mô tả chung
    desc_cols = ["fuel", "speed"]
    if all(c in df.columns for c in desc_cols):
        lines.append("## Descriptive Statistics (toàn bộ)")
        lines.append(describe_extended(df, desc_cols).to_markdown())
        lines.append("")

    # Thống kê theo xe (chi tiết)
    lines.append("## Thống kê theo từng xe")
    lines.append(car_info_df.to_markdown(index=False))
    lines.append("")

    # GPS stats
    if "latitude" in df.columns and "longitude" in df.columns:
        lines.append("## GPS Statistics (toàn bộ)")
        lines.append(f"- Latitude min: {df['latitude'].min():.6f}, max: {df['latitude'].max():.6f}")
        lines.append(f"- Longitude min: {df['longitude'].min():.6f}, max: {df['longitude'].max():.6f}")
        lines.append("")

    # Data Quality
    quality_issues = check_data_quality(df)
    lines.append("## Data Quality Checks")
    for key, info in quality_issues.items():
        icon = "⚠️" if info["count"] > 0 else "✅"
        lines.append(f"- {icon} {info['description']}: {info['count']} dòng ({info['percent']}%)")
    lines.append("")

    # Danh sách biểu đồ
    lines.append("## Danh sách biểu đồ đã sinh")
    for fname in sorted(figures.keys()):
        lines.append(f"- `figures/{fname}`")
    lines.append("")

    # Nhận xét tự động
    lines.append("## Automatic Comments")
    lines.append(generate_auto_comments(df, sampling_stats, quality_issues, car_info_df))

    return "\n".join(lines)


def run_step2(input_path: Path = None, output_dir: Path = None) -> dict:
    if input_path is None:
        input_path = INPUT_CSV
    if output_dir is None:
        cfg = EDAConfig()
        output_dir = cfg.output_dir
    else:
        cfg = EDAConfig(output_dir=output_dir)

    ensure_dir(cfg.output_dir)
    logger.info("===== EDA (nhiều xe) =====")

    csv_path = _convert_excel_to_csv(input_path)
    df = load_data(csv_path)

    # Ghi nhận các car_id
    car_ids = df["car_id"].unique()
    logger.info("Các xe trong dữ liệu: %s", car_ids)

    # Kiểm tra chất lượng
    quality = check_data_quality(df)
    logger.info("=== Kiểm tra chất lượng dữ liệu ===")
    for key, info in quality.items():
        if info["count"] > 0:
            logger.warning("%-40s : %5d dòng (%.2f%%)", info["description"] + ":", info["count"], info["percent"])

    figures = generate_all_plots(df, cfg.figures_dir)
    logger.info("Đã tạo %d biểu đồ.", len(figures))

    summary_content = build_summary_md(df, figures, cfg.output_dir)
    cfg.summary_path.write_text(summary_content, encoding="utf-8")
    logger.info("Đã lưu báo cáo: %s", cfg.summary_path)

    logger.info("===== HOÀN TẤT =====")
    return {"figures_dir": cfg.figures_dir, "summary": cfg.summary_path}


def main() -> None:
    try:
        run_step2()
    except FileNotFoundError as e:
        logger.error("Không tìm thấy file input (%s). Kiểm tra đường dẫn trong config.py", INPUT_CSV)
        logger.error(e)
        sys.exit(1)
    except Exception as e:
        logger.error("Thất bại: %s", e)
        logger.error(traceback.format_exc())
        sys.exit(2)


if __name__ == "__main__":
    main()