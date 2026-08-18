import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Dữ liệu từ log huấn luyện cuối cùng (sau khi bổ sung Refuel synthetic)
data = {
    'Tập': ['Train', 'Train', 'Train', 'Train',
            'Validation', 'Validation', 'Validation', 'Validation',
            'Test', 'Test', 'Test', 'Test'],
    'Trạng thái': ['Driving', 'Idle', 'Refuel', 'Theft'] * 3,
    'Số segment': [3406, 2123, 1864, 1363,   # Train
                   632, 130, 142, 76,        # Validation
                   611, 135, 158, 85]        # Test
}
df = pd.DataFrame(data)

# Sắp xếp thứ tự tập
order = ['Train', 'Validation', 'Test']
df['Tập'] = pd.Categorical(df['Tập'], categories=order, ordered=True)

# ============================================================
# BÁO CÁO TỶ LỆ PHẦN TRĂM
# ============================================================
print("\n" + "="*60)
print("BÁO CÁO PHÂN BỐ TRẠNG THÁI THEO TẬP DỮ LIỆU (CUỐI CÙNG)")
print("="*60)

total_per_set = df.groupby('Tập')['Số segment'].sum()

for tap in order:
    subset = df[df['Tập'] == tap]
    total = total_per_set[tap]
    print(f"\nTập {tap} (tổng: {total} segment)")
    print("-" * 40)
    for _, row in subset.iterrows():
        percent = (row['Số segment'] / total) * 100
        print(f"  {row['Trạng thái']:<10} : {row['Số segment']:5d}  ({percent:5.1f}%)")
    print(f"  {'TỔNG':<10} : {total:5d}  (100.0%)")

print("\n" + "="*60)
# ============================================================

# Thiết lập style và vẽ biểu đồ
sns.set_style("whitegrid")
plt.figure(figsize=(10, 6))

ax = sns.barplot(
    data=df,
    x='Tập',
    y='Số segment',
    hue='Trạng thái',
    palette='viridis',
    edgecolor='black'
)

# Thêm nhãn giá trị lên đầu mỗi cột
for container in ax.containers:
    ax.bar_label(container, fmt='%d', label_type='edge', fontsize=8, padding=2)

plt.title('Phân bố 4 trạng thái theo từng tập dữ liệu (Dataset cuối cùng)', fontsize=14, fontweight='bold')
plt.xlabel('Tập dữ liệu')
plt.ylabel('Số lượng segment')
plt.legend(title='Trạng thái', bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()

# Lưu ảnh thay vì hiển thị (nếu chạy trên server không có GUI)
plt.savefig('final_dataset_distribution.png', dpi=150, bbox_inches='tight')
print("Biểu đồ đã được lưu: final_dataset_distribution.png")