export function customTooltipFormatter(points, rawData) {
    return function (params) {
        if (!params || params.length === 0) return '';
        const dataIndex = params[0].dataIndex;
        const item = rawData[dataIndex];
        if (!item) return '';

        // Định dạng timestamp
        let formattedTime = item.timestamp || '--';
        if (formattedTime !== '--') {
            try {
                const date = new Date(formattedTime);
                if (!isNaN(date.getTime())) {
                    const hh = String(date.getHours()).padStart(2, '0');
                    const mm = String(date.getMinutes()).padStart(2, '0');
                    const ss = String(date.getSeconds()).padStart(2, '0');
                    const dd = String(date.getDate()).padStart(2, '0');
                    const MM = String(date.getMonth() + 1).padStart(2, '0');
                    const yyyy = date.getFullYear();
                    formattedTime = `${hh}:${mm}:${ss} - ${dd}/${MM}/${yyyy}`;
                }
            } catch (e) { /* giữ nguyên nếu lỗi parse */ }
        }

        let html = `<div style="font-family: Inter, sans-serif; font-size: 13px; line-height: 1.6; min-width: 190px;">
            <div style="font-weight: 600; color: #111827; margin-bottom: 4px; border-bottom: 1px solid #e5e7eb; padding-bottom: 4px;">${formattedTime}</div>`;

        // ==== HIỂN THỊ RAW & FILTERED RÕ RÀNG ====
        html += `
            <div style="display: flex; justify-content: space-between; gap: 8px;">
                <span>Nhiên liệu (RAW)</span>
                <strong style="color: #2563EB;">${item.fuel_raw != null ? item.fuel_raw.toFixed(2) : '--'} L</strong>
            </div>
            <div style="display: flex; justify-content: space-between; gap: 8px;">
                <span>Nhiên liệu (FILTERED)</span>
                <strong style="color: #16A34A;">${item.fuel_filter != null ? item.fuel_filter.toFixed(2) : '--'} L</strong>
            </div>
            <div style="display: flex; justify-content: space-between; gap: 8px;">
                <span>Speed</span>
                <strong>${item.speed?.toFixed(2) ?? '--'} km/h</strong>
            </div>
            <div style="display: flex; justify-content: space-between; gap: 8px;">
                <span>Prediction</span>
                <strong>${item.prediction || item.label || '--'}</strong>
            </div>
            <div style="display: flex; justify-content: space-between; gap: 8px;">
                <span>Status</span>
                <strong>${item.point_status === 'thinking' ? 'Thinking' : item.point_status === 'confirmed' ? 'Confirmed' : item.point_status === 'normal' ? 'Normal' : '--'}</strong>
            </div>
            <div style="display: flex; justify-content: space-between; gap: 8px;">
                <span>Confidence</span>
                <strong>${item.confidence != null ? item.confidence.toFixed(2) + '%' : 'N/A'}</strong>
            </div>`;

        html += `</div>`;
        return html;
    };
}