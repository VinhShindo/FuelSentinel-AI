export function customTooltipFormatter(points, rawData) {
    return function (params) {
        if (!params || params.length === 0) return '';
        const dataIndex = params[0].dataIndex;
        const item = rawData[dataIndex];
        if (!item) return '';

        const isPredictedData = item.prediction !== undefined;

        // Định dạng timestamp: HH:MM:SS - DD/MM/YYYY
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
            } catch (e) {
                // Giữ nguyên nếu lỗi parse
            }
        }

        let html = `<div style="font-family: Inter, sans-serif; font-size: 13px; line-height: 1.6; min-width: 160px;">
            <div style="font-weight: 600; color: #111827; margin-bottom: 4px; border-bottom: 1px solid #e5e7eb; padding-bottom: 4px;">${formattedTime}</div>`;

        if (isPredictedData) {
            // [FIX] Xóa emoji '🤔', '✅', '⛽', '🚀', '🧠', '📊', '🎯'
            const statusLabel = item.point_status === 'thinking' ? 'Thinking' 
                               : (item.point_status === 'confirmed' ? 'Confirmed' 
                               : (item.point_status === 'normal' ? 'Normal' : '--'));
            
            const confidenceDisplay = item.confidence != null ? `${item.confidence.toFixed(2)}%` : 'N/A';

            html += `<div style="display: flex; justify-content: space-between; gap: 8px;">
                        <span>Fuel</span>
                        <strong>${item.fuel?.toFixed(2)} L</strong>
                     </div>
                     <div style="display: flex; justify-content: space-between; gap: 8px;">
                        <span>Speed</span>
                        <strong>${item.speed?.toFixed(2)} km/h</strong>
                     </div>
                     <div style="display: flex; justify-content: space-between; gap: 8px;">
                        <span>Prediction</span>
                        <strong>${item.prediction || '--'}</strong>
                     </div>
                     <div style="display: flex; justify-content: space-between; gap: 8px;">
                        <span>Status</span>
                        <strong>${statusLabel}</strong>
                     </div>
                     <div style="display: flex; justify-content: space-between; gap: 8px;">
                        <span>Confidence</span>
                        <strong>${confidenceDisplay}</strong>
                     </div>`;
        } else {
            html += `<div style="display: flex; justify-content: space-between; gap: 8px;">
                        <span>Fuel</span>
                        <strong>${item.fuel?.toFixed(2)} L</strong>
                     </div>
                     <div style="display: flex; justify-content: space-between; gap: 8px;">
                        <span>Speed</span>
                        <strong>${item.speed?.toFixed(2)} km/h</strong>
                     </div>
                     <div style="display: flex; justify-content: space-between; gap: 8px;">
                        <span>Label</span>
                        <strong>${item.label || 'Driving'}</strong>
                     </div>`;
        }
        html += `</div>`;
        return html;
    };
}