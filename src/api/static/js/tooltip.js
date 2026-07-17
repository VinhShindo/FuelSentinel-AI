/**
 * Module: Custom ECharts Tooltip
 * Hiển thị Raw sensor và Model output tách biệt.
 */
export function customTooltipFormatter(data, rawData) {
    return function (params) {
        const p = params[0];
        if (!p) return '';
        // data là mảng [{raw, processed}]
        const row = data[p.dataIndex];
        if (!row) return `<strong>${p.axisValue}</strong>`;
        
        const raw = row.raw;
        const processed = row.processed;

        let extraEventHtml = '';
        if (processed.Prediction === 'Refuel' || processed.Prediction === 'Fuel Theft') {
            const prevIdx = p.dataIndex > 0 ? p.dataIndex - 1 : p.dataIndex;
            const prevRow = data[prevIdx];
            if (prevRow) {
                const prevFuel = prevRow.raw.Fuel;
                const delta = (raw.Fuel - prevFuel).toFixed(1);
                extraEventHtml = `
                    <div style="border-top: 1px solid #374151; padding-top: 6px; margin-top: 6px;">
                        <div style="display: flex; justify-content: space-between;"><span style="color:#9ca3af;">Fuel Δ:</span> <strong class="${delta > 0 ? 'text-success' : 'text-danger'}">${delta > 0 ? '+' : ''}${delta} L</strong></div>
                    </div>
                `;
            }
        }

        return `
            <div style="font-family: 'Inter', sans-serif; max-width: 260px; line-height: 1.6;">
                <div style="font-weight: 700; border-bottom: 1px solid #374151; padding-bottom: 6px; margin-bottom: 6px; display: flex; justify-content: space-between;">
                    <span>${raw.Timestamp}</span>
                    <span>${processed.Prediction}</span>
                </div>
                
                <div style="margin-bottom: 8px;">
                    <div style="font-weight: 600; color: #9ca3af; font-size: 0.75rem; text-transform: uppercase;">Raw Sensor</div>
                    <div style="display: flex; justify-content: space-between;"><span style="color:#9ca3af;">⛽ Fuel:</span> <strong>${raw.Fuel} L</strong></div>
                    <div style="display: flex; justify-content: space-between;"><span style="color:#9ca3af;">🚛 Speed:</span> <strong>${raw.Speed} km/h</strong></div>
                    <div style="display: flex; justify-content: space-between;"><span style="color:#9ca3af;">📡 ADC:</span> <strong>${raw.ADC}</strong></div>
                    <div style="display: flex; justify-content: space-between;"><span style="color:#9ca3af;">📍 GPS:</span> <strong>${raw.Latitude.toFixed(4)}, ${raw.Longitude.toFixed(4)}</strong></div>
                </div>

                <div style="border-top: 1px solid #374151; padding-top: 6px;">
                    <div style="font-weight: 600; color: #9ca3af; font-size: 0.75rem; text-transform: uppercase;">Model Output</div>
                    <div style="display: flex; justify-content: space-between;"><span style="color:#9ca3af;">📈 Fuel Rate:</span> <strong>${processed.FuelRate.toFixed(3)} L/s</strong></div>
                    <div style="display: flex; justify-content: space-between;"><span style="color:#9ca3af;">📉 Slope:</span> <strong>${processed.RegressionSlope.toFixed(4)}</strong></div>
                    <div style="display: flex; justify-content: space-between;"><span style="color:#9ca3af;">📊 Moving Avg:</span> <strong>${processed.MovingAvg.toFixed(1)} L</strong></div>
                    <div style="display: flex; justify-content: space-between;"><span style="color:#9ca3af;">📊 Rolling Std:</span> <strong>${processed.RollingStd.toFixed(3)}</strong></div>
                    <div style="display: flex; justify-content: space-between;"><span style="color:#9ca3af;">⏱ Stop Dur:</span> <strong>${processed.StopDuration || 0}s</strong></div>
                    ${extraEventHtml}
                    <div style="display: flex; justify-content: space-between; border-top: 1px solid #374151; padding-top: 4px; margin-top: 4px;">
                        <span style="color:#9ca3af;">📡 Confidence:</span> <span style="color: #16A34A;">${processed.Confidence || 98.5}%</span>
                    </div>
                </div>
            </div>
        `;
    };
}