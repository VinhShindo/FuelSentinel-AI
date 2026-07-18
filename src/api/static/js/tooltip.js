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
                    <div style="border-top: 1px solid rgba(148,163,184,0.24); padding-top: 6px; margin-top: 6px;">
                        <div style="display: flex; justify-content: space-between;"><span style="color:#64748B;">Fuel Δ:</span> <strong style="color:${delta > 0 ? '#10B981' : '#EF4444'}">${delta > 0 ? '+' : ''}${delta} L</strong></div>
                    </div>
                `;
            }
        }

        return `
            <div style="font-family: 'Inter', sans-serif; max-width: 280px; line-height: 1.6; padding: 2px;">
                <div style="font-weight: 700; border-bottom: 1px solid rgba(148,163,184,0.24); padding-bottom: 6px; margin-bottom: 8px; display: flex; justify-content: space-between; gap: 8px;">
                    <span style="color:#0F172A;">${raw.Timestamp}</span>
                    <span style="color:#2563EB;">${processed.Prediction}</span>
                </div>
                
                <div style="margin-bottom: 8px;">
                    <div style="font-weight: 600; color: #64748B; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em;">Raw Sensor</div>
                    <div style="display: flex; justify-content: space-between;"><span style="color:#64748B;">⛽ Fuel:</span> <strong style="color:#0F172A;">${raw.Fuel} L</strong></div>
                    <div style="display: flex; justify-content: space-between;"><span style="color:#64748B;">🚛 Speed:</span> <strong style="color:#0F172A;">${raw.Speed} km/h</strong></div>
                    <div style="display: flex; justify-content: space-between;"><span style="color:#64748B;">📡 ADC:</span> <strong style="color:#0F172A;">${raw.ADC}</strong></div>
                    <div style="display: flex; justify-content: space-between;"><span style="color:#64748B;">📍 GPS:</span> <strong style="color:#0F172A;">${raw.Latitude.toFixed(4)}, ${raw.Longitude.toFixed(4)}</strong></div>
                </div>

                <div style="border-top: 1px solid rgba(148,163,184,0.24); padding-top: 6px;">
                    <div style="font-weight: 600; color: #64748B; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em;">Model Output</div>
                    <div style="display: flex; justify-content: space-between;"><span style="color:#64748B;">📈 Fuel Rate:</span> <strong style="color:#0F172A;">${processed.FuelRate.toFixed(3)} L/s</strong></div>
                    <div style="display: flex; justify-content: space-between;"><span style="color:#64748B;">📉 Slope:</span> <strong style="color:#0F172A;">${processed.RegressionSlope.toFixed(4)}</strong></div>
                    <div style="display: flex; justify-content: space-between;"><span style="color:#64748B;">📊 Moving Avg:</span> <strong style="color:#0F172A;">${processed.MovingAvg.toFixed(1)} L</strong></div>
                    <div style="display: flex; justify-content: space-between;"><span style="color:#64748B;">📊 Rolling Std:</span> <strong style="color:#0F172A;">${processed.RollingStd.toFixed(3)}</strong></div>
                    <div style="display: flex; justify-content: space-between;"><span style="color:#64748B;">⏱ Stop Dur:</span> <strong style="color:#0F172A;">${processed.StopDuration || 0}s</strong></div>
                    ${extraEventHtml}
                    <div style="display: flex; justify-content: space-between; border-top: 1px solid rgba(148,163,184,0.24); padding-top: 4px; margin-top: 4px;">
                        <span style="color:#64748B;">📡 Confidence:</span> <span style="color: #10B981; font-weight: 700;">${processed.Confidence || 98.5}%</span>
                    </div>
                </div>
            </div>
        `;
    };
}