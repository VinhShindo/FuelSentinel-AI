import { findSegments, findEvents } from './eventSegment.js';

// Hàm tiện ích: trích xuất giờ:phút từ timestamp
function formatTimeLabel(timeStr) {
    if (!timeStr || typeof timeStr !== 'string') return '';
    if (timeStr.includes('T')) return timeStr.split('T')[1].substring(0, 5);
    if (timeStr.includes(' ')) return timeStr.split(' ')[1].substring(0, 5);
    return timeStr.substring(0, 5);
}

const STATE_COLORS = {
    'Idle': '#E5E7EB',
    'Driving': '#FEF08A',
    'Refuel': 'rgba(34, 197, 94, 0.25)',
    'Fuel Theft': 'rgba(239, 68, 68, 0.25)'
};

function buildMarkArea(segments, totalLength) {
    const markAreas = [];
    segments.forEach((seg, i) => {
        if (typeof seg.startIdx !== 'number') return;
        const endBoundary = (i < segments.length - 1) ? segments[i+1].startIdx : (totalLength - 1);
        markAreas.push([
            { 
                xAxis: seg.startIdx, 
                itemStyle: { 
                    color: STATE_COLORS[seg.state] || '#fff', 
                    opacity: 0.85, 
                    borderWidth: 1,
                    borderColor: 'rgba(0,0,0,0.05)' 
                } 
            },
            { xAxis: endBoundary }
        ]);
    });
    return markAreas;
}

// ============================================================
//  HIỂN THỊ LABEL THEO ĐỘ DÀI SEGMENT
// ============================================================
function buildSegmentLabels(segments) {
    return segments.map(seg => {
        // Chỉ bỏ nếu segment có đúng 1 mẫu (startIdx === endIdx)
        if (typeof seg.midIndex !== 'number' || (seg.endIdx - seg.startIdx) === 0) return [];
        
        const segmentLength = seg.endIdx - seg.startIdx + 1;
        let segLabel;
        if (segmentLength < 5) {
            // Segment ngắn (< 5 mẫu) chỉ hiển thị tên trạng thái
            segLabel = `${seg.state}`;
        } else {
            const startTimeLabel = formatTimeLabel(seg.startTime);
            const endTimeLabel = formatTimeLabel(seg.endTime);
            segLabel = `${seg.state}\n${startTimeLabel} - ${endTimeLabel}`;
        }
        
        // Luôn đặt label ở dưới cùng biểu đồ (trục y = 0)
        const labelYPosition = 0;
        
        // Điều chỉnh font dựa trên độ dài segment
        let fontSize = 10;
        if (segmentLength <= 3) fontSize = 8;
        else if (segmentLength <= 6) fontSize = 9;
        
        return {
            coord: [seg.midIndex, labelYPosition],
            symbol: 'rect', symbolSize: [0, 0],
            label: { 
                show: true, 
                formatter: segLabel, 
                position: 'top',
                color: '#1f2937',
                fontSize: fontSize, 
                fontWeight: 'bold', 
                fontFamily: 'Inter', 
                lineHeight: 14 
            }
        };
    }).flat();
}

function buildFuelLabels(segments) {
    const labels = [];
    if (!segments || segments.length === 0) return labels;
    const segCount = segments.length;
    segments.forEach((seg, i) => {
        if (typeof seg.startIdx !== 'number' || typeof seg.endIdx !== 'number') return;
        const prevSeg = i > 0 ? segments[i-1] : null;
        const nextSeg = i < segCount - 1 ? segments[i+1] : null;
        
        let showStart = true;
        if (prevSeg && Math.abs(seg.startIdx - prevSeg.endIdx) < 2) showStart = false;
        if (i === 0) showStart = true;
        if (showStart) {
            labels.push({ 
                coord: [seg.startIdx, seg.startFuel], 
                symbol: 'circle', symbolSize: 0, 
                label: { show: true, position: 'top', formatter: `${seg.startFuel.toFixed(0)}L`, fontSize: 11, fontWeight: 'bold', color: '#1f2937' } 
            });
        }

        let showEnd = true;
        if (nextSeg && Math.abs(nextSeg.startIdx - seg.endIdx) < 2) showEnd = false;
        if (i === segCount - 1) showEnd = true;
        if (showEnd) {
            labels.push({ 
                coord: [seg.endIdx, seg.endFuel], 
                symbol: 'circle', symbolSize: 0, 
                label: { show: true, position: 'top', formatter: `${seg.endFuel.toFixed(0)}L`, fontSize: 11, fontWeight: 'bold', color: '#1f2937' } 
            });
        }
    });
    return labels;
}

function buildBoundaryMarkers(segments) {
    const boundaries = [];
    segments.forEach(seg => {
        if (typeof seg.startIdx !== 'number') return;
        const markerStyle = { color: STATE_COLORS[seg.state] || '#6b7280', borderColor: '#fff', borderWidth: 2, shadowBlur: 4, shadowColor: 'rgba(0,0,0,0.2)' };
        boundaries.push({ coord: [seg.startIdx, seg.startFuel], symbol: 'circle', symbolSize: 8, itemStyle: markerStyle });
        boundaries.push({ coord: [seg.endIdx, seg.endFuel], symbol: 'circle', symbolSize: 8, itemStyle: markerStyle });
    });
    return boundaries;
}

function buildEventPins(events) {
    return events.map(evt => {
        if (typeof evt.startIdx !== 'number' || typeof evt.endIdx !== 'number') return [];
        const color = evt.state === 'Refuel' ? '#16A34A' : '#DC2626';
        const icon = evt.state === 'Refuel' ? '⛽' : '⚠';
        return [
            { coord: [evt.startIdx, evt.startValue], symbol: 'pin', symbolSize: 30, itemStyle: { color: color, borderColor: '#fff', borderWidth: 2 }, label: { formatter: icon, fontSize: 14, position: 'top', color: '#fff' } },
            { coord: [evt.endIdx, evt.endValue], symbol: 'pin', symbolSize: 30, itemStyle: { color: color, borderColor: '#fff', borderWidth: 2 }, label: { formatter: icon, fontSize: 14, position: 'top', color: '#fff' } }
        ];
    }).flat();
}

export function buildChartOption(x_data, y_data, states, rawData) {
    const segments = findSegments(x_data, y_data, states);
    const events = findEvents(segments, x_data, y_data);
    
    const markAreaData = buildMarkArea(segments, y_data.length);
    const segmentLabelData = buildSegmentLabels(segments);
    const fuelLabelData = buildFuelLabels(segments);
    const boundaryMarkerData = buildBoundaryMarkers(segments);
    const eventPinData = buildEventPins(events);

    const combinedMarkPoints = [...segmentLabelData, ...fuelLabelData, ...boundaryMarkerData, ...eventPinData];

    // ============================================================
    //  TRỤC Y CỐ ĐỊNH 100, VƯỢT MỚI TỰ TĂNG, KHOẢNG 10
    // ============================================================
    const maxDataValue = Math.max(...y_data);
    const yMax = Math.max(100, maxDataValue);
    // ============================================================

    return {
        tooltip: { /* tooltip xử lý riêng */ },
        legend: { data: ['Mức nhiên liệu'], bottom: 0 },
        dataZoom: [{ type: 'slider', start: 0, end: 100, height: 10, bottom: 35 }, { type: 'inside' }],
        grid: { left: '5%', right: '5%', top: '12%', bottom: '28%' },
        
        xAxis: {
            type: 'category', boundaryGap: false, data: x_data,
            axisLabel: { 
                fontSize: 10, fontWeight: '600', color: '#6b7280',
                formatter: function(value) { return formatTimeLabel(value); }
            },
            splitLine: { show: true, lineStyle: { color: '#f3f4f6', type: 'dashed' } }
        },
        yAxis: {
            type: 'value', name: 'Nhiên liệu (L)', 
            min: 0,
            max: yMax,
            interval: 10, // Khoảng cách 10
            splitLine: { lineStyle: { color: '#f3f4f6' } }
        },
        animation: true,
        series: [{
            name: 'Mức nhiên liệu', type: 'line', data: y_data,
            smooth: true,
            symbol: 'circle', symbolSize: 0,
            areaStyle: { color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [{ offset: 0, color: 'rgba(37, 99, 235, 0.25)' }, { offset: 1, color: 'rgba(37, 99, 235, 0.01)' }]) },
            lineStyle: { color: '#2563EB', width: 2 },
            markArea: { silent: true, data: markAreaData },
            markPoint: { 
                data: combinedMarkPoints,
                symbol: (value, params) => { return params.symbol; }
            }
        }]
    };
}