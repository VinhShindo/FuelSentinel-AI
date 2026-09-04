/**
 * Module: chartBuilder.js
 * Muc tieu: Xay dung cau hinh ECharts cho bieu do Fuel (RAW + FILTERED) + Speed Timeline
 */
import { findSegments, findEvents, findThinkingSegments } from './eventSegment.js';

export const STATE_COLORS = {
    'Idle': 'rgba(107, 114, 128, 0.30)',
    'Driving': 'rgba(245, 158, 11, 0.26)',
    'Refuel': 'rgba(22, 163, 74, 0.40)',
    'Theft': 'rgba(220, 38, 38, 0.40)',
    'Fuel Theft': 'rgba(220, 38, 38, 0.40)'
};

const EVENT_SEMANTIC_STATES = new Set(['Refuel', 'Theft', 'Fuel Theft']);

function isDarkMode() {
    return document.body.classList.contains('dark-mode') || 
           document.documentElement.getAttribute('data-theme') === 'dark';
}

function getChartColors() {
    return {
        textDark: isDarkMode() ? '#F3F4F6' : '#111827',
        textMuted: isDarkMode() ? '#9CA3AF' : '#4B5563',
        gridLine: isDarkMode() ? 'rgba(148,163,184,0.2)' : 'rgba(148,163,184,0.35)',
        borderLight: isDarkMode() ? '#4B5563' : '#E5E7EB',
        bgColor: isDarkMode() ? '#111827' : '#FFFFFF'
    };
}

const TEXT_DARK = isDarkMode() ? '#F3F4F6' : '#111827';
const TEXT_MUTED = isDarkMode() ? '#9CA3AF' : '#4B5563';
const GRID_LINE = isDarkMode() ? 'rgba(148,163,184,0.2)' : 'rgba(148,163,184,0.35)';
const BORDER_LIGHT = isDarkMode() ? '#4B5563' : '#E5E7EB';

function formatTimeLabel(timeStr) {
    if (!timeStr || typeof timeStr !== 'string') return '';
    if (timeStr.includes('T')) return timeStr.split('T')[1].substring(0, 5);
    if (timeStr.includes(' ')) return timeStr.split(' ')[1].substring(0, 5);
    return timeStr.substring(0, 5);
}

function computeRenderStates(states, pointStatuses) {
    if (!pointStatuses || pointStatuses.length === 0) return states;
    return states.map((state, i) => {
        const isEventState = EVENT_SEMANTIC_STATES.has(state);
        const isConfirmed = pointStatuses[i] === 'confirmed';
        if (isEventState && !isConfirmed) {
            return 'Driving';
        }
        return state;
    });
}

function mergeAdjacentSegments(segments) {
    if (!segments || segments.length < 2) return segments;
    const merged = [{ ...segments[0] }];
    for (let i = 1; i < segments.length; i++) {
        const prev = merged[merged.length - 1];
        const cur = segments[i];
        const contiguous =
            (typeof prev.endIdx === 'number' && typeof cur.startIdx === 'number')
                ? cur.startIdx <= prev.endIdx + 1
                : false;
        if (contiguous && prev.state === cur.state) {
            prev.endIdx = cur.endIdx;
            if ('endTime' in cur) prev.endTime = cur.endTime;
            if ('endValue' in cur) prev.endValue = cur.endValue;
            if (typeof prev.startIdx === 'number' && typeof prev.endIdx === 'number') {
                prev.midIndex = Math.round((prev.startIdx + prev.endIdx) / 2);
            }
        } else {
            merged.push({ ...cur });
        }
    }
    return merged;
}

function getShiftAmount(seg, i, boundaryPoints) {
    let shift = 1;
    if (i > 0 && (seg.state === 'Refuel' || seg.state === 'Theft')) {
        let boundaryIdx = seg.startIdx - 2;
        if (boundaryIdx >= 0 && boundaryPoints && boundaryPoints[boundaryIdx] === true) {
            shift = 2;
        }
    }
    return shift;
}

function buildMarkArea(segments, totalLength, boundaryPoints) {
    const markAreas = [];
    if (!Array.isArray(segments) || segments.length === 0) return markAreas;

    for (let i = 0; i < segments.length; i++) {
        const seg = segments[i];
        if (typeof seg.startIdx !== 'number' || typeof seg.endIdx !== 'number') continue;

        let endIdx = Math.min(Math.max(seg.startIdx, seg.endIdx), totalLength - 1);
        let shiftAmount = getShiftAmount(seg, i, boundaryPoints);
        let startIdx = Math.max(0, seg.startIdx - shiftAmount);

        markAreas.push([
            {
                xAxis: startIdx,
                itemStyle: {
                    color: STATE_COLORS[seg.state] || 'rgba(255,255,255,0)',
                    opacity: 1,
                    borderWidth: 1,
                    borderColor: BORDER_LIGHT,
                    borderType: 'solid'
                }
            },
            { xAxis: endIdx }
        ]);
    }
    return markAreas;
}

function buildThinkingOverlayAreas(thinkingSegments, totalLength) {
    const areas = [];
    if (!Array.isArray(thinkingSegments) || thinkingSegments.length === 0) return areas;
    thinkingSegments.forEach((seg) => {
        if (typeof seg.startIdx !== 'number' || typeof seg.endIdx !== 'number') return;
        const startIdx = Math.max(0, seg.startIdx);
        const endIdx = Math.min(Math.max(startIdx, seg.endIdx), totalLength - 1);
        areas.push([
            {
                xAxis: startIdx,
                itemStyle: {
                    color: 'rgba(0,0,0,0)',
                    opacity: 1,
                    borderWidth: 1.5,
                    borderColor: '#6B7280',
                    borderType: 'dashed'
                }
            },
            { xAxis: endIdx }
        ]);
    });
    return areas;
}

function buildThinkingOverlayLabels(thinkingSegments) {
    if (!Array.isArray(thinkingSegments)) return [];
    return thinkingSegments
        .filter(seg => typeof seg.midIndex === 'number')
        .map(seg => ({
            coord: [seg.midIndex, 0],
            symbol: 'rect', symbolSize: [0, 0],
            label: {
                show: true,
                formatter: 'Thinking…',
                position: 'bottom',
                color: TEXT_MUTED,
                fontStyle: 'italic',
                fontSize: 9,
                fontWeight: 'bold',
                fontFamily: 'Inter'
            }
        }));
}

function buildSegmentLabels(segments) {
    return segments.map(seg => {
        if (typeof seg.midIndex !== 'number' || (seg.endIdx - seg.startIdx) === 0) return [];
        const segmentLength = seg.endIdx - seg.startIdx + 1;
        const displayName = seg.state;
        let segLabel;
        if (segmentLength < 5) {
            segLabel = `${displayName}`;
        } else {
            const startTimeLabel = formatTimeLabel(seg.startTime);
            const endTimeLabel = formatTimeLabel(seg.endTime);
            segLabel = `${displayName}\n${startTimeLabel} - ${endTimeLabel}`;
        }
        let fontSize = 10;
        if (segmentLength <= 3) fontSize = 8;
        else if (segmentLength <= 6) fontSize = 9;
        return {
            coord: [seg.midIndex, 0],
            symbol: 'rect', symbolSize: [0, 0],
            label: {
                show: true,
                formatter: segLabel,
                position: 'top',
                color: TEXT_DARK,
                fontStyle: 'normal',
                fontSize: fontSize,
                fontWeight: 'bold',
                fontFamily: 'Inter',
                lineHeight: 14
            }
        };
    }).flat();
}

function buildValueLabels(segments, valueKeyStart, valueKeyEnd, unitSuffix) {
    const labels = [];
    if (!segments || segments.length === 0) return labels;
    const segCount = segments.length;
    segments.forEach((seg, i) => {
        if (typeof seg.startIdx !== 'number' || typeof seg.endIdx !== 'number') return;
        const prevSeg = i > 0 ? segments[i - 1] : null;
        const nextSeg = i < segCount - 1 ? segments[i + 1] : null;
        const startVal = seg[valueKeyStart];
        const endVal = seg[valueKeyEnd];
        let showStart = true;
        if (prevSeg && Math.abs(seg.startIdx - prevSeg.endIdx) < 2) showStart = false;
        if (i === 0) showStart = true;
        if (showStart && typeof startVal === 'number') {
            labels.push({
                coord: [seg.startIdx, startVal],
                symbol: 'circle', symbolSize: 0,
                label: { show: true, position: 'top', formatter: `${startVal.toFixed(0)}${unitSuffix}`, fontSize: 11, fontWeight: 'bold', color: TEXT_DARK }
            });
        }
        let showEnd = true;
        if (nextSeg && Math.abs(nextSeg.startIdx - seg.endIdx) < 2) showEnd = false;
        if (i === segCount - 1) showEnd = true;
        if (showEnd && typeof endVal === 'number') {
            labels.push({
                coord: [seg.endIdx, endVal],
                symbol: 'circle', symbolSize: 0,
                label: { show: true, position: 'top', formatter: `${endVal.toFixed(0)}${unitSuffix}`, fontSize: 11, fontWeight: 'bold', color: TEXT_DARK }
            });
        }
    });
    return labels;
}

function buildBoundaryMarkers(segments, valueKeyStart, valueKeyEnd) {
    const boundaries = [];
    segments.forEach(seg => {
        if (typeof seg.startIdx !== 'number') return;
        const dotColor = STATE_COLORS[seg.state] || '#6b7280';
        const markerStyle = { color: dotColor, borderColor: '#FFFFFF', borderWidth: 2, shadowBlur: 4, shadowColor: 'rgba(0,0,0,0.15)' };
        if (typeof seg[valueKeyStart] === 'number') {
            boundaries.push({ coord: [seg.startIdx, seg[valueKeyStart]], symbol: 'circle', symbolSize: 8, itemStyle: markerStyle });
        }
        if (typeof seg[valueKeyEnd] === 'number') {
            boundaries.push({ coord: [seg.endIdx, seg[valueKeyEnd]], symbol: 'circle', symbolSize: 8, itemStyle: markerStyle });
        }
    });
    return boundaries;
}

function buildEventPins(events, boundaryPoints) {
    return events.map(evt => {
        if (typeof evt.startIdx !== 'number' || typeof evt.endIdx !== 'number') return [];
        const color = evt.state === 'Refuel' ? '#16A34A' : '#DC2626';
        const charLabel = evt.state === 'Refuel' ? 'R' : 'T';

        let shift = 1;
        if (evt.state === 'Refuel' || evt.state === 'Theft') {
            let boundaryIdx = evt.startIdx - 2;
            if (boundaryIdx >= 0 && boundaryPoints && boundaryPoints[boundaryIdx] === true) {
                shift = 2;
            }
        }

        return [
            {
                coord: [Math.max(0, evt.startIdx - shift), evt.startValue],
                symbol: 'pin',
                symbolSize: 34,
                symbolOffset: [0, '-50%'],
                itemStyle: {
                    color: color,
                    borderColor: '#FFFFFF',
                    borderWidth: 2.5
                },
                label: {
                    show: true,
                    formatter: charLabel,
                    color: '#FFFFFF',
                    fontWeight: 'bold',
                    fontSize: 15,
                    position: 'inside'
                }
            },
            {
                coord: [evt.endIdx, evt.endValue],
                symbol: 'pin',
                symbolSize: 34,
                symbolOffset: [0, '-50%'],
                itemStyle: {
                    color: color,
                    borderColor: '#FFFFFF',
                    borderWidth: 2.5
                },
                label: {
                    show: true,
                    formatter: charLabel,
                    color: '#FFFFFF',
                    fontWeight: 'bold',
                    fontSize: 15,
                    position: 'inside'
                }
            }
        ];
    }).flat();
}

export function buildChartOption(x_data, y_raw_data, y_filtered_data, speed_data, states, rawData, pointStatuses, yAxisMax, yAxisMin) {
    const boundaryPoints = rawData.map(item => item.boundary_point || false);

    const renderStates = computeRenderStates(states, pointStatuses);

    // Sử dụng y_raw_data cho các segment (màu nền, label, pin)
    const fuelSegments = mergeAdjacentSegments(findSegments(x_data, y_raw_data, renderStates));
    const events = findEvents(fuelSegments, x_data, y_raw_data);

    const hasSpeed = Array.isArray(speed_data) && speed_data.length === y_raw_data.length;
    const speedSegments = hasSpeed
        ? mergeAdjacentSegments(findSegments(x_data, speed_data, renderStates))
        : [];

    const thinkingSegments = findThinkingSegments(x_data, pointStatuses);

    const fuelMarkArea = [
        ...buildMarkArea(fuelSegments, y_raw_data.length, boundaryPoints),
        ...buildThinkingOverlayAreas(thinkingSegments, y_raw_data.length)
    ];
    const speedMarkArea = hasSpeed
        ? [
            ...buildMarkArea(speedSegments, speed_data.length, boundaryPoints),
            ...buildThinkingOverlayAreas(thinkingSegments, speed_data.length)
        ]
        : [];

    const segmentLabelData = buildSegmentLabels(fuelSegments);
    const fuelLabelData = buildValueLabels(fuelSegments, 'startValue', 'endValue', 'L');
    const boundaryMarkerData = buildBoundaryMarkers(fuelSegments, 'startValue', 'endValue');
    const eventPinData = buildEventPins(events, boundaryPoints);
    const thinkingLabelData = buildThinkingOverlayLabels(thinkingSegments);
    const combinedFuelMarkPoints = [...segmentLabelData, ...fuelLabelData, ...boundaryMarkerData, ...eventPinData, ...thinkingLabelData];

    const speedLabelData = hasSpeed ? buildValueLabels(speedSegments, 'startValue', 'endValue', '') : [];
    const speedBoundaryData = hasSpeed ? buildBoundaryMarkers(speedSegments, 'startValue', 'endValue') : [];
    const combinedSpeedMarkPoints = [...speedLabelData, ...speedBoundaryData];

    const FLOOR_MAX = 500;
    const finalYMax = Math.max(yAxisMax || FLOOR_MAX, FLOOR_MAX);
    const finalYMin = typeof yAxisMin === 'number' ? yAxisMin : 0;
    const maxSpeed = hasSpeed ? Math.max(20, ...speed_data) : 20;
    const speedYMax = Math.ceil((maxSpeed * 1.15) / 10) * 10;

    const colors = getChartColors();

    const option = {
        backgroundColor: colors.bgColor,
        textStyle: { color: colors.textDark },
        tooltip: { trigger: 'axis' },
        legend: { 
            show: true, 
            data: ['Nhiên liệu (RAW)', 'Nhiên liệu (FILTERED)', 'Tốc độ'],
            textStyle: { color: colors.textDark }
        },
        axisPointer: { link: [{ xAxisIndex: hasSpeed ? [0, 1] : [0] }] },
        dataZoom: hasSpeed ? [
            { type: 'slider', xAxisIndex: [0, 1], start: 0, end: 100, height: 10, bottom: 6, borderColor: BORDER_LIGHT },
            { type: 'inside', xAxisIndex: [0, 1] }
        ] : [
            { type: 'slider', start: 0, end: 100, height: 10, bottom: 6, borderColor: BORDER_LIGHT },
            { type: 'inside' }
        ],
        grid: hasSpeed ? [
            { left: '2%', right: '1%', top: '6%', height: '52%', containLabel: true },
            { left: '2%', right: '1%', top: '66%', height: '24%', containLabel: true }
        ] : [
            { left: '2%', right: '1%', top: '8%', bottom: '18%', containLabel: true }
        ],
        xAxis: hasSpeed ? [
            {
                type: 'category', boundaryGap: false, data: x_data, gridIndex: 0,
                axisLabel: { show: false },
                axisLine: { show: true, lineStyle: { color: BORDER_LIGHT } },
                axisTick: { show: false }, splitLine: { show: false }
            },
            {
                type: 'category', boundaryGap: false, data: x_data, gridIndex: 1,
                axisLabel: { fontSize: 10, fontWeight: '600', color: TEXT_MUTED, formatter: (v) => formatTimeLabel(v) },
                axisLine: { show: true, lineStyle: { color: BORDER_LIGHT } },
                axisTick: { show: false }, splitLine: { show: false }
            }
        ] : [{
            type: 'category', boundaryGap: false, data: x_data,
            axisLabel: { fontSize: 10, fontWeight: '600', color: TEXT_MUTED, formatter: (v) => formatTimeLabel(v) },
            axisLine: { show: true, lineStyle: { color: BORDER_LIGHT } },
            axisTick: { show: false }, splitLine: { show: false }
        }],
        yAxis: hasSpeed ? [
            {
                type: 'value', name: 'Nhiên liệu (L)', gridIndex: 0,
                nameTextStyle: { color: TEXT_MUTED }, min: finalYMin, max: finalYMax,
                axisLine: { show: false }, axisTick: { show: false },
                axisLabel: { color: TEXT_MUTED },
                splitLine: { show: true, lineStyle: { color: GRID_LINE, type: 'dashed' } }
            },
            {
                type: 'value', name: 'Tốc độ (km/h)', gridIndex: 1,
                nameTextStyle: { color: TEXT_MUTED }, min: 0, max: speedYMax,
                axisLine: { show: false }, axisTick: { show: false },
                axisLabel: { color: TEXT_MUTED },
                splitLine: { show: true, lineStyle: { color: GRID_LINE, type: 'dashed' } }
            }
        ] : [{
            type: 'value', name: 'Nhiên liệu (L)',
            nameTextStyle: { color: TEXT_MUTED }, min: finalYMin, max: finalYMax,
            axisLine: { show: false }, axisTick: { show: false },
            axisLabel: { color: TEXT_MUTED },
            splitLine: { show: true, lineStyle: { color: GRID_LINE, type: 'dashed' } }
        }],
        animation: true,
        series: []
    };

    // Series RAW - xanh dương, nét liền
    option.series.push({
        name: 'Nhiên liệu (RAW)',
        type: 'line',
        data: y_raw_data,
        xAxisIndex: 0, yAxisIndex: 0,
        smooth: true,
        symbol: 'circle',
        symbolSize: 6,
        areaStyle: {
            color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                { offset: 0, color: 'rgba(37, 99, 235, 0.20)' },
                { offset: 1, color: 'rgba(37, 99, 235, 0.02)' }
            ])
        },
        lineStyle: { color: '#2563EB', width: 3, shadowBlur: 8, shadowColor: 'rgba(37,99,235,0.18)' },
        itemStyle: { color: '#2563EB' },
        markArea: { silent: true, data: fuelMarkArea },
        markPoint: { data: combinedFuelMarkPoints, symbol: (value, params) => params.symbol }
    });

    // Series FILTERED - xanh lá, nét đứt
    option.series.push({
        name: 'Nhiên liệu (FILTERED)',
        type: 'line',
        data: y_filtered_data,
        xAxisIndex: 0, yAxisIndex: 0,
        smooth: true,
        symbol: 'circle',
        symbolSize: 4,
        lineStyle: { color: '#DC2626', width: 2, type: 'dashed' }, // ĐỔI MÀU ĐỎ
        itemStyle: { color: '#DC2626' },                            // ĐỔI MÀU ĐỎ
        markPoint: { data: [] }
    });

    if (hasSpeed) {
        option.series.push({
            name: 'Tốc độ',
            type: 'line',
            data: speed_data,
            xAxisIndex: 1, yAxisIndex: 1,
            smooth: true,
            symbol: 'circle',
            symbolSize: 5,
            lineStyle: { color: '#7C3AED', width: 2.5 },
            itemStyle: { color: '#7C3AED' },
            markArea: { silent: true, data: speedMarkArea },
            markPoint: { data: combinedSpeedMarkPoints, symbol: (value, params) => params.symbol }
        });
    }

    return option;
}