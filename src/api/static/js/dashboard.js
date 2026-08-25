import { buildChartOption } from './chartBuilder.js';
import { customTooltipFormatter } from './tooltip.js';

const API_BASE = window.location.origin;
let fuelChart, lifecycleChart;
let currentVehicle = window.currentVehicle || 'Car 2';

const DEFAULT_VISIBLE_POINTS = 20;
const MAX_POINTS = 500;
const FUEL_Y_FLOOR = 500;
let fuelYAxisMax = FUEL_Y_FLOOR;

let chartPoints = [];
let lastChartTimestamp = null;

function toDate(ts) {
    if (!ts) return null;
    return new Date(String(ts).replace(' ', 'T'));
}

const SAMPLE_LIFECYCLE = {
    confidence_history: [
        { timestamp: '10:16:00', label: 'Driving', confidence: 0.05 },
        { timestamp: '10:16:20', label: 'Driving', confidence: 0.12 },
        { timestamp: '10:17:00', label: 'Refuel', confidence: 0.28 },
        { timestamp: '10:17:20', label: 'Refuel', confidence: 0.48 },
        { timestamp: '10:18:00', label: 'Refuel', confidence: 0.68 },
        { timestamp: '10:18:20', label: 'Refuel', confidence: 0.85 },
        { timestamp: '10:18:40', label: 'Refuel', confidence: 0.93 },
        { timestamp: '10:20:00', label: 'Refuel', confidence: 0.97 },
        { timestamp: '10:21:00', label: 'Refuel', confidence: 0.98 },
        { timestamp: '10:22:10', label: 'Refuel', confidence: 0.90 },
        { timestamp: '10:23:00', label: 'Driving', confidence: 0.30 },
        { timestamp: '10:23:30', label: 'Driving', confidence: 0.10 },
        { timestamp: '10:24:00', label: 'Driving', confidence: 0.05 }
    ],
    status: 'finished'
};
const SAMPLE_EVENT = {
    event_type: 'Refuel',
    start_time: '10:18:40',
    end_time: '10:22:10',
    duration_s: 210,
    min_fuel: 49.6,
    fuel_added: 2.1,
    max_confidence: 98,
    stage_history: [
        { stage: 'normal', timestamp: '10:14:30' },
        { stage: 'candidate', timestamp: '10:17:20' },
        { stage: 'confirmed', timestamp: '10:18:40' },
        { stage: 'monitoring', timestamp: '10:20:30' },
        { stage: 'finished', timestamp: '10:22:10' }
    ]
};

const lifecycleBaseOption = {
    backgroundColor: '#FFFFFF',
    title: { left: 'center', textStyle: { color: '#111827', fontSize: 14 } },
    tooltip: {
        trigger: 'axis',
        backgroundColor: '#FFFFFF',
        borderColor: '#E5E7EB',
        textStyle: { color: '#111827' },
        formatter: function (params) {
            if (!params || params.length === 0) return '';
            const item = params[0];
            const timestamp = item.axisValueLabel;
            const conf = item.data;
            const formattedTime = formatFullTimestamp(timestamp);
            const formattedConf = conf != null ? conf.toFixed(2) : '--';
            return `
                <div style="font-family: Inter, sans-serif; font-size: 13px; line-height: 1.6; min-width: 140px;">
                    <div style="font-weight: 600; color: #111827; margin-bottom: 4px;">${formattedTime}</div>
                    <div>Confidence: <strong>${formattedConf}</strong></div>
                </div>
            `;
        }
    },
    xAxis: {
        type: 'category',
        data: [],
        axisLabel: {
            color: '#4B5563',
            interval: 0,
            fontSize: 10,
            fontWeight: 'bold',
            margin: 8
        },
        axisLine: { lineStyle: { color: '#E5E7EB' } }
    },
    yAxis: {
        type: 'value', min: 0, max: 1,
        axisLabel: {
            color: '#4B5563',
            fontSize: 10,
            fontWeight: 'bold',
            formatter: '{value}'
        },
        axisLine: { lineStyle: { color: '#E5E7EB' } },
        splitLine: { lineStyle: { color: 'rgba(148,163,184,0.35)', type: 'dashed' } }
    },
    series: [{
        name: 'Confidence', type: 'line', data: [], smooth: true,
        lineStyle: { color: '#D97706', width: 2 },
        itemStyle: { color: '#D97706' },
        markLine: {
            silent: true, symbol: 'none',
            data: [
                { yAxis: 0.6, lineStyle: { color: '#F59E0B', type: 'dashed' }, label: { formatter: 'Candidate (0.6)', color: '#92400E' } },
                { yAxis: 0.8, lineStyle: { color: '#DC2626', type: 'dashed' }, label: { formatter: 'Confirmed (0.8)', color: '#991B1B' } }
            ]
        },
        markArea: { data: [] }
    }]
};

let lifecycleTimestamps = [];
let lifecycleConfs = [];
let lifecycleLabels = [];
let usingSampleLifecycle = false;

function formatStopDurationMinutes(seconds) {
    const duration = Number(seconds || 0);
    return `${(duration / 60).toFixed(2)} m`;
}

function formatDurationShort(seconds) {
    const s = Number(seconds || 0);
    if (s === 0) return '--';
    const m = Math.floor(s / 60);
    const r = Math.round(s % 60);
    if (m > 0) return `${m}m ${r}s`;
    return `${r}s`;
}

function shortTime(ts) {
    if (!ts) return '--';
    if (typeof ts !== 'string') return String(ts);
    if (ts.includes('T')) return (ts.split('T')[1] || '').split('.')[0].substring(0, 8) || ts;
    if (ts.includes(' ')) return ts.split(' ')[1]?.substring(0, 8) || ts;
    return ts;
}

function formatFullTimestamp(isoString) {
    if (!isoString) return '--';
    try {
        const date = new Date(isoString);
        const hours = String(date.getHours()).padStart(2, '0');
        const minutes = String(date.getMinutes()).padStart(2, '0');
        const seconds = String(date.getSeconds()).padStart(2, '0');
        const day = String(date.getDate()).padStart(2, '0');
        const month = String(date.getMonth() + 1).padStart(2, '0');
        const year = date.getFullYear();
        return `${hours}:${minutes}:${seconds} - ${day}/${month}/${year}`;
    } catch (e) { return isoString; }
}

document.addEventListener('DOMContentLoaded', () => {
    const fuelDom = document.getElementById('fuel-chart');
    fuelChart = echarts.init(fuelDom, null, { renderer: 'canvas' });

    const lifecycleDom = document.getElementById('lifecycle-chart');
    lifecycleChart = echarts.init(lifecycleDom, null, { renderer: 'canvas' });
    lifecycleChart.setOption(lifecycleBaseOption);

    window.resizeCharts = function () {
        if (fuelChart && !fuelChart.isDisposed()) fuelChart.resize();
        if (lifecycleChart && !lifecycleChart.isDisposed()) lifecycleChart.resize();
    };

    renderStageStepper(SAMPLE_EVENT.stage_history, null);
    renderSummaryCards(SAMPLE_EVENT, true);

    let currentPage = 1;
    const itemsPerPage = 10;
    let allLogData = [];

    function renderPagination() {
        const tbody = document.getElementById('event-tbody');
        if (!tbody) return;
        const start = (currentPage - 1) * itemsPerPage;
        const end = start + itemsPerPage;
        const pageData = allLogData.slice(start, end);
        const totalPages = Math.ceil(allLogData.length / itemsPerPage) || 1;

        tbody.innerHTML = '';
        pageData.forEach(row => {
            let timeStr = shortTime(row.Timestamp);
            const statusBadge = row.PointStatus === 'thinking'
                ? '<span class="badge bg-light text-dark border px-2 py-1">Thinking…</span>'
                : `<span class="badge bg-${row.Prediction === 'Refuel' ? 'success' : row.Prediction === 'Fuel Theft' ? 'danger' : 'secondary'} px-2 py-1">${row.Prediction || '--'}</span>`;
            tbody.innerHTML += `<tr>
                <td>${timeStr}</td>
                <td><strong>${currentVehicle}</strong></td>
                <td>${row.FuelBefore?.toFixed(2) ?? '--'}</td>
                <td>${row.Fuel?.toFixed(2) ?? '--'}</td>
                <td>${(row.Change)?.toFixed(2) ?? '--'}</td>
                <td>${row.RegressionSlope?.toFixed(2) ?? '--'}</td>
                <td>${row.Confidence > 0 ? row.Confidence.toFixed(2) + '%' : 'N/A'}</td>
                <td>${statusBadge}</td>
                <td>${formatStopDurationMinutes(row.StopDuration)}</td>
            </tr>`;
        });

        document.getElementById('page-info').innerText = `Page ${currentPage} / ${totalPages}`;
        document.getElementById('btn-prev').disabled = currentPage === 1;
        document.getElementById('btn-next').disabled = currentPage === totalPages;
    }

    window.changePage = function (delta) {
        const totalPages = Math.ceil(allLogData.length / itemsPerPage) || 1;
        const newPage = currentPage + delta;
        if (newPage >= 1 && newPage <= totalPages) {
            currentPage = newPage;
            renderPagination();
        }
    };

    function computeDefaultZoom(totalPoints) {
        if (totalPoints <= DEFAULT_VISIBLE_POINTS) {
            return { start: 0, end: 100 };
        }
        const startPct = ((totalPoints - DEFAULT_VISIBLE_POINTS) / totalPoints) * 100;
        return { start: startPct, end: 100 };
    }

    function updateFuelYAxisMax(yData) {
        const maxVal = Math.max(0, ...yData);
        if (maxVal > fuelYAxisMax) {
            fuelYAxisMax = Math.ceil((maxVal * 1.05) / 50) * 50;
        }
        return fuelYAxisMax;
    }

    function applyZoom(option, xData) {
        const zoom = computeDefaultZoom(xData.length);
        option.dataZoom.forEach(dz => { dz.start = zoom.start; dz.end = zoom.end; });
        return option;
    }

    function renderFuelChart() {
        if (chartPoints.length === 0) return;

        const xData = chartPoints.map(p => p.timestamp);
        const yData = chartPoints.map(p => p.fuel);
        const speedData = chartPoints.map(p => p.speed);
        const states = chartPoints.map(p => p.label);
        const pointStatuses = chartPoints.map(p => p.point_status);

        const rawLike = chartPoints.map(p => ({
            timestamp: p.timestamp,
            fuel: p.fuel,
            speed: p.speed,
            label: p.label,
            prediction: p.label,
            point_status: p.point_status,
            confidence: p.confidence,
            boundary_point: p.boundary_point || false  // <-- Thêm dòng này
        }));

        const yMax = updateFuelYAxisMax(yData);
        const option = buildChartOption(xData, yData, speedData, states, rawLike, pointStatuses, yMax, 0);
        option.tooltip = {
            trigger: 'axis',
            backgroundColor: '#FFFFFF',
            borderColor: '#E5E7EB',
            textStyle: { color: '#111827' },
            formatter: customTooltipFormatter(rawLike, rawLike)
        };
        applyZoom(option, xData);
        fuelChart.setOption(option, true);

        const last = chartPoints[chartPoints.length - 1];
        document.getElementById('kpi-fuel').innerText = last.fuel?.toFixed(2) ?? '--';
        document.getElementById('kpi-speed').innerText = last.speed?.toFixed(2) ?? '--';
    }

    async function loadBaselineFuelChart(carId) {
        try {
            const res = await fetch(`${API_BASE}/api/car_data?car_id=${carId}&limit=${MAX_POINTS}`);
            const data = await res.json();
            if (!Array.isArray(data) || data.length === 0) return;

            chartPoints = data.map(d => ({
                timestamp: d.timestamp,
                fuel: d.fuel,
                speed: d.speed,
                label: d.label || 'Driving',
                point_status: 'normal'
            }));
            lastChartTimestamp = chartPoints[chartPoints.length - 1].timestamp;
            renderFuelChart();
        } catch (e) {
            console.error('Error fetching baseline car data:', e);
        }
    }

    function tsKey(ts) {
        const d = toDate(ts);
        return d ? d.getTime() : ts;
    }

    async function appendPredictedPoints() {
        try {
            const res = await fetch(`${API_BASE}/api/points?car_id=${currentVehicle}&limit=300&offset=0`);
            const payload = await res.json();
            const points = payload.points || [];
            if (points.length === 0) return false;

            const chartPointMap = new Map(chartPoints.map(p => [tsKey(p.timestamp), p]));
            let appended = false;
            let updatedCount = 0;
            let newPointCount = 0;

            const statusPriority = { 'normal': 0, 'thinking': 1, 'confirmed': 2 };

            for (const p of points) {
                const key = tsKey(p.timestamp);

                if (chartPointMap.has(key)) {
                    const existing = chartPointMap.get(key);
                    const newStatus = p.point_status || 'normal';
                    const oldStatus = existing.point_status;

                    if (statusPriority[newStatus] > statusPriority[oldStatus]) {
                        existing.point_status = newStatus;
                        updatedCount++;
                        appended = true;
                    }

                    if (p.prediction && newStatus === 'confirmed') {
                        existing.label = p.prediction;
                    }
                    if (p.confidence) existing.confidence = p.confidence;
                }
                else {
                    const newPoint = {
                        timestamp: p.timestamp,
                        fuel: p.fuel,
                        speed: p.speed,
                        label: p.prediction || 'Driving',
                        point_status: p.point_status || 'normal',
                        confidence: p.confidence || 0,
                        boundary_point: p.boundary_point || false  // <-- Thêm dòng này
                    };
                    chartPoints.push(newPoint);
                    chartPointMap.set(key, newPoint);
                    newPointCount++;
                    appended = true;
                }
            }

            chartPoints.sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));
            if (chartPoints.length > MAX_POINTS) {
                const targetSize = Math.floor(MAX_POINTS * 0.8);
                const removedCount = chartPoints.length - targetSize;
                if (removedCount > 0) {
                    chartPoints.splice(0, removedCount);
                }
            }

            if (chartPoints.length > 0) {
                lastChartTimestamp = chartPoints[chartPoints.length - 1].timestamp;
            }

            if (appended) {
                renderFuelChart();
            }
            return true;
        } catch (e) {
            console.error('Error fetching predicted points:', e);
            return false;
        }
    }

    function _sampleSixPoints(dataList) {
        if (!dataList || dataList.length === 0) return [];
        const len = dataList.length;
        if (len <= 6) return dataList;
        const selectedIndices = [0];
        const step = (len - 1) / 5;
        for (let i = 1; i < 5; i++) {
            selectedIndices.push(Math.round(i * step));
        }
        selectedIndices.push(len - 1);
        return selectedIndices.map(i => dataList[i]);
    }

    function updateLifecycleChart(trackerStatus) {
        const realHistory = trackerStatus && trackerStatus.confidence_history;
        const hasReal = realHistory && realHistory.length > 0;

        if (!hasReal) {
            if (!usingSampleLifecycle) {
                usingSampleLifecycle = true;
                const rawSample = SAMPLE_LIFECYCLE.confidence_history;
                const sampled = _sampleSixPoints(rawSample);
                lifecycleTimestamps = sampled.map(h => h.timestamp);
                lifecycleConfs = sampled.map(h => h.confidence);
                lifecycleLabels = sampled.map(h => h.label);

                const opt = {
                    title: { text: 'Event Lifecycle (dữ liệu mẫu)' },
                    xAxis: {
                        data: [...lifecycleTimestamps],
                        axisLabel: {
                            interval: 0,
                            fontSize: 10,
                            fontWeight: 'bold',
                            margin: 8,
                            color: '#4B5563',
                            formatter: (value) => shortTime(value)
                        }
                    },
                    series: [{
                        data: [...lifecycleConfs],
                        markArea: {
                            data: [[
                                { xAxis: lifecycleTimestamps[1], yAxis: 0, itemStyle: { color: 'rgba(22,163,74,0.12)' } },
                                { xAxis: lifecycleTimestamps[lifecycleTimestamps.length - 2], yAxis: 1 }
                            ]]
                        }
                    }]
                };
                lifecycleChart.setOption(opt, false);
                document.getElementById('lifecycle-sample-note').style.display = 'block';
                document.getElementById('lifecycle-live-pill').innerHTML = '<i class="bi bi-info-circle"></i> Sample';
            }
            return;
        }

        if (usingSampleLifecycle) {
            usingSampleLifecycle = false;
            lifecycleTimestamps = [];
            lifecycleConfs = [];
            lifecycleLabels = [];
            lifecycleChart.setOption(lifecycleBaseOption, true);
            document.getElementById('lifecycle-sample-note').style.display = 'none';
            document.getElementById('lifecycle-live-pill').innerHTML = '<i class="bi bi-lightning-charge"></i> Active tracking';
        }

        const sampledPoints = _getStateAwareSample(realHistory, trackerStatus.active_summary);
        lifecycleTimestamps = sampledPoints.map(p => p.timestamp);
        lifecycleConfs = sampledPoints.map(p => p.confidence);
        lifecycleLabels = sampledPoints.map(p => p.label);

        const newOption = {
            title: { text: 'Event Lifecycle' },
            xAxis: {
                data: [...lifecycleTimestamps],
                axisLabel: {
                    interval: 0,
                    fontSize: 10,
                    fontWeight: 'bold',
                    margin: 8,
                    color: '#4B5563',
                    formatter: (value) => shortTime(value)
                }
            },
            series: [{ data: [...lifecycleConfs], markArea: { data: [] } }]
        };

        if (trackerStatus.status === 'confirmed' || trackerStatus.status === 'finished') {
            newOption.series[0].markArea.data.push([
                { xAxis: lifecycleTimestamps[0], yAxis: 0, itemStyle: { color: 'rgba(22, 163, 74, 0.12)' } },
                { xAxis: lifecycleTimestamps[lifecycleTimestamps.length - 1], yAxis: 1 }
            ]);
        } else if (trackerStatus.status === 'candidate') {
            newOption.series[0].markArea.data.push([
                { xAxis: lifecycleTimestamps[0], yAxis: 0, itemStyle: { color: 'rgba(245, 158, 11, 0.10)' } },
                { xAxis: lifecycleTimestamps[lifecycleTimestamps.length - 1], yAxis: 1 }
            ]);
        }

        lifecycleChart.setOption(newOption, false);
    }

    function renderStageStepper(stageHistory, activeStage) {
        const stages = ['normal', 'candidate', 'confirmed', 'monitoring', 'finished'];
        const reached = {};
        (stageHistory || []).forEach(s => { reached[s.stage] = s.timestamp; });

        stages.forEach(stage => {
            const el = document.querySelector(`.stage-step[data-stage="${stage}"]`);
            const timeEl = document.querySelector(`.stage-time[data-time="${stage}"]`);
            if (!el) return;

            el.classList.remove('is-active', 'is-done', 'is-pending',
                'is-active-normal', 'is-active-candidate', 'is-active-confirmed', 'is-active-monitoring', 'is-active-finished',
                'is-done-normal', 'is-done-candidate', 'is-done-confirmed', 'is-done-monitoring', 'is-done-finished');

            if (reached[stage]) {
                if (stage === activeStage) {
                    el.classList.add(`is-active-${stage}`, 'is-active');
                } else {
                    el.classList.add(`is-done-${stage}`, 'is-done');
                }
                if (timeEl) timeEl.innerText = shortTime(reached[stage]);
            } else {
                el.classList.add('is-pending');
                if (timeEl) timeEl.innerText = '--';
            }
        });
    }

    function computeDurationSeconds(startTime) {
        if (!startTime) return 0;
        const now = new Date();
        const start = new Date(startTime);
        const seconds = Math.floor((now - start) / 1000);
        return Math.max(0, seconds);
    }

    function formatHumanDuration(seconds) {
        if (!seconds || seconds === 0) return '--';
        if (seconds < 0) seconds = 0;
        const days = Math.floor(seconds / 86400);
        const hours = Math.floor((seconds % 86400) / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        const secs = Math.floor(seconds % 60);

        let parts = [];
        if (days > 0) parts.push(`${days}d`);
        if (hours > 0) parts.push(`${hours}h`);
        if (minutes > 0) parts.push(`${minutes}m`);
        if (secs > 0 || parts.length === 0) parts.push(`${secs}s`);

        return parts.join(' ');
    }

    function renderSummaryCards(ev, isSample) {
        let confidenceDisplay = '--';
        if (ev.max_confidence != null) {
            let val = ev.max_confidence;
            if (val <= 1 && val > 0) val = val * 100;
            confidenceDisplay = `${val.toFixed?.(2) ?? val} %`;
        } else if (ev.confidence != null) {
            let val = ev.confidence;
            if (val <= 1 && val > 0) val = val * 100;
            confidenceDisplay = `${val.toFixed?.(2) ?? val} %`;
        }

        document.getElementById('ls-event-type').innerText = ev.event_type || '--';
        document.getElementById('ls-start-time').innerText = shortTime(ev.start_time);

        const isOngoing = !ev.end_time;
        document.getElementById('ls-end-time').innerText = isOngoing ? '(đang diễn ra)' : shortTime(ev.end_time);

        let durationDisplay = '--';
        if (ev.duration_s != null && ev.duration_s > 0) {
            durationDisplay = formatHumanDuration(ev.duration_s) + (isOngoing ? ' (đang diễn ra)' : '');
        }
        document.getElementById('ls-duration').innerText = durationDisplay;

        let fuelChangeDisplay = '--';
        if (!isSample) {
            const delta = ev.delta_fuel ?? (ev.fuel_added ?? 0);

            if (!isOngoing && ev.fuel_added != null) {
                const sign = ev.fuel_added > 0 ? '+' : '';
                fuelChangeDisplay = `${sign}${parseFloat(ev.fuel_added).toFixed(2)} L`;
            }
            else if (isOngoing && ev.fuel_start != null && ev.fuel_current != null) {
                const deltaVal = delta;
                const sign = deltaVal > 0 ? '+' : '';

                let statusLabel = '';
                if (ev.event === 'Idle') {
                    statusLabel = 'Đứng yên';
                } else {
                    switch (ev.fuel_status) {
                        case 'Refueling': statusLabel = 'Đang nạp'; break;
                        case 'Theft': statusLabel = 'Bị mất cắp'; break;
                        case 'Consuming': statusLabel = 'Tiêu hao'; break;
                        default: statusLabel = 'Đứng yên'; break;
                    }
                }
                fuelChangeDisplay = `${sign}${deltaVal.toFixed(2)} L (${statusLabel})`;
            }
        }
        document.getElementById('ls-fuel-change').innerText = fuelChangeDisplay;

        document.getElementById('ls-max-confidence').innerText = confidenceDisplay;
        document.getElementById('lifecycle-summary').classList.toggle('is-sample', !!isSample);
    }

    async function updateLifecycleDetails(trackerStatus) {
        if (trackerStatus.active_summary) {
            const s = trackerStatus.active_summary;
            renderStageStepper(s.stage_history, s.stage || trackerStatus.status);
            renderSummaryCards({
                event_type: s.event_type,
                start_time: s.start_time,
                end_time: null,
                duration_s: s.duration_s,
                fuel_start: s.fuel_start,
                fuel_current: s.fuel_current,
                delta_fuel: s.delta_fuel,
                fuel_status: s.fuel_status,
                confidence: s.confidence
            }, false);
            document.getElementById('lifecycle-sample-note').style.display = 'none';
            return;
        }

        try {
            const res = await fetch(`${API_BASE}/api/events?car_id=${currentVehicle}`);
            const events = await res.json();
            if (Array.isArray(events) && events.length > 0) {
                const last = events[events.length - 1];
                renderStageStepper(last.stage_history, null);
                renderSummaryCards({
                    event_type: last.state,
                    start_time: last.start_time,
                    end_time: last.end_time,
                    duration_s: last.duration_s,
                    fuel_added: last.fuel_added,
                    max_confidence: last.max_confidence
                }, false);
                document.getElementById('lifecycle-sample-note').style.display = 'none';
                return;
            }
        } catch (e) {
            console.error('Error fetching events for lifecycle summary:', e);
        }

        if (trackerStatus.confidence_history && trackerStatus.confidence_history.length > 0) {
            const startTimestamp = trackerStatus.confidence_history[0].timestamp;
            const normalEvent = {
                event_type: 'Driving',
                start_time: startTimestamp,
                end_time: null,
                duration_s: null,
                fuel_added: null,
                fuel_start: null,
                fuel_current: null,
                max_confidence: 100
            };
            const realHistory = [{ stage: 'normal', timestamp: startTimestamp }];
            renderStageStepper(realHistory, 'normal');
            renderSummaryCards(normalEvent, false);
            document.getElementById('lifecycle-sample-note').style.display = 'none';
            return;
        }

        renderStageStepper(SAMPLE_EVENT.stage_history, null);
        renderSummaryCards(SAMPLE_EVENT, true);
        document.getElementById('lifecycle-sample-note').style.display = 'block';
    }

    function syncCurrentVehicle() {
        const sel = document.getElementById('vehicle-selector');
        if (sel) {
            const newCar = sel.value;
            if (newCar !== currentVehicle) {
                currentVehicle = newCar;
                document.getElementById('sidebar-vehicle').innerText = currentVehicle + ' • Active';

                lifecycleTimestamps = [];
                lifecycleConfs = [];
                lifecycleLabels = [];
                usingSampleLifecycle = false;
                lifecycleChart.setOption(lifecycleBaseOption, true);

                chartPoints = [];
                lastChartTimestamp = null;
                fuelYAxisMax = FUEL_Y_FLOOR;
            }
        }
    }

    async function updateKPIs() {
        try {
            const res = await fetch(`${API_BASE}/api/dashboard?car_id=${currentVehicle}`);
            const data = await res.json();
            document.getElementById('kpi-slope') && (document.getElementById('kpi-slope').innerText = data.slope?.toFixed(2) ?? '--');
            document.getElementById('kpi-confidence') && (document.getElementById('kpi-confidence').innerText = data.confidence ? data.confidence.toFixed(2) + '%' : 'N/A');
            document.getElementById('kpi-stop') && (document.getElementById('kpi-stop').innerText = formatStopDurationMinutes(data.stop_duration));
            document.getElementById('kpi-alerts') && (document.getElementById('kpi-alerts').innerText = data.alerts || 0);
            const statusLabel = data.point_status === 'thinking' ? `${data.status} (thinking…)` : (data.status || 'Idle');
            const badgeClass = data.point_status === 'thinking' ? 'bg-light text-dark border' : 'bg-secondary';
            document.getElementById('kpi-status').innerHTML = `<span class="badge ${badgeClass} px-3 py-2">${statusLabel}</span>`;
        } catch (e) {
            console.error('Error updating KPIs:', e);
        }
    }

    function _getStateAwareSample(confidenceHistory, activeSummary) {
        if (!confidenceHistory || confidenceHistory.length === 0) return [];
        let stateStartPoint = null;
        if (activeSummary && activeSummary.start_time) {
            const startTimeMs = new Date(activeSummary.start_time).getTime();
            let found = confidenceHistory.find(p => new Date(p.timestamp).getTime() >= startTimeMs);
            if (found) stateStartPoint = found;
        }
        if (!stateStartPoint && confidenceHistory.length > 0) {
            stateStartPoint = confidenceHistory[0];
        }
        let lastFivePoints = confidenceHistory.slice(-5);
        const pointMap = new Map();
        if (stateStartPoint) {
            pointMap.set(stateStartPoint.timestamp, stateStartPoint);
        }
        for (const p of lastFivePoints) {
            if (!pointMap.has(p.timestamp)) {
                pointMap.set(p.timestamp, p);
            }
        }
        const finalPoints = Array.from(pointMap.values());
        finalPoints.sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));
        return finalPoints;
    }

    async function fetchAll() {
        try {
            syncCurrentVehicle();

            if (chartPoints.length === 0) {
                await loadBaselineFuelChart(currentVehicle);
            }
            await appendPredictedPoints();

            const trackerRes = await fetch(`${API_BASE}/api/tracker_status?car_id=${currentVehicle}`);
            const trackerStatus = await trackerRes.json();

            updateLifecycleChart(trackerStatus);
            await updateLifecycleDetails(trackerStatus);
            await updateKPIs();

            const historyRes = await fetch(`${API_BASE}/api/history?car_id=${currentVehicle}`);
            const history = await historyRes.json();
            if (Array.isArray(history)) {
                allLogData = history.slice().map((d, idx, arr) => {
                    const prevRaw = idx > 0 ? arr[idx - 1].raw : null;
                    const fuelBefore = prevRaw ? prevRaw.Fuel : d.raw.Fuel;
                    return {
                        Timestamp: d.raw.Timestamp,
                        Fuel: d.raw.Fuel,
                        FuelBefore: fuelBefore,
                        Change: d.raw.Fuel - fuelBefore,
                        Prediction: d.processed.Prediction,
                        PointStatus: d.processed.PointStatus,
                        RegressionSlope: d.processed.RegressionSlope,
                        StopDuration: d.processed.StopDuration,
                        Confidence: d.processed.Confidence
                    };
                }).reverse();
                currentPage = 1;
                renderPagination();
            }
        } catch (e) {
            console.error("Fetch error:", e);
        }
    }

    loadBaselineFuelChart(currentVehicle);
    fetchAll();

    let fetchTimer = null;
    const originalFetchAll = fetchAll;
    fetchAll = function () {
        if (fetchTimer) clearTimeout(fetchTimer);
        originalFetchAll.call(this);
    };
    setInterval(fetchAll, 1000);

    window.addEventListener('resize', () => {
        if (window.resizeCharts) window.resizeCharts();
    });

    setTimeout(() => {
        if (window.resizeCharts) window.resizeCharts();
    }, 100);
});