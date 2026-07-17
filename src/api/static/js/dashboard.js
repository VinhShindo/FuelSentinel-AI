import { buildChartOption } from './chartBuilder.js';
import { customTooltipFormatter } from './tooltip.js';

const API_BASE = window.location.origin;

document.addEventListener('DOMContentLoaded', () => {
    const chartDom = document.getElementById('fuel-chart');
    const chart = echarts.init(chartDom, null, { renderer: 'canvas' });
    
    const MAX_POINTS = 100;
    const STATE_BADGES = {
        'Idle': 'secondary', 'Driving': 'dark', 'Refuel': 'success', 'Fuel Theft': 'danger'
    };

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
            const badge = STATE_BADGES[row.Prediction] || 'secondary';
            // Trích xuất giờ:phút an toàn
            let timeStr = '--';
            if (row.Timestamp) {
                if (row.Timestamp.includes('T')) {
                    timeStr = row.Timestamp.split('T')[1]?.substring(0,8) || '--';
                } else if (row.Timestamp.includes(' ')) {
                    timeStr = row.Timestamp.split(' ')[1]?.substring(0,8) || '--';
                } else {
                    timeStr = row.Timestamp.substring(0,8);
                }
            }
            tbody.innerHTML += `<tr>
                <td>${timeStr}</td>
                <td><strong>VN001</strong></td>
                <td>${row.FuelBefore?.toFixed(1) ?? '--'}</td>
                <td>${row.Fuel?.toFixed(1) ?? '--'}</td>
                <td>${(row.Change)?.toFixed(1) ?? '--'}</td>
                <td>${row.RegressionSlope?.toFixed(4) ?? '--'}</td>
                <td>${row.Confidence > 0 ? row.Confidence.toFixed(1) + '%' : 'N/A'}</td>
                <td><span class="badge bg-${badge} px-2 py-1">${row.Prediction || '--'}</span></td>
                <td>${row.StopDuration ?? 0}s</td>
            </tr>`;
        });

        const pageInfo = document.getElementById('page-info');
        const btnPrev = document.getElementById('btn-prev');
        const btnNext = document.getElementById('btn-next');
        if(pageInfo) pageInfo.innerText = `Page ${currentPage} / ${totalPages}`;
        if(btnPrev) btnPrev.disabled = currentPage === 1;
        if(btnNext) btnNext.disabled = currentPage === totalPages;
    }

    window.changePage = function(delta) {
        const totalPages = Math.ceil(allLogData.length / itemsPerPage) || 1;
        const newPage = currentPage + delta;
        if (newPage >= 1 && newPage <= totalPages) {
            currentPage = newPage;
            renderPagination();
        }
    };

    function updateDashboard(data) {
        if (!Array.isArray(data) || data.length === 0) {
            console.warn("Dashboard data is empty");
            return;
        }
        
        const validData = data.filter(d => d && d.raw && typeof d.raw.Fuel === 'number');
        if (validData.length === 0) return;

        const xData = validData.map(d => d.raw.Timestamp);
        const yData = validData.map(d => d.raw.Fuel);
        const states = validData.map(d => d.processed ? d.processed.Prediction : 'Idle');

        const option = buildChartOption(xData, yData, states, validData);
        option.tooltip = {
            trigger: 'axis',
            backgroundColor: '#1f2937',
            borderColor: '#1f2937',
            textStyle: { color: '#fff' },
            formatter: customTooltipFormatter(validData, validData)
        };
        chart.setOption(option, true);
        chart.resize();

        // Legend
        const legendEl = document.getElementById('explain-legend');
        if(legendEl) {
            legendEl.innerHTML = `
                <div class="legend-row fw-bold border-bottom pb-2 mb-1"><span style="background:#2563EB;"></span> Fuel Level</div>
                <div class="legend-row"><span style="background:#E5E7EB;"></span> Idle (Speed = 0)</div>
                <div class="legend-row"><span style="background:#FEF08A;"></span> Driving (Speed > 0)</div>
                <div class="legend-row"><span style="background:rgba(34, 197, 94, 0.4);"></span> Refuel (Fuel tăng)</div>
                <div class="legend-row"><span style="background:rgba(239, 68, 68, 0.4);"></span> Fuel Theft (Fuel giảm)</div>
                <div class="legend-row fw-bold border-bottom pb-2 mt-2 mb-1"><span style="background:#000; width:10px; height:4px;"></span> Boundary Marker</div>
                <div class="legend-row text-muted small fst-italic">Event Pin Start/End</div>
            `;
        }

        // Cập nhật các KPI và panels
        const last = validData[validData.length - 1];
        const lastProcessed = last.processed || {};
        const lastRaw = last.raw || {};
        
        const rulePanel = document.getElementById('rule-panel');
        if(rulePanel) {
            rulePanel.innerHTML = `
                <div class="rule-section">1. Current Window</div>
                <div class="row g-2 small mb-2">
                    <div class="col-6"><span class="text-muted">Speed:</span> <strong>${lastRaw.Speed ?? 0} km/h</strong></div>
                    <div class="col-6"><span class="text-muted">Duration:</span> <strong>${lastProcessed.StopDuration ?? 0}s</strong></div>
                    <div class="col-6"><span class="text-muted">Fuel Δ:</span> <strong>${lastProcessed.FuelDiff ?? 0} L</strong></div>
                    <div class="col-6"><span class="text-muted">Slope:</span> <strong>${(lastProcessed.RegressionSlope ?? 0).toFixed(4)}</strong></div>
                    <div class="col-6"><span class="text-muted">Decision:</span> <strong class="text-${STATE_BADGES[lastProcessed.Prediction] || 'secondary'}">${lastProcessed.Prediction || 'Idle'}</strong></div>
                    <div class="col-12"><span class="text-muted">Confidence:</span> <span style="color: #16A34A;">${lastProcessed.Confidence > 0 ? lastProcessed.Confidence.toFixed(1) + '%' : 'N/A'}</span></div>
                </div>
            `;
        }

        const updateEl = (id, val) => { const el = document.getElementById(id); if(el) el.innerText = val; };
        updateEl('kpi-vehicle', 'VN001');
        updateEl('kpi-fuel', lastRaw.Fuel?.toFixed(2) ?? '--');
        updateEl('kpi-speed', lastRaw.Speed ?? '--');
        updateEl('kpi-slope', (lastProcessed.RegressionSlope ?? 0).toFixed(4));
        updateEl('kpi-confidence', lastProcessed.Confidence > 0 ? lastProcessed.Confidence.toFixed(1) + '%' : 'N/A');
        updateEl('kpi-stop', (lastProcessed.StopDuration ?? 0) + 's');
        updateEl('kpi-alerts', validData.filter(d => d.processed && ['Refuel', 'Fuel Theft'].includes(d.processed.Prediction)).length);
        const statusEl = document.getElementById('kpi-status');
        if(statusEl) statusEl.innerHTML = `<span class="badge bg-secondary px-3 py-2">${lastProcessed.Prediction || 'Idle'}</span>`;
        const panelDecision = document.getElementById('panel-decision');
        if(panelDecision) panelDecision.innerHTML = `<span class="badge bg-${STATE_BADGES[lastProcessed.Prediction] || 'secondary'} p-2 fs-6">${lastProcessed.Prediction || 'Idle'}</span>`;

        // ===== SỬA CÁCH XÂY DỰNG allLogData =====
        allLogData = validData.slice().map((d, idx, arr) => {
            const prevRaw = idx > 0 ? arr[idx-1].raw : null;
            const fuelBefore = prevRaw ? prevRaw.Fuel : d.raw.Fuel;
            const change = d.raw.Fuel - fuelBefore;
            return {
                Timestamp: d.raw.Timestamp,
                Fuel: d.raw.Fuel,
                FuelBefore: fuelBefore,
                Change: change,
                Prediction: d.processed ? d.processed.Prediction : 'Idle',
                RegressionSlope: d.processed ? d.processed.RegressionSlope : 0,
                StopDuration: d.processed ? (d.processed.StopDuration ?? 0) : 0,
                Confidence: d.processed ? (d.processed.Confidence ?? 0) : 0
            };
        }).reverse();   // Mới nhất lên đầu
        currentPage = 1;
        renderPagination();
    }

    async function init() {
        try {
            const res = await fetch(`${API_BASE}/api/history`);
            if (!res.ok) throw new Error('Failed to fetch history');
            const history = await res.json();
            window.localData = history; 
            updateDashboard(history);
        } catch (e) {
            console.error("Init error:", e);
            setTimeout(init, 1000);
        }
    }
    init();

    setInterval(async () => {
        try {
            const res = await fetch(`${API_BASE}/api/realtime`);
            if (!res.ok) return;
            const newData = await res.json();
            
            if (!window.localData) window.localData = [];
            const entry = {
                raw: newData.raw,
                processed: newData.processed
            };
            window.localData.push(entry);
            if (window.localData.length > MAX_POINTS) {
                window.localData.shift();
            }
            updateDashboard(window.localData);
        } catch (e) {
            console.error("Realtime error:", e);
        }
    }, 1000);

    window.addEventListener('resize', () => { chart.resize(); });
});