/**
 * Module: Xử lý phân đoạn dữ liệu (Event Segmentation)
 * Giữ nguyên logic ổn định
 */
export function computeSegmentStatistics(seg, x_data, y_data) {
    const startIdx = seg.startIdx;
    const endIdx = seg.endIdx;
    const startFuel = y_data[startIdx];
    const endFuel = y_data[endIdx];
    const deltaFuel = endFuel - startFuel;
    const durationSec = (new Date(x_data[endIdx]) - new Date(x_data[startIdx])) / 1000;
    const rate = durationSec !== 0 ? deltaFuel / durationSec : 0;
    const midIdx = Math.floor((startIdx + endIdx) / 2);
    
    return {
        ...seg,
        duration: durationSec,
        startFuel: startFuel,
        endFuel: endFuel,
        deltaFuel: deltaFuel,
        rate: rate,
        midIndex: midIdx,
        midTime: x_data[midIdx]
    };
}

export function findSegments(x_data, y_data, states) {
    if (!states || states.length === 0) return [];
    
    const validStates = states
        .map((state, index) => ({ state, index }))
        .filter(item => item.state && item.state !== 'undefined');
    
    if (validStates.length === 0) return [];

    const segments = [];
    let currentState = validStates[0].state;
    let startIdx = validStates[0].index;
    
    for (let i = 1; i < validStates.length; i++) {
        if (validStates[i].state !== currentState) {
            segments.push({
                state: currentState,
                startIdx: startIdx,
                endIdx: validStates[i-1].index,
                startTime: x_data[startIdx],
                endTime: x_data[validStates[i-1].index]
            });
            currentState = validStates[i].state;
            startIdx = validStates[i].index;
        }
    }
    
    if (startIdx !== undefined) {
        segments.push({
            state: currentState,
            startIdx: startIdx,
            endIdx: validStates[validStates.length - 1].index,
            startTime: x_data[startIdx],
            endTime: x_data[validStates[validStates.length - 1].index]
        });
    }

    return segments.map(seg => computeSegmentStatistics(seg, x_data, y_data));
}

export function findEvents(segments, x_data, y_data) {
    const events = [];
    segments.forEach((seg) => {
        if (typeof seg.startIdx !== 'number' || typeof seg.endIdx !== 'number') return;
        if (seg.state === 'Refuel' || seg.state === 'Fuel Theft') {
            events.push({
                state: seg.state,
                startIdx: seg.startIdx,
                endIdx: seg.endIdx,
                startTime: seg.startTime,
                endTime: seg.endTime,
                startValue: seg.startFuel,
                endValue: seg.endFuel,
                delta: seg.deltaFuel,
                duration: seg.duration,
                rate: seg.rate
            });
        }
    });
    return events;
}