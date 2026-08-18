/**
 * Module: Xử lý phân đoạn dữ liệu (Event Segmentation)
 * =====================================================================
 * Logic gốc chính xác: Trả lại startIdx = validStates[i].index.
 * =====================================================================
 */
export function computeSegmentStatistics(seg, x_data, y_data) {
    const startIdx = seg.startIdx;
    const endIdx = seg.endIdx;

    const startValue = y_data[startIdx];
    const endValue = y_data[endIdx];

    const deltaValue = endValue - startValue;

    const durationSec =
        (new Date(x_data[endIdx]) -
         new Date(x_data[startIdx])) / 1000;

    const rate =
        durationSec !== 0
            ? deltaValue / durationSec
            : 0;

    const midIdx =
        Math.floor((startIdx + endIdx) / 2);

    return {
        ...seg,

        duration: durationSec,

        startValue,
        endValue,

        deltaValue,
        rate,

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
                endIdx: validStates[i - 1].index,
                startTime: x_data[startIdx],
                endTime: x_data[validStates[i - 1].index]
            });
            currentState = validStates[i].state;
            // Segment mới bắt đầu tại đúng index của trạng thái mới
            startIdx = validStates[i].index; 
        }
    }

    if (startIdx !== undefined && startIdx < x_data.length) {
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

export function findThinkingSegments(x_data, pointStatuses) {
    if (!pointStatuses || pointStatuses.length === 0) return [];

    const segments = [];
    let curStart = null;

    for (let i = 0; i < pointStatuses.length; i++) {
        const isThinking = pointStatuses[i] === 'thinking';

        if (isThinking && curStart === null) {
            curStart = i;
        }

        if (!isThinking && curStart !== null) {
            const endIdx = i - 1;
            segments.push({
                startIdx: curStart,
                endIdx: endIdx,
                startTime: x_data[curStart],
                endTime: x_data[endIdx],
                midIndex: Math.floor((curStart + endIdx) / 2)
            });
            curStart = null;
        }
    }

    if (curStart !== null) {
        const endIdx = pointStatuses.length - 1;
        segments.push({
            startIdx: curStart,
            endIdx: endIdx,
            startTime: x_data[curStart],
            endTime: x_data[endIdx],
            midIndex: Math.floor((curStart + endIdx) / 2)
        });
    }

    return segments;
}

export function findEvents(segments, x_data, y_data) {
    const events = [];
    segments.forEach((seg) => {
        if (typeof seg.startIdx !== 'number' || typeof seg.endIdx !== 'number') return;
        if (seg.state === 'Refuel' || seg.state === 'Theft' || seg.state === 'Fuel Theft') {
            events.push({
                state: seg.state,
                startIdx: seg.startIdx,
                endIdx: seg.endIdx,
                startTime: seg.startTime,
                endTime: seg.endTime,
                startValue: seg.startValue,
                endValue: seg.endValue,
                delta: seg.deltaValue,
                duration: seg.duration,
                rate: seg.rate
            });
        }
    });
    return events;
}