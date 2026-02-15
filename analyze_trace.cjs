
const fs = require('fs');

const tracePath = 'trace/trace.json';
if (!fs.existsSync(tracePath)) {
    console.error('Trace file not found:', tracePath);
    process.exit(1);
}

const rawData = fs.readFileSync(tracePath);
const trace = JSON.parse(rawData);
const events = trace.traceEvents || trace; // Handle different formats

const stats = {};

events.forEach(event => {
    if (event.dur && event.name) {
        if (!stats[event.name]) {
            stats[event.name] = { count: 0, totalDur: 0 };
        }
        stats[event.name].count++;
        stats[event.name].totalDur += event.dur;
    }
});

const sortedStats = Object.entries(stats)
    .sort(([, a], [, b]) => b.totalDur - a.totalDur)
    .slice(0, 20);

console.log('Top 20 most time-consuming functions:');
console.table(sortedStats.map(([name, data]) => ({
    name,
    count: data.count,
    totalDurMs: (data.totalDur / 1000).toFixed(2),
    avgDurMs: (data.totalDur / data.count / 1000).toFixed(2)
})));
