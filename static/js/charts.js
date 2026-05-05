/**
 * charts.js — Chart.js Visualizations
 * Creates all charts: ranking bars, scenario breakdown, model comparison.
 */

const CHART_COLORS = {
    green: '#00d2d3',
    yellow: '#feca57',
    orange: '#ff9f43',
    red: '#ff6b6b',
    deepRed: '#ff0844',
    blue: '#54a0ff',
    accent: '#6c5ce7',
    accentLight: '#a29bfe',
    textMuted: '#9090b0',
    gridColor: 'rgba(255,255,255,0.06)',
};

const CHART_DEFAULTS = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
        legend: { labels: { color: CHART_COLORS.textMuted, font: { family: "'Inter', sans-serif", size: 12 } } },
    },
    scales: {
        x: { ticks: { color: CHART_COLORS.textMuted, font: { size: 11 } }, grid: { color: CHART_COLORS.gridColor } },
        y: { ticks: { color: CHART_COLORS.textMuted, font: { size: 11 } }, grid: { color: CHART_COLORS.gridColor } },
    },
};

function getBarColor(score) {
    if (score >= 85) return CHART_COLORS.deepRed;
    if (score >= 70) return CHART_COLORS.red;
    if (score >= 50) return CHART_COLORS.orange;
    if (score >= 25) return CHART_COLORS.yellow;
    return CHART_COLORS.green;
}

// Store chart instances for cleanup
const chartInstances = {};

function destroyChart(id) {
    if (chartInstances[id]) {
        chartInstances[id].destroy();
        delete chartInstances[id];
    }
}

// ── Ranking Bar Chart ──
function createRankingChart(events) {
    destroyChart('ranking');
    const ctx = document.getElementById('ranking-chart');
    if (!ctx) return;

    const labels = events.map((e, i) => `#${e.rank} ${(e.headline || '').substring(0, 30)}...`);
    const scores = events.map(e => e.impact_score);
    const colors = scores.map(s => getBarColor(s));

    chartInstances['ranking'] = new Chart(ctx, {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                label: 'Impact Score',
                data: scores,
                backgroundColor: colors.map(c => c + '88'),
                borderColor: colors,
                borderWidth: 2,
                borderRadius: 6,
                barPercentage: 0.7,
            }],
        },
        options: {
            ...CHART_DEFAULTS,
            indexAxis: 'y',
            plugins: {
                ...CHART_DEFAULTS.plugins,
                legend: { display: false },
            },
            scales: {
                x: { ...CHART_DEFAULTS.scales.x, max: 100, title: { display: true, text: 'Impact Score', color: CHART_COLORS.textMuted } },
                y: { ...CHART_DEFAULTS.scales.y, ticks: { color: CHART_COLORS.textMuted, font: { size: 10 } } },
            },
        },
    });
}

// ── Scenario Chart ──
function createScenarioChart(scenario) {
    destroyChart('scenario');
    const ctx = document.getElementById('scenario-chart');
    if (!ctx) return;

    const individual = scenario.individual_scores || [];
    const labels = individual.map((_, i) => `Event ${i + 1}`);
    labels.push('Compound');

    const data = [...individual, scenario.compound_score];
    const colors = data.map((s, i) => i === data.length - 1 ? CHART_COLORS.accent : getBarColor(s));

    chartInstances['scenario'] = new Chart(ctx, {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                label: 'Impact Score',
                data,
                backgroundColor: colors.map(c => c + '88'),
                borderColor: colors,
                borderWidth: 2,
                borderRadius: 6,
                barPercentage: 0.6,
            }],
        },
        options: {
            ...CHART_DEFAULTS,
            plugins: { ...CHART_DEFAULTS.plugins, legend: { display: false } },
            scales: {
                x: { ...CHART_DEFAULTS.scales.x },
                y: { ...CHART_DEFAULTS.scales.y, max: 100, title: { display: true, text: 'Impact Score', color: CHART_COLORS.textMuted } },
            },
        },
    });
}

// ── Model Comparison Chart ──
function createComparisonChart(lr, nn) {
    destroyChart('compare');
    const ctx = document.getElementById('compare-chart');
    if (!ctx) return;

    const metrics = ['accuracy', 'precision', 'recall', 'f1_score'];
    const labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score'];

    chartInstances['compare'] = new Chart(ctx, {
        type: 'bar',
        data: {
            labels,
            datasets: [
                {
                    label: 'Logistic Regression',
                    data: metrics.map(m => ((lr[m] || 0) * 100).toFixed(1)),
                    backgroundColor: CHART_COLORS.blue + '88',
                    borderColor: CHART_COLORS.blue,
                    borderWidth: 2,
                    borderRadius: 6,
                    barPercentage: 0.5,
                },
                {
                    label: 'Neural Network (MLP)',
                    data: metrics.map(m => ((nn[m] || 0) * 100).toFixed(1)),
                    backgroundColor: CHART_COLORS.accent + '88',
                    borderColor: CHART_COLORS.accent,
                    borderWidth: 2,
                    borderRadius: 6,
                    barPercentage: 0.5,
                },
            ],
        },
        options: {
            ...CHART_DEFAULTS,
            scales: {
                x: { ...CHART_DEFAULTS.scales.x },
                y: { ...CHART_DEFAULTS.scales.y, max: 100, title: { display: true, text: 'Score (%)', color: CHART_COLORS.textMuted } },
            },
        },
    });
}
