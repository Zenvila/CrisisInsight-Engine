/**
 * app.js — Main Application Logic
 * Handles all UI interactions, API calls, and result rendering.
 */

const API = {
    predict: '/api/predict',
    predictBatch: '/api/predict-batch',
    rank: '/api/rank',
    scenario: '/api/scenario',
    riskPath: '/api/risk-path',
    compare: '/api/models/compare',
    explain: '/api/explainability',
    samples: '/api/samples',
};

const SAMPLES = [
    "Major earthquake strikes coastal city, thousands displaced and critical infrastructure destroyed",
    "Global stock markets plunge amid fears of widespread economic recession and banking instability",
    "Military forces launch large-scale offensive near disputed border region, ceasefire collapses",
    "Massive data breach exposes millions of user records from major financial technology platform",
    "Category 5 hurricane approaches populated coastline, mandatory evacuations ordered across three states",
    "Trade war escalates sharply as new tariffs imposed on semiconductor and energy imports",
    "Ceasefire negotiations collapse as renewed fighting reported across multiple front lines",
    "Ransomware attack cripples hospital and banking systems across multiple countries",
    "Devastating floods submerge entire districts after record monsoon rainfall hits region",
    "Central bank raises interest rates to historic high to combat spiraling inflation crisis",
    "Coordinated drone strikes damage major energy terminals near strategic port",
    "National payment network outage disrupts banking transactions for 14 hours",
    "Extreme heatwave pushes power demand to record levels across urban centers",
    "Government declares emergency after cyberattack disables air traffic systems",
    "Food supply shortages intensify as drought conditions spread across farmland",
    "Currency drops sharply after sovereign debt downgrade and investor selloff",
    "Port closures delay essential imports after severe cyclone impact",
    "Railway signaling failure halts intercity transport during evacuation efforts",
    "Large wildfire front threatens telecom towers and emergency command centers",
    "Hospitals report medicine shortages as logistics chain remains disrupted",
];

const SAMPLE_QUERIES = [
    {
        label: 'Critical War',
        text: 'Nuclear missile launched targeting major capital city',
    },
    {
        label: 'High Economy',
        text: 'Oil prices spike following Middle East conflict',
    },
    {
        label: 'Medium Technology',
        text: 'Cyber attack hits government infrastructure',
    },
    {
        label: 'Low Technology',
        text: 'New technology product launched by startup',
    },
    {
        label: 'Critical Disaster',
        text: 'Category 5 cyclone destroys coastal hospitals and power grid overnight',
    },
    {
        label: 'High Economy Crash',
        text: 'Stock exchange halts trading after sudden 11 percent market collapse',
    },
    {
        label: 'High Cyber Threat',
        text: 'Ransomware attack locks national health records and emergency dispatch systems',
    },
    {
        label: 'Medium War Alert',
        text: 'Border troops mobilize after repeated ceasefire violations near disputed zone',
    },
];

const SAMPLE_MULTI_QUERIES = [
    {
        label: 'War + Economy',
        texts: [
            'Border conflict escalates after missile strikes',
            'Oil prices spike following regional tensions',
            'Stock markets fall amid fear of recession',
        ],
    },
    {
        label: 'Disaster + Tech',
        texts: [
            'Major earthquake strikes coastal city',
            'Cyber attack disrupts emergency response systems',
            'Cloud outage affects hospitals and transport',
        ],
    },
    {
        label: 'Flood + Inflation',
        texts: [
            'Monsoon floods cut off key agricultural regions',
            'Food prices jump 18 percent in wholesale markets',
            'Emergency imports announced to stabilize supply',
        ],
    },
    {
        label: 'Cyber + Finance',
        texts: [
            'Core banking servers hit by coordinated malware attack',
            'Digital payment channels fail during peak business hours',
            'Public queues surge as cash withdrawals are restricted',
        ],
    },
    {
        label: 'War + Refugee',
        texts: [
            'Heavy shelling reported near border settlements',
            'Thousands displaced toward temporary refugee camps',
            'Aid agencies request urgent cross-border assistance',
        ],
    },
    {
        label: 'Energy + Logistics',
        texts: [
            'Pipeline explosion reduces national fuel output',
            'Truck delivery costs spike due to diesel shortage',
            'Medical supply distribution delayed across provinces',
        ],
    },
];

const SAMPLE_SCENARIO_QUERIES = [
    {
        label: 'Triple Crisis',
        texts: [
            'War breaks out near border region',
            'Global inflation rises after supply shock',
            'Hospitals hit by ransomware attack',
        ],
    },
    {
        label: 'Climate + Economy',
        texts: [
            'Severe floods displace thousands of families',
            'Food prices rise sharply across the region',
            'Emergency funding announced by the government',
        ],
    },
    {
        label: 'Energy Shock Chain',
        texts: [
            'Pipeline explosion disrupts regional fuel supply',
            'Power outages spread across major industrial zones',
            'Manufacturing slowdown triggers export losses',
        ],
    },
    {
        label: 'Cyber + Health + Panic',
        texts: [
            'Ransomware attack disables hospital scheduling systems',
            'Ambulance dispatch network faces nationwide outage',
            'Public panic grows as emergency services delay response',
        ],
    },
    {
        label: 'War + Refugee Pressure',
        texts: [
            'Artillery strikes intensify near civilian settlements',
            'Mass displacement overwhelms border refugee camps',
            'Regional governments request urgent humanitarian support',
        ],
    },
    {
        label: 'Market Crash Scenario',
        texts: [
            'Central bank warns of persistent inflation risk',
            'Major stock index falls 9 percent in one day',
            'Corporate layoffs accelerate amid liquidity stress',
        ],
    },
];

const SAMPLE_RISKPATH_QUERIES = [
    {
        label: 'Worst-Case Cascade',
        texts: [
            'Military offensive begins near capital city',
            'Oil supply is cut after pipeline attack',
            'Massive data breach affects national bank',
            'Earthquake damages critical infrastructure',
        ],
    },
    {
        label: 'War Risk Path',
        texts: [
            'Border conflict intensifies',
            'Currency markets crash as investors flee',
            'Humanitarian crisis deepens in the region',
            'Tech systems fail during emergency response',
        ],
    },
    {
        label: 'Disaster Domino',
        texts: [
            'Category 5 cyclone destroys coastal power grid',
            'Hospital systems shift to emergency fuel reserves',
            'Water contamination spreads after flood overflow',
            'Food distribution collapses in rural districts',
        ],
    },
    {
        label: 'Cyber-Finance Cascade',
        texts: [
            'Core banking servers hit by coordinated malware attack',
            'ATM and digital payment channels go offline nationwide',
            'Retail supply chains halt due to transaction failures',
            'Public unrest increases as cash withdrawals are limited',
        ],
    },
    {
        label: 'Conflict Supply Spiral',
        texts: [
            'Naval blockade disrupts key trade corridor',
            'Fuel prices surge as imports are delayed',
            'Inflation spikes and currency weakens sharply',
            'Social protests escalate in major urban centers',
        ],
    },
    {
        label: 'Tech Outage Escalation',
        texts: [
            'National cloud platform suffers prolonged outage',
            'Emergency communication channels degrade across regions',
            'Transport control systems operate in manual mode',
            'Critical services face cascading operational failures',
        ],
    },
];

// ══════════════════════════════════════
// TAB NAVIGATION
// ══════════════════════════════════════
document.querySelectorAll('.nav-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
        btn.classList.add('active');
        const tab = document.getElementById('tab-' + btn.dataset.tab);
        if (tab) tab.classList.add('active');
        if (btn.dataset.tab === 'compare') loadModelComparison();
    });
});

document.addEventListener('DOMContentLoaded', () => {
    renderSampleButtons('sample-query-list', SAMPLE_QUERIES, sample => setSampleQuery(sample.text));
    renderSampleButtons('multi-sample-query-list', SAMPLE_MULTI_QUERIES, sample => fillMultiInputs(sample.texts));
    renderSampleButtons('scenario-sample-query-list', SAMPLE_SCENARIO_QUERIES, sample => fillScenarioInputs(sample.texts));
    renderSampleButtons('riskpath-sample-query-list', SAMPLE_RISKPATH_QUERIES, sample => fillRiskPathInputs(sample.texts));
});

// ══════════════════════════════════════
// UTILITIES
// ══════════════════════════════════════
function showLoading() { document.getElementById('loading-overlay').style.display = 'flex'; }
function hideLoading() { document.getElementById('loading-overlay').style.display = 'none'; }

function showToast(msg, type = 'error') {
    const t = document.createElement('div');
    t.className = 'toast toast-' + type;
    t.textContent = msg;
    document.body.appendChild(t);
    setTimeout(() => t.remove(), 3600);
}

async function apiFetch(url, body = null) {
    const opts = body
        ? { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }
        : { method: 'GET' };
    const res = await fetch(url, opts);
    const data = await res.json();
    if (!res.ok || data.error) throw new Error(data.error || 'Request failed');
    return data;
}

function getRiskColorClass(level) {
    return 'risk-' + (level || 'MEDIUM');
}

function getImpactColor(score) {
    if (score >= 85) return 'var(--deep-red)';
    if (score >= 70) return 'var(--red)';
    if (score >= 50) return 'var(--orange)';
    if (score >= 25) return 'var(--yellow)';
    return 'var(--green)';
}

function riskBadgeHTML(risk) {
    if (!risk) return '';
    return `<span class="risk-badge ${getRiskColorClass(risk.level)}">${risk.emoji || ''} ${risk.level}</span>`;
}

function categoryBadgeHTML(result) {
    return `<span class="category-badge" style="border-color:${result.category_color || '#888'}40">${result.category_icon || '❓'} ${result.category}</span>`;
}

// ══════════════════════════════════════
// TAB 1: SINGLE ANALYSIS
// ══════════════════════════════════════
async function analyzeSingle() {
    const text = document.getElementById('single-input').value.trim();
    if (!text || text.length < 10) return showToast('Enter at least 10 characters.');
    const model = document.getElementById('model-select').value;

    showLoading();
    try {
        const [predData, explData] = await Promise.all([
            apiFetch(API.predict, { text, model }),
            apiFetch(API.explain, { text, model }),
        ]);
        renderSingleResult(predData.result);
        renderExplainability(explData.explainability);
    } catch (e) {
        showToast(e.message);
    } finally { hideLoading(); }
}

function renderSingleResult(r) {
    const card = document.getElementById('single-result');
    card.style.display = 'block';
    const impColor = getImpactColor(r.impact_score);

    card.querySelector('#single-result-content').innerHTML = `
        <div class="result-metrics">
            <div class="metric-box">
                <div class="metric-label">Category</div>
                <div style="margin-top:8px">${categoryBadgeHTML(r)}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Impact Score</div>
                <div class="metric-value" style="color:${impColor}"><span class="counter" data-target="${r.impact_score}">0</span></div>
                <div class="severity-meter"><div class="severity-fill" style="width:${r.impact_score}%;background-position:${r.impact_score}% 0"></div></div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Risk Level</div>
                <div style="margin-top:8px">${riskBadgeHTML(r.risk)}</div>
                <div class="metric-sub">${r.risk?.description || ''}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Confidence</div>
                <div class="metric-value text-accent"><span class="counter" data-target="${r.confidence}">0</span>%</div>
                <div class="metric-sub">Model: ${r.model_used === 'neural_network' ? 'Neural Net' : 'Logistic Reg.'}</div>
            </div>
        </div>
    `;
    animateCounters(card);
    card.classList.add('fade-in');
}

function renderExplainability(ex) {
    const card = document.getElementById('explain-card');
    card.style.display = 'block';

    const weights = ex.category_weights || [];
    const wordsHTML = weights.map(w => {
        const cls = w.direction === 'positive' ? 'word-positive' : w.direction === 'negative' ? 'word-negative' : 'word-neutral';
        const score = Math.abs(w.contribution).toFixed(3);
        return `<span class="word-tag ${cls}">${w.word} <small>(${score})</small></span>`;
    }).join('');

    const topFeatures = (ex.top_features || []).map(f =>
        `<span class="word-tag word-neutral">${f.word} <small>(${f.score})</small></span>`
    ).join('');

    document.getElementById('explain-content').innerHTML = `
        <h4 style="margin-bottom:8px;font-size:0.88rem;color:var(--text-secondary)">Key Words Affecting Prediction</h4>
        <div class="word-cloud">${wordsHTML || topFeatures || '<span class="text-muted">No significant features found.</span>'}</div>
        <div class="explanation-text">${ex.explanation || ''}</div>
    `;
    card.classList.add('fade-in');
}

function loadSample() {
    const s = SAMPLES[Math.floor(Math.random() * SAMPLES.length)];
    document.getElementById('single-input').value = s;
}

function setSampleQuery(text) {
    const input = document.getElementById('single-input');
    if (!input) return;
    input.value = text;
    input.focus();
}

function renderSampleButtons(containerId, samples, onSelect) {
    const container = document.getElementById(containerId);
    if (!container) return;

    container.innerHTML = '';
    samples.forEach(sample => {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'btn btn-ghost sample-query-btn';
        button.textContent = sample.label;
        button.addEventListener('click', () => onSelect(sample));
        container.appendChild(button);
    });
}

function createSampleInputRow(index, className, placeholder, text, removeHandlerName) {
    const row = document.createElement('div');
    row.className = 'multi-input-row';

    const number = document.createElement('span');
    number.className = 'row-num';
    number.textContent = index;

    const input = document.createElement('input');
    input.type = 'text';
    input.className = className;
    input.placeholder = placeholder;
    input.value = text;

    const removeButton = document.createElement('button');
    removeButton.type = 'button';
    removeButton.className = 'btn-icon-only btn-remove';
    removeButton.textContent = '✕';
    removeButton.title = 'Remove';
    removeButton.setAttribute('onclick', `${removeHandlerName}(this)`);

    row.append(number, input, removeButton);
    return row;
}

function fillMultiInputs(texts) {
    const container = document.getElementById('multi-inputs');
    if (!container) return;
    container.innerHTML = '';
    texts.forEach((text, index) => {
        container.appendChild(createSampleInputRow(index + 1, 'multi-news-input', 'Enter news headline...', text, 'removeMultiInput'));
    });
}

function fillScenarioInputs(texts) {
    const container = document.getElementById('scenario-inputs');
    if (!container) return;
    container.innerHTML = '';
    texts.forEach((text, index) => {
        container.appendChild(createSampleInputRow(index + 1, 'scenario-news-input', 'Enter crisis event...', text, 'removeScenarioInput'));
    });
}

function fillRiskPathInputs(texts) {
    const container = document.getElementById('riskpath-inputs');
    if (!container) return;
    container.innerHTML = '';
    texts.forEach((text, index) => {
        container.appendChild(createSampleInputRow(index + 1, 'riskpath-news-input', 'Enter crisis event...', text, 'removeRiskPathInput'));
    });
}

// ══════════════════════════════════════
// TAB 2: MULTI-EVENT RANKING
// ══════════════════════════════════════
function addMultiInput() {
    const container = document.getElementById('multi-inputs');
    const count = container.children.length + 1;
    const row = document.createElement('div');
    row.className = 'multi-input-row';
    row.innerHTML = `<span class="row-num">${count}</span><input type="text" class="multi-news-input" placeholder="Enter news headline..."><button class="btn-icon-only btn-remove" onclick="removeMultiInput(this)" title="Remove">✕</button>`;
    container.appendChild(row);
}

function removeMultiInput(btn) {
    const container = document.getElementById('multi-inputs');
    if (container.children.length <= 2) return showToast('Need at least 2 events.');
    btn.closest('.multi-input-row').remove();
    renumberRows(container);
}

function renumberRows(container) {
    container.querySelectorAll('.row-num').forEach((el, i) => el.textContent = i + 1);
}

function loadMultiSamples() {
    const container = document.getElementById('multi-inputs');
    container.innerHTML = '';
    const picks = SAMPLES.sort(() => 0.5 - Math.random()).slice(0, 5);
    picks.forEach((s, i) => {
        container.appendChild(createSampleInputRow(i + 1, 'multi-news-input', 'Enter news headline...', s, 'removeMultiInput'));
    });
}

async function rankEvents() {
    const inputs = document.querySelectorAll('.multi-news-input');
    const texts = Array.from(inputs).map(el => el.value.trim()).filter(Boolean);
    if (texts.length < 2) return showToast('Enter at least 2 news events.');
    const model = document.getElementById('multi-model-select').value;

    showLoading();
    try {
        const data = await apiFetch(API.rank, { texts, model });
        renderRanking(data.ranking);
    } catch (e) { showToast(e.message); }
    finally { hideLoading(); }
}

function renderRanking(ranking) {
    const statsEl = document.getElementById('ranking-stats');
    const tableWrap = document.getElementById('ranking-table-wrap');
    const resultCard = document.getElementById('ranking-result');
    resultCard.style.display = 'block';

    statsEl.innerHTML = `
        <div class="stat-pill"><span class="stat-label">Events:</span><span class="stat-val">${ranking.total_events}</span></div>
        <div class="stat-pill"><span class="stat-label">Avg Impact:</span><span class="stat-val">${ranking.average_impact}</span></div>
        <div class="stat-pill"><span class="stat-label">Max:</span><span class="stat-val text-red">${ranking.max_impact}</span></div>
        <div class="stat-pill"><span class="stat-label">Critical:</span><span class="stat-val text-deepred">${ranking.critical_count}</span></div>
        <div class="stat-pill"><span class="stat-label">Algorithm:</span><span class="stat-val text-accent">Greedy</span></div>
    `;

    const rows = ranking.ranked_events.map(e => {
        const rankCls = e.rank <= 3 ? `rank-${e.rank}` : 'rank-default';
        const impColor = getImpactColor(e.impact_score);
        return `<tr>
            <td><div class="rank-num ${rankCls}">${e.rank}</div></td>
            <td style="max-width:340px">${e.headline || 'N/A'}</td>
            <td>${categoryBadgeHTML(e)}</td>
            <td><div class="impact-bar-wrap"><div class="impact-bar"><div class="impact-bar-fill" style="width:${e.impact_score}%;background:${impColor}"></div></div><span class="impact-value" style="color:${impColor}">${e.impact_score}</span></div></td>
            <td>${riskBadgeHTML(e.risk)}</td>
        </tr>`;
    }).join('');

    tableWrap.innerHTML = `<table class="ranking-table"><thead><tr><th>Rank</th><th>Event</th><th>Category</th><th>Impact</th><th>Risk</th></tr></thead><tbody>${rows}</tbody></table>`;

    // Chart
    const chartCard = document.getElementById('ranking-chart-card');
    chartCard.style.display = 'block';
    createRankingChart(ranking.ranked_events);

    resultCard.classList.add('fade-in');
    chartCard.classList.add('fade-in');
}

// ══════════════════════════════════════
// TAB 3: SCENARIO SIMULATION
// ══════════════════════════════════════
function addScenarioInput() {
    const container = document.getElementById('scenario-inputs');
    const count = container.children.length + 1;
    const row = document.createElement('div');
    row.className = 'multi-input-row';
    row.innerHTML = `<span class="row-num">${count}</span><input type="text" class="scenario-news-input" placeholder="Enter crisis event..."><button class="btn-icon-only btn-remove" onclick="removeScenarioInput(this)" title="Remove">✕</button>`;
    container.appendChild(row);
}

function removeScenarioInput(btn) {
    const container = document.getElementById('scenario-inputs');
    if (container.children.length <= 2) return showToast('Need at least 2 events.');
    btn.closest('.multi-input-row').remove();
    renumberRows(container);
}

function loadScenarioSamples() {
    const container = document.getElementById('scenario-inputs');
    container.innerHTML = '';
    const picks = [
        "Armed forces launch military offensive in border region, ceasefire collapses",
        "Global oil prices surge dramatically after pipeline disruption and refinery shutdown",
        "Devastating earthquake strikes populated coastal area causing widespread destruction"
    ];
    picks.forEach((s, i) => {
        container.appendChild(createSampleInputRow(i + 1, 'scenario-news-input', 'Enter crisis event...', s, 'removeScenarioInput'));
    });
}

async function analyzeScenario() {
    const inputs = document.querySelectorAll('.scenario-news-input');
    const texts = Array.from(inputs).map(el => el.value.trim()).filter(Boolean);
    if (texts.length < 2) return showToast('Enter at least 2 events for scenario analysis.');

    showLoading();
    try {
        const data = await apiFetch(API.scenario, { texts });
        renderScenario(data.scenario);
    } catch (e) { showToast(e.message); }
    finally { hideLoading(); }
}

function renderScenario(sc) {
    const card = document.getElementById('scenario-result');
    card.style.display = 'block';

    const impColor = getImpactColor(sc.compound_score);
    const interactionsHTML = (sc.interactions || []).map(it => `
        <div class="interaction-card" style="border-left-color:${it.multiplier >= 1.5 ? 'var(--red)' : 'var(--accent)'}">
            <span class="category-badge">${it.category1}</span>
            <span class="interaction-arrow">⟷</span>
            <span class="category-badge">${it.category2}</span>
            <span style="font-size:0.82rem;color:var(--text-secondary)">${it.effect}</span>
            <span class="interaction-mult" style="color:${it.multiplier >= 1.5 ? 'var(--red)' : 'var(--accent-light)'}">${it.multiplier}×</span>
        </div>
    `).join('');

    document.getElementById('scenario-content').innerHTML = `
        <div class="result-metrics">
            <div class="metric-box">
                <div class="metric-label">Compound Score</div>
                <div class="metric-value" style="color:${impColor}"><span class="counter" data-target="${sc.compound_score}">0</span></div>
                <div class="severity-meter"><div class="severity-fill" style="width:${sc.compound_score}%;background-position:${sc.compound_score}% 0"></div></div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Base Score</div>
                <div class="metric-value text-blue"><span class="counter" data-target="${sc.base_score}">0</span></div>
                <div class="metric-sub">Before interaction effects</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Total Multiplier</div>
                <div class="metric-value text-accent">${sc.total_multiplier}×</div>
                <div class="metric-sub">Compound amplification</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Risk Level</div>
                <div style="margin-top:8px">${riskBadgeHTML(sc.risk_level)}</div>
            </div>
        </div>
        <h4 style="margin:18px 0 10px;font-size:0.92rem">Category Interactions</h4>
        ${interactionsHTML}
        <div class="explanation-text mt-16">${sc.scenario_description || ''}</div>
    `;
    animateCounters(card);

    // Chart
    const chartCard = document.getElementById('scenario-chart-card');
    chartCard.style.display = 'block';
    createScenarioChart(sc);

    card.classList.add('fade-in');
}

// ══════════════════════════════════════
// TAB 4: RISK PATH (A*)
// ══════════════════════════════════════
function addRiskPathInput() {
    const container = document.getElementById('riskpath-inputs');
    const count = container.children.length + 1;
    const row = document.createElement('div');
    row.className = 'multi-input-row';
    row.innerHTML = `<span class="row-num">${count}</span><input type="text" class="riskpath-news-input" placeholder="Enter crisis event..."><button class="btn-icon-only btn-remove" onclick="removeRiskPathInput(this)" title="Remove">✕</button>`;
    container.appendChild(row);
}

function removeRiskPathInput(btn) {
    const container = document.getElementById('riskpath-inputs');
    if (container.children.length <= 2) return showToast('Need at least 2 events.');
    btn.closest('.multi-input-row').remove();
    renumberRows(container);
}

function loadRiskPathSamples() {
    const container = document.getElementById('riskpath-inputs');
    container.innerHTML = '';
    const picks = SAMPLES.sort(() => 0.5 - Math.random()).slice(0, 4);
    picks.forEach((s, i) => {
        container.appendChild(createSampleInputRow(i + 1, 'riskpath-news-input', 'Enter crisis event...', s, 'removeRiskPathInput'));
    });
}

async function findRiskPath() {
    const inputs = document.querySelectorAll('.riskpath-news-input');
    const texts = Array.from(inputs).map(el => el.value.trim()).filter(Boolean);
    if (texts.length < 2) return showToast('Enter at least 2 events.');

    showLoading();
    try {
        const data = await apiFetch(API.riskPath, { texts });
        renderRiskPath(data.risk_path);
    } catch (e) { showToast(e.message); }
    finally { hideLoading(); }
}

function renderRiskPath(rp) {
    const card = document.getElementById('riskpath-result');
    card.style.display = 'block';

    const statsHTML = `
        <div class="stats-row mb-8">
            <div class="stat-pill"><span class="stat-label">Algorithm:</span><span class="stat-val text-accent">${rp.algorithm}</span></div>
            <div class="stat-pill"><span class="stat-label">Nodes Explored:</span><span class="stat-val">${rp.nodes_explored}</span></div>
            <div class="stat-pill"><span class="stat-label">Compound Risk:</span><span class="stat-val text-red">${rp.total_compound_risk}</span></div>
            <div class="stat-pill"><span class="stat-label">Avg Risk:</span><span class="stat-val">${rp.average_risk}</span></div>
            <div class="stat-pill">${riskBadgeHTML(rp.risk_level)}</div>
        </div>
    `;

    const pathHTML = (rp.optimal_path || []).map((e, i) => {
        const isLast = i === rp.optimal_path.length - 1;
        const impColor = getImpactColor(e.impact_score);
        const interaction = rp.interactions && rp.interactions[i];
        return `
            <div class="path-node fade-in" style="animation-delay:${i * 0.15}s">
                <div class="path-connector">
                    <div class="path-dot" style="border-color:${impColor}"></div>
                    ${!isLast ? '<div class="path-line"></div>' : ''}
                </div>
                <div style="flex:1">
                    <div class="path-card">
                        <div style="display:flex;justify-content:space-between;align-items:start;flex-wrap:wrap;gap:8px">
                            <div>
                                <div style="font-size:0.72rem;color:var(--text-muted);margin-bottom:4px">STEP ${e.path_position || i + 1}</div>
                                <div style="font-weight:600;margin-bottom:6px">${e.headline || 'Event'}</div>
                                ${categoryBadgeHTML(e)}
                            </div>
                            <div style="text-align:right">
                                <div class="impact-value" style="color:${impColor};font-size:1.2rem">${e.impact_score}</div>
                                ${riskBadgeHTML(e.risk)}
                            </div>
                        </div>
                    </div>
                    ${interaction ? `<div style="padding:6px 0 0 18px;font-size:0.78rem;color:var(--text-muted)">↓ ${interaction.effect} with next event (${interaction.multiplier}× multiplier)</div>` : ''}
                </div>
            </div>
        `;
    }).join('');

    document.getElementById('riskpath-content').innerHTML = statsHTML + '<div class="mt-8">' + pathHTML + '</div>';
    card.classList.add('fade-in');
}

// ══════════════════════════════════════
// TAB 5: MODEL COMPARISON
// ══════════════════════════════════════
let comparisonLoaded = false;

async function loadModelComparison() {
    if (comparisonLoaded) return;
    try {
        const data = await apiFetch(API.compare);
        renderModelComparison(data.report);
        comparisonLoaded = true;
    } catch (e) {
        document.getElementById('compare-content').innerHTML = `<div class="glass-card"><p class="text-red">Failed to load: ${e.message}</p></div>`;
    }
}

function renderModelComparison(report) {
    const models = report.models || {};
    const lr = models.logistic_regression || {};
    const nn = models.neural_network || {};
    const ridge = models.linear_regression || {};
    const info = report.dataset_info || report.dataset || {};

    const fmtPercent = value => Number.isFinite(Number(value)) ? `${(Number(value) * 100).toFixed(1)}%` : 'N/A';
    const fmtNumber = value => Number.isFinite(Number(value)) ? Number(value).toFixed(2) : 'N/A';
    const fmtCount = value => Number.isFinite(Number(value)) ? Number(value) : 'N/A';
    const pickMetric = (model, key) => {
        if (key === 'accuracy') return model.test_accuracy ?? model.accuracy;
        const weighted = model.weighted_avg || {};
        return weighted[key] ?? model[key];
    };
    const normalizeClassModel = model => ({
        accuracy: pickMetric(model, 'accuracy'),
        precision: pickMetric(model, 'precision'),
        recall: pickMetric(model, 'recall'),
        f1_score: pickMetric(model, 'f1_score'),
    });
    const formatArchitecture = architecture => {
        if (!architecture) return 'Multi-Layer Perceptron';
        if (typeof architecture === 'string') return architecture;
        const layers = Array.isArray(architecture.layers) ? architecture.layers : [];
        const layerNames = layers.map(layer => {
            if (layer.type === 'Dense') return `Dense ${layer.units}`;
            if (layer.type === 'Dropout') return `Dropout ${layer.rate}`;
            return layer.type || 'Layer';
        });
        return layerNames.length ? layerNames.join(' → ') : 'Multi-Layer Perceptron';
    };

    const lrMetrics = normalizeClassModel(lr);
    const nnMetrics = normalizeClassModel(nn);

    document.getElementById('compare-content').innerHTML = `
        <div class="glass-card">
            <div class="model-card-header">
                <div class="model-icon lr">📐</div>
                <div><h3>Logistic Regression</h3><p class="text-muted" style="font-size:0.78rem">Classification Model</p></div>
            </div>
            <div class="result-metrics">
                <div class="metric-box"><div class="metric-label">Accuracy</div><div class="metric-value text-green">${fmtPercent(lrMetrics.accuracy)}</div></div>
                <div class="metric-box"><div class="metric-label">Precision</div><div class="metric-value text-blue">${fmtPercent(lrMetrics.precision)}</div></div>
                <div class="metric-box"><div class="metric-label">Recall</div><div class="metric-value text-yellow">${fmtPercent(lrMetrics.recall)}</div></div>
                <div class="metric-box"><div class="metric-label">F1 Score</div><div class="metric-value text-accent">${fmtPercent(lrMetrics.f1_score)}</div></div>
            </div>
        </div>
        <div class="glass-card">
            <div class="model-card-header">
                <div class="model-icon nn">🧠</div>
                <div><h3>Neural Network (MLP)</h3><p class="text-muted" style="font-size:0.78rem">${formatArchitecture(nn.architecture)}</p></div>
            </div>
            <div class="result-metrics">
                <div class="metric-box"><div class="metric-label">Accuracy</div><div class="metric-value text-green">${fmtPercent(nnMetrics.accuracy)}</div></div>
                <div class="metric-box"><div class="metric-label">Precision</div><div class="metric-value text-blue">${fmtPercent(nnMetrics.precision)}</div></div>
                <div class="metric-box"><div class="metric-label">Recall</div><div class="metric-value text-yellow">${fmtPercent(nnMetrics.recall)}</div></div>
                <div class="metric-box"><div class="metric-label">F1 Score</div><div class="metric-value text-accent">${fmtPercent(nnMetrics.f1_score)}</div></div>
            </div>
        </div>
    `;

    // Regression card
    const regCard = document.getElementById('regression-card');
    regCard.style.display = 'block';
    document.getElementById('regression-content').innerHTML = `
        <div class="result-metrics">
            <div class="metric-box"><div class="metric-label">MAE</div><div class="metric-value text-green">${fmtNumber(ridge.mae)}</div><div class="metric-sub">Mean Absolute Error</div></div>
            <div class="metric-box"><div class="metric-label">RMSE</div><div class="metric-value text-orange">${fmtNumber(ridge.rmse)}</div><div class="metric-sub">Root Mean Squared Error</div></div>
            <div class="metric-box"><div class="metric-label">R² Score</div><div class="metric-value text-accent">${fmtNumber(ridge.r2_score)}</div><div class="metric-sub">Coefficient of Determination</div></div>
        </div>
        <div class="stats-row mt-16">
            <div class="stat-pill"><span class="stat-label">Dataset:</span><span class="stat-val">${fmtCount(info.total_samples ?? info.total_records)} records</span></div>
            <div class="stat-pill"><span class="stat-label">Features:</span><span class="stat-val">${fmtCount(info.tfidf_features_lr ?? info.features)} TF-IDF</span></div>
            <div class="stat-pill"><span class="stat-label">Train:</span><span class="stat-val">${fmtCount(info.train_samples ?? info.train_size)}</span></div>
            <div class="stat-pill"><span class="stat-label">Test:</span><span class="stat-val">${fmtCount(info.test_samples ?? info.test_size)}</span></div>
        </div>
    `;

    // Comparison chart
    const chartCard = document.getElementById('compare-chart-card');
    chartCard.style.display = 'block';
    createComparisonChart(lrMetrics, nnMetrics);
}
