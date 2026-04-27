// Restore the loading placeholder that `<html class="no-js">` + the
// `.no-js #loading { display: none }` rule in charts.css hide by default.
// Coupled with the <noscript> block in charts.html: JS-disabled users see
// the "JavaScript is required" message; JS-enabled users see "Loading...".
document.documentElement.classList.remove('no-js');

Chart.defaults.font.family = '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
Chart.defaults.plugins.legend.position = 'bottom';
Chart.defaults.plugins.tooltip.backgroundColor = 'rgba(0, 0, 0, 0.8)';

// Must match SCHEMA_VERSION in bench-support/src/json_results.rs.
const EXPECTED_SCHEMA_MAJOR = '1';

// Substituted by render_docs from bench_support::registry::POLICIES
// (the @POLICY_COLORS@ sentinel below is replaced with a JSON literal).
// Edit colors in registry.rs, NOT here. Unknown names fall back to a
// deterministic HSL color hashed from the display name in colorForPolicy().
const POLICY_COLORS = /* @POLICY_COLORS@ */ {};

function colorForPolicy(name) {
    if (POLICY_COLORS[name]) return POLICY_COLORS[name];
    // FNV-1a 32-bit, then map to HSL hue for stable distinct fallbacks.
    let h = 0x811c9dc5;
    for (let i = 0; i < name.length; i++) {
        h ^= name.charCodeAt(i);
        h = Math.imul(h, 0x01000193);
    }
    const hue = Math.abs(h) % 360;
    return `hsl(${hue}, 65%, 55%)`;
}

function safeText(value, fallback = 'unknown') {
    return value == null || value === '' ? fallback : String(value);
}

function formatTimestamp(iso) {
    if (!iso) return 'unknown';
    const d = new Date(iso);
    return Number.isNaN(d.getTime()) ? iso : d.toLocaleString();
}

function formatNumber(n) {
    if (n == null || Number.isNaN(Number(n))) return 'unknown';
    return Number(n).toLocaleString('en-US');
}

function appendMetadataItem(grid, label, value) {
    const item = document.createElement('div');
    item.className = 'metadata-item';
    const l = document.createElement('span');
    l.className = 'metadata-label';
    l.textContent = `${label}:`;
    const v = document.createElement('span');
    v.textContent = String(value);
    item.append(l, v);
    grid.append(item);
}

// Draws a dashed reference line + label at a value on the value axis.
// Works for both vertical (x scale) and horizontal (y scale) bar charts.
function makeReferenceLinePlugin(value, label, axis = 'x') {
    return {
        id: 'referenceLine',
        afterDatasetsDraw(chart) {
            const scale = chart.scales[axis];
            if (!scale) return;
            const pos = scale.getPixelForValue(value);
            const { top, bottom, left, right } = chart.chartArea;
            const ctx = chart.ctx;
            ctx.save();
            ctx.strokeStyle = 'rgba(44, 62, 80, 0.6)';
            ctx.setLineDash([4, 4]);
            ctx.lineWidth = 1;
            ctx.beginPath();
            if (axis === 'x') {
                if (pos < left || pos > right) { ctx.restore(); return; }
                ctx.moveTo(pos, top);
                ctx.lineTo(pos, bottom);
            } else {
                if (pos < top || pos > bottom) { ctx.restore(); return; }
                ctx.moveTo(left, pos);
                ctx.lineTo(right, pos);
            }
            ctx.stroke();
            if (label) {
                ctx.setLineDash([]);
                ctx.fillStyle = 'rgba(44, 62, 80, 0.85)';
                ctx.font = '11px -apple-system, sans-serif';
                ctx.textBaseline = 'top';
                if (axis === 'x') {
                    ctx.textAlign = 'left';
                    ctx.fillText(label, pos + 4, top + 2);
                } else {
                    ctx.textAlign = 'right';
                    ctx.fillText(label, right - 4, pos + 2);
                }
            }
            ctx.restore();
        }
    };
}

fetch('results.json')
    .then(response => {
        if (!response.ok) {
            throw new Error(`Failed to load results.json (HTTP ${response.status})`);
        }
        return response.json();
    })
    .then(data => {
        if (data && data.schema_version) {
            const major = String(data.schema_version).split('.')[0];
            if (major !== EXPECTED_SCHEMA_MAJOR) {
                throw new Error(
                    `Unsupported schema version ${data.schema_version}, ` +
                    `expected ${EXPECTED_SCHEMA_MAJOR}.x. Re-render with a matching ` +
                    `version of render_docs.`
                );
            }
        }
        document.getElementById('loading').style.display = 'none';
        document.getElementById('content').style.display = 'block';

        renderMetadata((data && data.metadata) || {});
        renderCharts(Array.isArray(data && data.results) ? data.results : []);
    })
    .catch(error => {
        document.getElementById('loading').style.display = 'none';
        const errEl = document.getElementById('error');
        errEl.style.display = 'block';
        errEl.textContent = '';

        const msg = document.createElement('strong');
        msg.textContent = `Error loading benchmark data: ${error.message}`;
        errEl.append(msg);

        if (window.location.protocol === 'file:') {
            errEl.append(
                '\n\nBrowsers block fetch() over file://. Serve this directory over HTTP, e.g.\n'
            );
            const cmd = document.createElement('code');
            cmd.textContent = 'python3 -m http.server';
            errEl.append(cmd);
            errEl.append('\nthen open http://localhost:8000/charts.html');
        } else {
            errEl.append('\n\nMake sure results.json exists in the same directory.');
        }
    });

function showChartEmptyState(canvasId, message) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const container = canvas.parentElement;
    const empty = document.createElement('div');
    empty.className = 'chart-empty';
    empty.textContent = message;
    container.replaceChildren(empty);
}

function renderMetadata(metadata) {
    const metadataDiv = document.getElementById('metadata');
    const config = metadata.config || {};
    const items = [
        ['Date', formatTimestamp(metadata.timestamp)],
        ['Commit', safeText(metadata.git_commit)],
        ['Branch', safeText(metadata.git_branch)],
        ['Dirty', metadata.git_dirty ? 'Yes' : 'No'],
        ['Rustc', safeText(metadata.rustc_version)],
        ['Host', safeText(metadata.host_triple)],
        ['CPU', safeText(metadata.cpu_model)],
        ['Capacity', formatNumber(config.capacity)],
        ['Operations', formatNumber(config.operations)]
    ];

    const grid = document.createElement('div');
    grid.className = 'metadata-grid';
    for (const [label, value] of items) {
        appendMetadataItem(grid, label, value);
    }
    metadataDiv.replaceChildren(grid);
}

function renderCharts(results) {
    const byCase = {};
    for (const r of results) {
        if (!byCase[r.case_id]) byCase[r.case_id] = [];
        byCase[r.case_id].push(r);
    }

    renderMatrixBarChart('hitRateChart', byCase['hit_rate'] || [], {
        emptyMessage: 'No hit-rate data available for this run.',
        extract: r => r.metrics.hit_stats ? r.metrics.hit_stats.hit_rate * 100 : null,
        yAxisLabel: 'Hit Rate (%)',
        yAxisMax: 100,
        tickCallback: v => v + '%',
        tooltipFormat: v => `${v.toFixed(2)}%`
    });

    const comprehensive = byCase['comprehensive'] || [];
    renderMatrixBarChart('throughputChart', comprehensive, {
        emptyMessage: 'No throughput data available for this run.',
        extract: r => r.metrics.throughput ? r.metrics.throughput.ops_per_sec / 1_000_000 : null,
        yAxisLabel: 'Million ops/sec',
        tooltipFormat: v => `${v.toFixed(2)} M ops/sec`
    });
    renderMatrixBarChart('latencyChart', comprehensive, {
        emptyMessage: 'No latency data available for this run.',
        extract: r => r.metrics.latency ? r.metrics.latency.p99_ns : null,
        yAxisLabel: 'P99 Latency (ns)',
        tooltipFormat: v => `${formatNumber(v)} ns`
    });

    renderRankedBarChart('scanResistanceChart', byCase['scan_resistance'] || [], {
        emptyMessage: 'No scan-resistance data available for this run.',
        datasetLabel: 'Resistance Score',
        extract: r => r.metrics.scan_resistance ? r.metrics.scan_resistance.resistance_score : null,
        xAxisLabel: 'Score (1.0 = perfect recovery)',
        xAxisMaxFn: scores => Math.max(1.0, ...scores) * 1.05,
        tooltipFormat: v => v.toFixed(3),
        referenceLine: { value: 1.0, label: 'Perfect (1.0)' }
    });

    renderRankedBarChart('adaptationChart', byCase['adaptation'] || [], {
        emptyMessage: 'No adaptation data available for this run.',
        datasetLabel: 'Operations to 80%',
        extract: r => r.metrics.adaptation ? r.metrics.adaptation.ops_to_80_percent : null,
        xAxisLabel: 'Operations to reach 80% of stable hit rate (lower is better)',
        tooltipFormat: v => formatNumber(v)
    });

    renderAdaptationCurveChart('adaptationCurveChart', byCase['adaptation'] || []);
}

// Records first-seen order for a key across results, preserving the
// semantically meaningful ordering produced by the bench runner instead
// of collapsing to lexicographic.
function collectInsertionOrder(results, key) {
    const seen = new Set();
    const order = [];
    for (const r of results) {
        const v = r[key];
        if (v != null && !seen.has(v)) {
            seen.add(v);
            order.push(v);
        }
    }
    return order;
}

function renderMatrixBarChart(canvasId, results, opts) {
    const workloads = collectInsertionOrder(results, 'workload_name');
    const policies = collectInsertionOrder(results, 'policy_name');

    const byPolicy = {};
    for (const r of results) {
        const value = opts.extract(r);
        if (value == null) continue;
        if (!byPolicy[r.policy_name]) byPolicy[r.policy_name] = {};
        byPolicy[r.policy_name][r.workload_name] = value;
    }

    const datasets = policies
        .filter(p => byPolicy[p])
        .map(policy => ({
            label: policy,
            // null leaves a gap rather than misrepresenting "missing" as 0.
            data: workloads.map(w =>
                Object.prototype.hasOwnProperty.call(byPolicy[policy], w)
                    ? byPolicy[policy][w]
                    : null
            ),
            backgroundColor: colorForPolicy(policy),
            borderColor: colorForPolicy(policy),
            borderWidth: 1
        }));

    if (datasets.length === 0) {
        showChartEmptyState(canvasId, opts.emptyMessage);
        return;
    }

    const yScale = {
        beginAtZero: true,
        title: { display: true, text: opts.yAxisLabel }
    };
    if (opts.yAxisMax != null) yScale.max = opts.yAxisMax;
    if (opts.tickCallback) yScale.ticks = { callback: opts.tickCallback };

    new Chart(document.getElementById(canvasId), {
        type: 'bar',
        data: { labels: workloads, datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: yScale,
                x: { title: { display: true, text: 'Workload' } }
            },
            plugins: {
                tooltip: {
                    callbacks: {
                        label: ctx => {
                            const v = ctx.parsed.y;
                            if (v == null) return `${ctx.dataset.label}: n/a`;
                            return `${ctx.dataset.label}: ${opts.tooltipFormat(v)}`;
                        }
                    }
                }
            }
        }
    });
}

function renderAdaptationCurveChart(canvasId, results) {
    // Keep only rows that actually carry a curve (older artifacts won't).
    const usable = results.filter(r =>
        r.metrics.adaptation
        && Array.isArray(r.metrics.adaptation.hit_rate_curve)
        && r.metrics.adaptation.hit_rate_curve.length > 0
    );

    if (usable.length === 0) {
        showChartEmptyState(
            canvasId,
            'No adaptation curve data available. Re-run benchmarks with bench-support ≥ schema 1.2.'
        );
        return;
    }

    // The harness uses a single window size across policies; pick the
    // first non-zero one for axis labelling, falling back to indices.
    const windowSize = usable
        .map(r => r.metrics.adaptation.window_size || 0)
        .find(s => s > 0) || 0;
    const maxLen = usable.reduce(
        (m, r) => Math.max(m, r.metrics.adaptation.hit_rate_curve.length),
        0
    );

    const labels = [];
    for (let i = 0; i < maxLen; i++) {
        labels.push(windowSize > 0 ? formatNumber((i + 1) * windowSize) : `W${i + 1}`);
    }

    const datasets = collectInsertionOrder(usable, 'policy_name')
        .map(policy => {
            const row = usable.find(r => r.policy_name === policy);
            if (!row) return null;
            const curve = row.metrics.adaptation.hit_rate_curve;
            const data = labels.map((_, i) =>
                i < curve.length ? curve[i] * 100 : null
            );
            return {
                label: policy,
                data,
                borderColor: colorForPolicy(policy),
                backgroundColor: colorForPolicy(policy),
                // Connect across any null tail so a short curve doesn't
                // visually drop to zero at the end.
                spanGaps: false,
                tension: 0.2,
                pointRadius: 2,
                borderWidth: 2,
                fill: false
            };
        })
        .filter(Boolean);

    new Chart(document.getElementById(canvasId), {
        type: 'line',
        data: { labels, datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'nearest', axis: 'x', intersect: false },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    title: { display: true, text: 'Hit Rate (%)' },
                    ticks: { callback: v => v + '%' }
                },
                x: {
                    title: {
                        display: true,
                        text: windowSize > 0
                            ? 'Operations after workload shift'
                            : 'Adaptation window'
                    }
                }
            },
            plugins: {
                tooltip: {
                    callbacks: {
                        label: ctx => {
                            const v = ctx.parsed.y;
                            if (v == null) return `${ctx.dataset.label}: n/a`;
                            return `${ctx.dataset.label}: ${v.toFixed(1)}%`;
                        }
                    }
                }
            }
        }
    });
}

function renderRankedBarChart(canvasId, results, opts) {
    const policies = [];
    const values = [];
    for (const r of results) {
        const v = opts.extract(r);
        if (v == null) continue;
        policies.push(r.policy_name);
        values.push(v);
    }

    if (values.length === 0) {
        showChartEmptyState(canvasId, opts.emptyMessage);
        return;
    }

    const xScale = {
        beginAtZero: true,
        title: { display: true, text: opts.xAxisLabel }
    };
    if (typeof opts.xAxisMaxFn === 'function') {
        xScale.max = opts.xAxisMaxFn(values);
    } else if (opts.xAxisMax != null) {
        xScale.max = opts.xAxisMax;
    }

    const config = {
        type: 'bar',
        data: {
            labels: policies,
            datasets: [{
                label: opts.datasetLabel,
                data: values,
                backgroundColor: policies.map(colorForPolicy),
                borderColor: policies.map(colorForPolicy),
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            indexAxis: 'y',
            scales: { x: xScale },
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: ctx => `${opts.datasetLabel}: ${opts.tooltipFormat(ctx.parsed.x)}`
                    }
                }
            }
        },
        plugins: opts.referenceLine
            ? [makeReferenceLinePlugin(opts.referenceLine.value, opts.referenceLine.label, 'x')]
            : []
    };

    new Chart(document.getElementById(canvasId), config);
}
