// Algorithm definitions with latency formulas
const ALGORITHMS = {
    'Ring_Simple': {
        name: 'Ring + Simple',
        color: '#9b59b6',
        enabled: true,
        // T = 2(N-1)*α + 2*β*S*(N-1)/N
        latency: (params, S) => {
            const N = params.totalRanks;
            const alpha = params.alphaNetwork;
            const beta = 1 / (params.bwNetwork * 1e9); // Convert GB/s to s/byte
            return 2 * (N - 1) * alpha * 1e-6 + 2 * beta * S * (N - 1) / N;
        }
    },
    'Ring_LL128': {
        name: 'Ring + LL128',
        color: '#4d96ff',
        enabled: true,
        latency: (params, S) => {
            const N = params.totalRanks;
            const alpha = params.alphaNetwork * 1.5; // Higher latency for LL128
            const beta = (1 / (params.bwNetwork * 1e9)) * params.ll128Overhead;
            return 2 * (N - 1) * alpha * 1e-6 + 2 * beta * S * (N - 1) / N;
        }
    },
    'Ring_LL': {
        name: 'Ring + LL',
        color: '#6bcf7f',
        enabled: true,
        latency: (params, S) => {
            const N = params.totalRanks;
            const alpha = params.alphaNetwork * 1.2; // Slightly higher latency
            const beta = (1 / (params.bwNetwork * 1e9)) * params.llOverhead;
            return 2 * (N - 1) * alpha * 1e-6 + 2 * beta * S * (N - 1) / N;
        }
    },
    'Tree_Simple': {
        name: 'Tree + Simple',
        color: '#ffd93d',
        enabled: true,
        // T = 2*log2(N)*α + log2(N)*β*S
        latency: (params, S) => {
            const N = params.totalRanks;
            const logN = Math.log2(N);
            const alpha = params.alphaNetwork;
            const beta = 1 / (params.bwNetwork * 1e9);
            return 2 * logN * alpha * 1e-6 + logN * beta * S;
        }
    },
    'Tree_LL128': {
        name: 'Tree + LL128',
        color: '#ff8c42',
        enabled: true,
        latency: (params, S) => {
            const N = params.totalRanks;
            const logN = Math.log2(N);
            const alpha = params.alphaNetwork * 1.3;
            const beta = (1 / (params.bwNetwork * 1e9)) * params.ll128Overhead;
            return 2 * logN * alpha * 1e-6 + logN * beta * S;
        }
    },
    'Tree_LL': {
        name: 'Tree + LL',
        color: '#ff6b6b',
        enabled: true,
        latency: (params, S) => {
            const N = params.totalRanks;
            const logN = Math.log2(N);
            const alpha = params.alphaNetwork * 1.1;
            const beta = (1 / (params.bwNetwork * 1e9)) * params.llOverhead;
            return 2 * logN * alpha * 1e-6 + logN * beta * S;
        }
    },
    'NVLS_Tree_Simple': {
        name: 'NVLS Tree + Simple',
        color: '#f39c12',
        enabled: true,
        // Intra-node: T = 2*α_nvls + 2*β_nvlink*S
        // Multi-node: adds 2*log2(M)*α_network + 2*log2(M)*β_network*S
        latency: (params, S) => {
            const M = params.numNodes;
            const logM = Math.log2(M);
            const alphaNVLS = params.alphaNVLS;
            const alphaNet = params.alphaNetwork;
            const betaNVLink = 1 / (params.bwNVLink * 1e9);
            const betaNet = 1 / (params.bwNetwork * 1e9);

            // Intra-node NVLink + inter-node network
            return 2 * alphaNVLS * 1e-6 + 2 * betaNVLink * S +
                   2 * logM * alphaNet * 1e-6 + 2 * logM * betaNet * S;
        }
    },
    'PAT_Simple': {
        name: 'PAT + Simple',
        color: '#1abc9c',
        enabled: true,
        // T = log2(N)*α + β*S*(N-1)/N
        latency: (params, S) => {
            const N = params.totalRanks;
            const logN = Math.log2(N);
            const alpha = params.alphaNetwork;
            const beta = 1 / (params.bwNetwork * 1e9);
            return logN * alpha * 1e-6 + beta * S * (N - 1) / N;
        }
    }
};

// State
let canvas, ctx;
let params = {
    collective: 'AllReduce',
    numNodes: 16,
    ranksPerNode: 8,
    totalRanks: 128,
    alphaNVLink: 1.5,
    alphaNetwork: 15,
    alphaNVLS: 2,
    bwNVLink: 600,
    bwNetwork: 50,
    llOverhead: 3,
    ll128Overhead: 1.0625
};

// Message sizes (powers of 2 from 1KB to 16GB)
const messageSizes = [];
for (let i = 10; i <= 34; i++) {
    messageSizes.push(Math.pow(2, i)); // 1KB to 16GB
}

// Initialize
window.addEventListener('DOMContentLoaded', () => {
    canvas = document.getElementById('perfChart');
    ctx = canvas.getContext('2d');

    setupControls();
    setupAlgorithmCheckboxes();
    updateChart();
});

function setupControls() {
    document.getElementById('collective').addEventListener('change', (e) => {
        params.collective = e.target.value;
        updateChart();
    });

    document.getElementById('numNodes').addEventListener('change', (e) => {
        params.numNodes = parseInt(e.target.value);
        params.totalRanks = params.numNodes * params.ranksPerNode;
        updateChart();
    });

    document.getElementById('ranksPerNode').addEventListener('change', (e) => {
        params.ranksPerNode = parseInt(e.target.value);
        params.totalRanks = params.numNodes * params.ranksPerNode;
        updateChart();
    });

    // Hardware parameters
    const hwParams = ['alphaNVLink', 'alphaNetwork', 'alphaNVLS', 'bwNVLink', 'bwNetwork', 'llOverhead', 'll128Overhead'];
    hwParams.forEach(param => {
        document.getElementById(param).addEventListener('change', (e) => {
            params[param] = parseFloat(e.target.value);
            updateChart();
        });
    });
}

function setupAlgorithmCheckboxes() {
    const container = document.getElementById('algorithmCheckboxes');

    Object.keys(ALGORITHMS).forEach(key => {
        const algo = ALGORITHMS[key];
        const item = document.createElement('div');
        item.className = 'checkbox-item';

        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.id = `algo_${key}`;
        checkbox.checked = algo.enabled;
        checkbox.addEventListener('change', (e) => {
            algo.enabled = e.target.checked;
            updateChart();
        });

        const colorIndicator = document.createElement('div');
        colorIndicator.className = 'color-indicator';
        colorIndicator.style.backgroundColor = algo.color;

        const label = document.createElement('label');
        label.htmlFor = `algo_${key}`;
        label.textContent = algo.name;

        item.appendChild(checkbox);
        item.appendChild(colorIndicator);
        item.appendChild(label);
        container.appendChild(item);
    });
}

function calculateBandwidth(latency, messageSize, collective, nranks) {
    // Algorithm bandwidth = messageSize / latency (in GB/s)
    if (latency <= 0) return 0;
    const algBw = messageSize / latency / 1e9; // Convert to GB/s

    // Bus bandwidth factor depends on collective type (from NCCL tests)
    let busFactor;
    switch(collective) {
        case 'AllReduce':
            busFactor = (2 * (nranks - 1)) / nranks;
            break;
        case 'AllGather':
        case 'ReduceScatter':
            busFactor = (nranks - 1) / nranks;
            break;
        case 'Broadcast':
            busFactor = 1.0;
            break;
        default:
            busFactor = 1.0;
    }

    return algBw * busFactor; // Return bus bandwidth
}

function updateChart() {
    const width = canvas.width;
    const height = canvas.height;
    const margin = { top: 40, right: 200, bottom: 60, left: 80 };
    const plotWidth = width - margin.left - margin.right;
    const plotHeight = height - margin.top - margin.bottom;

    // Clear canvas
    ctx.fillStyle = '#0d1117';
    ctx.fillRect(0, 0, width, height);

    // Calculate performance data for each algorithm
    const perfData = {};
    const maxBandwidths = new Array(messageSizes.length).fill(0);

    Object.keys(ALGORITHMS).forEach(key => {
        const algo = ALGORITHMS[key];
        if (!algo.enabled) return;

        perfData[key] = messageSizes.map((size, idx) => {
            const latency = algo.latency(params, size);
            const bandwidth = calculateBandwidth(latency, size, params.collective, params.totalRanks);

            // Track maximum bandwidth at each message size
            if (bandwidth > maxBandwidths[idx]) {
                maxBandwidths[idx] = bandwidth;
            }

            return { size, latency, bandwidth };
        });
    });

    // Find bounds
    const maxBW = Math.max(...maxBandwidths, 1);
    const minSize = messageSizes[0];
    const maxSize = messageSizes[messageSizes.length - 1];

    // Draw chart area
    ctx.save();
    ctx.translate(margin.left, margin.top);
    ctx.beginPath();
    ctx.rect(0, 0, plotWidth, plotHeight);
    ctx.clip();

    // Draw algorithm lines
    Object.keys(ALGORITHMS).forEach(key => {
        const algo = ALGORITHMS[key];
        if (!algo.enabled || !perfData[key]) return;

        ctx.strokeStyle = algo.color;
        ctx.lineWidth = 2;
        ctx.beginPath();

        perfData[key].forEach((point, i) => {
            const x = sizeToCanvas(point.size, minSize, maxSize, plotWidth);
            const y = bwToCanvas(point.bandwidth, 0, maxBW, plotHeight);

            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        });

        ctx.stroke();
    });

    // Draw maximum performance envelope in bold
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 4;
    ctx.beginPath();

    messageSizes.forEach((size, i) => {
        const x = sizeToCanvas(size, minSize, maxSize, plotWidth);
        const y = bwToCanvas(maxBandwidths[i], 0, maxBW, plotHeight);

        if (i === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
    });

    ctx.stroke();

    // Add white outline for better visibility
    ctx.strokeStyle = '#0d1117';
    ctx.lineWidth = 6;
    ctx.globalCompositeOperation = 'destination-over';
    ctx.beginPath();
    messageSizes.forEach((size, i) => {
        const x = sizeToCanvas(size, minSize, maxSize, plotWidth);
        const y = bwToCanvas(maxBandwidths[i], 0, maxBW, plotHeight);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.globalCompositeOperation = 'source-over';

    ctx.restore();

    // Draw axes and labels
    drawAxes(margin, plotWidth, plotHeight, minSize, maxSize, maxBW);

    // Draw legend
    drawLegend(margin, plotWidth, plotHeight);

    // Update info cards
    updateInfoCards(perfData, maxBandwidths);
}

function drawAxes(margin, width, height, minSize, maxSize, maxBW) {
    ctx.strokeStyle = '#30363d';
    ctx.lineWidth = 1;

    // X-axis
    ctx.beginPath();
    ctx.moveTo(margin.left, margin.top + height);
    ctx.lineTo(margin.left + width, margin.top + height);
    ctx.stroke();

    // Y-axis
    ctx.beginPath();
    ctx.moveTo(margin.left, margin.top);
    ctx.lineTo(margin.left, margin.top + height);
    ctx.stroke();

    // X-axis label
    ctx.fillStyle = '#8b949e';
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Message Size (log scale)', margin.left + width / 2, margin.top + height + 45);

    // Y-axis label
    ctx.save();
    ctx.translate(margin.left - 60, margin.top + height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Bus Bandwidth (GB/s)', 0, 0);
    ctx.restore();

    // Title
    ctx.font = 'bold 14px sans-serif';
    ctx.fillStyle = '#c9d1d9';
    ctx.textAlign = 'center';
    ctx.fillText(`${params.collective} Performance: ${params.totalRanks} ranks (${params.numNodes} nodes × ${params.ranksPerNode} GPUs/node)`,
                 margin.left + width / 2, 20);

    // X-axis ticks (powers of 2)
    const xTicks = [1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216, 67108864, 268435456, 1073741824, 4294967296, 17179869184];
    xTicks.forEach(size => {
        if (size >= minSize && size <= maxSize) {
            const x = margin.left + sizeToCanvas(size, minSize, maxSize, width);

            ctx.fillStyle = '#8b949e';
            ctx.font = '10px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText(formatSize(size), x, margin.top + height + 20);

            // Tick mark
            ctx.strokeStyle = '#30363d';
            ctx.beginPath();
            ctx.moveTo(x, margin.top + height);
            ctx.lineTo(x, margin.top + height + 5);
            ctx.stroke();

            // Grid line
            ctx.strokeStyle = '#21262d';
            ctx.setLineDash([2, 4]);
            ctx.beginPath();
            ctx.moveTo(x, margin.top);
            ctx.lineTo(x, margin.top + height);
            ctx.stroke();
            ctx.setLineDash([]);
        }
    });

    // Y-axis ticks
    const yTickCount = 10;
    for (let i = 0; i <= yTickCount; i++) {
        const bw = (maxBW / yTickCount) * i;
        const y = margin.top + bwToCanvas(bw, 0, maxBW, height);

        ctx.fillStyle = '#8b949e';
        ctx.font = '10px sans-serif';
        ctx.textAlign = 'right';
        ctx.fillText(bw.toFixed(1), margin.left - 10, y + 4);

        // Tick mark
        ctx.strokeStyle = '#30363d';
        ctx.beginPath();
        ctx.moveTo(margin.left - 5, y);
        ctx.lineTo(margin.left, y);
        ctx.stroke();

        // Grid line
        if (i > 0) {
            ctx.strokeStyle = '#21262d';
            ctx.setLineDash([2, 4]);
            ctx.beginPath();
            ctx.moveTo(margin.left, y);
            ctx.lineTo(margin.left + width, y);
            ctx.stroke();
            ctx.setLineDash([]);
        }
    }
}

function drawLegend(margin, plotWidth, plotHeight) {
    const legendX = margin.left + plotWidth + 20;
    const legendY = margin.top + 10;
    let yOffset = 0;

    ctx.font = 'bold 12px sans-serif';
    ctx.fillStyle = '#c9d1d9';
    ctx.textAlign = 'left';
    ctx.fillText('Algorithms', legendX, legendY + yOffset);
    yOffset += 20;

    // Max performance line
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(legendX, legendY + yOffset - 4);
    ctx.lineTo(legendX + 30, legendY + yOffset - 4);
    ctx.stroke();

    ctx.font = '11px sans-serif';
    ctx.fillStyle = '#c9d1d9';
    ctx.fillText('Max Performance', legendX + 35, legendY + yOffset);
    yOffset += 20;

    // Individual algorithms
    Object.keys(ALGORITHMS).forEach(key => {
        const algo = ALGORITHMS[key];
        if (!algo.enabled) return;

        ctx.strokeStyle = algo.color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(legendX, legendY + yOffset - 4);
        ctx.lineTo(legendX + 30, legendY + yOffset - 4);
        ctx.stroke();

        ctx.font = '11px sans-serif';
        ctx.fillStyle = '#c9d1d9';
        ctx.fillText(algo.name, legendX + 35, legendY + yOffset);
        yOffset += 18;
    });
}

function updateInfoCards(perfData, maxBandwidths) {
    const infoGrid = document.getElementById('infoGrid');
    infoGrid.innerHTML = '';

    // Peak bandwidth
    const peakBW = Math.max(...maxBandwidths);
    const peakIdx = maxBandwidths.indexOf(peakBW);
    const peakSize = messageSizes[peakIdx];

    // Find which algorithm achieves peak at that size
    let peakAlgo = 'Multiple';
    Object.keys(ALGORITHMS).forEach(key => {
        const algo = ALGORITHMS[key];
        if (!algo.enabled || !perfData[key]) return;

        const point = perfData[key][peakIdx];
        if (Math.abs(point.bandwidth - peakBW) < 0.01) {
            peakAlgo = algo.name;
        }
    });

    addInfoCard('Peak Bandwidth', `${peakBW.toFixed(1)} GB/s`, `At ${formatSize(peakSize)} (${peakAlgo})`);
    addInfoCard('Total Ranks', params.totalRanks.toString(), `${params.numNodes} nodes × ${params.ranksPerNode} GPUs/node`);

    // Network capacity
    const totalNetworkBW = params.totalRanks * params.bwNetwork;
    addInfoCard('Total Network BW', `${totalNetworkBW.toFixed(0)} GB/s`, `${params.bwNetwork} GB/s per GPU`);

    // Efficiency
    const efficiency = (peakBW / totalNetworkBW) * 100;
    addInfoCard('Peak Efficiency', `${efficiency.toFixed(1)}%`, `Bandwidth utilization`);
}

function addInfoCard(title, value, label) {
    const infoGrid = document.getElementById('infoGrid');
    const card = document.createElement('div');
    card.className = 'info-card';
    card.innerHTML = `
        <h4>${title}</h4>
        <div class="info-value">${value}</div>
        <div class="info-label">${label}</div>
    `;
    infoGrid.appendChild(card);
}

function sizeToCanvas(size, minSize, maxSize, width) {
    // Log scale
    const logSize = Math.log2(size);
    const logMin = Math.log2(minSize);
    const logMax = Math.log2(maxSize);
    return (logSize - logMin) / (logMax - logMin) * width;
}

function bwToCanvas(bw, minBW, maxBW, height) {
    // Linear scale, inverted (0 at bottom)
    return height - ((bw - minBW) / (maxBW - minBW)) * height;
}

function formatSize(bytes) {
    if (bytes < 1024) return bytes + 'B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(0) + 'KB';
    if (bytes < 1024 * 1024 * 1024) return (bytes / 1024 / 1024).toFixed(0) + 'MB';
    return (bytes / 1024 / 1024 / 1024).toFixed(0) + 'GB';
}
