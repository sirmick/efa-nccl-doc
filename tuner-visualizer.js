// NCCL Algorithm and Protocol constants
const ALGORITHMS = {
    0: 'TREE',
    1: 'RING',
    2: 'COLLNET_DIRECT',
    3: 'COLLNET_CHAIN',
    4: 'NVLS',
    5: 'NVLS_TREE',
    6: 'PAT'
};

const PROTOCOLS = {
    0: 'LL',
    1: 'LL128',
    2: 'SIMPLE'
};

const TUNER_MAX_SIZE = 100 * 1024 * 1024 * 1024; // 100 GB
const TUNER_MAX_RANKS = 1024 * 1024; // 1 million

// Color scheme for algorithm+protocol combinations
const COLOR_MAP = {
    'TREE_LL': '#ff6b6b',
    'TREE_LL128': '#ff8c42',
    'TREE_SIMPLE': '#ffd93d',
    'RING_LL': '#6bcf7f',
    'RING_LL128': '#4d96ff',
    'RING_SIMPLE': '#9b59b6',
    'NVLS_SIMPLE': '#e74c3c',
    'NVLS_TREE_SIMPLE': '#f39c12',
    'NVLS_TREE_LL128': '#d35400',
    'PAT_SIMPLE': '#1abc9c',
    'COLLNET_DIRECT_SIMPLE': '#95a5a6',
    'COLLNET_CHAIN_SIMPLE': '#7f8c8d'
};

// Helper function to extend a region
function extendRegion(a, b, z) {
    const dx = b.x - a.x;
    const dy = b.y - a.y;

    if (Math.abs(dy) < 1e-10) {
        // Horizontal line
        return { x: z.x, y: a.y };
    } else if (Math.abs(dx) < 1e-10) {
        // Vertical line
        return { x: a.x, y: z.y };
    } else {
        // General line: find intersection with max bounds
        const slope = dy / dx;

        // Try extending to max x
        const y_at_max_x = a.y + slope * (z.x - a.x);
        if (y_at_max_x <= z.y && y_at_max_x >= 0) {
            return { x: z.x, y: y_at_max_x };
        }

        // Try extending to max y
        const x_at_max_y = a.x + (z.y - a.y) / slope;
        if (x_at_max_y <= z.x && x_at_max_y >= 0) {
            return { x: x_at_max_y, y: z.y };
        }

        return z;
    }
}

// Region definitions extracted from nccl_ofi_regions.cpp
const REGION_DATA = {
    p5en: {
        '8rpn': {
            AllReduce: [
                {
                    algorithm: 0, // TREE
                    protocol: 0,  // LL
                    vertices: [
                        {x: 0, y: 16},
                        {x: 196608, y: 16},
                        {x: 196608, y: 1024},
                        extendRegion({x: 196608, y: 192}, {x: 196608, y: 1024}, {x: TUNER_MAX_SIZE, y: TUNER_MAX_RANKS}),
                        {x: 0, y: TUNER_MAX_RANKS}
                    ]
                },
                {
                    algorithm: 0, // TREE
                    protocol: 1,  // LL128
                    vertices: (() => {
                        const ext1 = extendRegion({x: 196608, y: 192}, {x: 196608, y: 1024}, {x: TUNER_MAX_SIZE, y: TUNER_MAX_RANKS});
                        const ext2 = extendRegion({x: 129499136, y: 127}, {x: 218103808, y: 1024}, {x: TUNER_MAX_SIZE, y: TUNER_MAX_RANKS});
                        return [ext1, {x: 196608, y: 1024}, {x: 196608, y: 16}, {x: 13107200, y: 16}, {x: 129499136, y: 127}, {x: 218103808, y: 1024}, ext2];
                    })()
                },
                {
                    algorithm: 1, // RING
                    protocol: 1,  // LL128
                    vertices: [
                        {x: 129499136, y: 127},
                        {x: 13107200, y: 16},
                        {x: 29884416, y: 16},
                        {x: 293601280, y: 32},
                        {x: 589299712, y: 64},
                        {x: 549453824, y: 128}
                    ]
                },
                {
                    algorithm: 5, // NVLS_TREE
                    protocol: 2,  // SIMPLE
                    vertices: (() => {
                        const ext1 = extendRegion({x: 129499136, y: 127}, {x: 218103808, y: 1024}, {x: TUNER_MAX_SIZE, y: TUNER_MAX_RANKS});
                        const ext2 = extendRegion({x: 7516192768, y: 256}, {x: 17179869184, y: 448}, {x: TUNER_MAX_SIZE, y: TUNER_MAX_RANKS});
                        return [
                            ext1,
                            {x: 218103808, y: 1024},
                            {x: 129499136, y: 127},
                            {x: 549453824, y: 128},
                            {x: 589299712, y: 64},
                            {x: 293601280, y: 32},
                            {x: 29884416, y: 16},
                            {x: TUNER_MAX_SIZE, y: 16},
                            {x: 509607936, y: 32},
                            {x: 7516192768, y: 256},
                            {x: 17179869184, y: 448},
                            ext2
                        ];
                    })()
                },
                {
                    algorithm: 1, // RING
                    protocol: 2,  // SIMPLE
                    vertices: (() => {
                        const ext = extendRegion({x: 7516192768, y: 256}, {x: 17179869184, y: 448}, {x: TUNER_MAX_SIZE, y: TUNER_MAX_RANKS});
                        return [ext, {x: 17179869184, y: 448}, {x: 7516192768, y: 256}, {x: 509607936, y: 32}, {x: TUNER_MAX_SIZE, y: 16}];
                    })()
                }
            ],
            AllGather: [
                {
                    algorithm: 1, // RING
                    protocol: 0,  // LL
                    vertices: (() => {
                        const ext = extendRegion({x: 28305408, y: 768}, {x: 33554432, y: 1024}, {x: TUNER_MAX_SIZE, y: TUNER_MAX_RANKS});
                        return [
                            {x: 0, y: 16},
                            {x: 786432, y: 16},
                            {x: 4718592, y: 128},
                            {x: 13630464, y: 384},
                            {x: 19922944, y: 512},
                            {x: 28305408, y: 768},
                            {x: 33554432, y: 1024},
                            ext,
                            {x: 0, y: TUNER_MAX_RANKS}
                        ];
                    })()
                },
                {
                    algorithm: 1, // RING
                    protocol: 1,  // LL128
                    vertices: (() => {
                        const ext1 = extendRegion({x: 28305408, y: 768}, {x: 33554432, y: 1024}, {x: TUNER_MAX_SIZE, y: TUNER_MAX_RANKS});
                        const ext2 = extendRegion({x: 7247754240, y: 384}, {x: 17179865088, y: 896}, {x: TUNER_MAX_SIZE, y: TUNER_MAX_RANKS});
                        return [
                            ext1,
                            {x: 33554432, y: 1024},
                            {x: 28305408, y: 768},
                            {x: 19922944, y: 512},
                            {x: 13630464, y: 384},
                            {x: 4718592, y: 128},
                            {x: 786432, y: 16},
                            {x: 236978176, y: 16},
                            {x: 956301312, y: 64},
                            {x: 7247754240, y: 384},
                            {x: 17179865088, y: 896},
                            ext2
                        ];
                    })()
                },
                {
                    algorithm: 1, // RING
                    protocol: 2,  // SIMPLE
                    vertices: (() => {
                        const ext = extendRegion({x: 7247754240, y: 384}, {x: 17179865088, y: 896}, {x: TUNER_MAX_SIZE, y: TUNER_MAX_RANKS});
                        return [
                            ext,
                            {x: 17179865088, y: 896},
                            {x: 7247754240, y: 384},
                            {x: 956301312, y: 64},
                            {x: 494927872, y: 32},
                            {x: 236978176, y: 16},
                            {x: TUNER_MAX_SIZE, y: 16}
                        ];
                    })()
                }
            ],
            ReduceScatter: [
                {
                    algorithm: 1, // RING
                    protocol: 0,  // LL
                    vertices: (() => {
                        const ext = extendRegion({x: 7340032, y: 256}, {x: 18874368, y: 512}, {x: TUNER_MAX_SIZE, y: TUNER_MAX_RANKS});
                        return [
                            {x: 0, y: 16},
                            {x: 786432, y: 16},
                            {x: 4718592, y: 128},
                            {x: 7340032, y: 256},
                            {x: 18874368, y: 512},
                            ext,
                            {x: 0, y: TUNER_MAX_RANKS}
                        ];
                    })()
                },
                {
                    algorithm: 1, // RING
                    protocol: 1,  // LL128
                    vertices: (() => {
                        const ext1 = extendRegion({x: 7340032, y: 256}, {x: 18874368, y: 512}, {x: TUNER_MAX_SIZE, y: TUNER_MAX_RANKS});
                        const ext2 = extendRegion({x: 7247757312, y: 256}, {x: 17179869184, y: 640}, {x: TUNER_MAX_SIZE, y: TUNER_MAX_RANKS});
                        return [
                            ext1,
                            {x: 18874368, y: 512},
                            {x: 7340032, y: 256},
                            {x: 4718592, y: 128},
                            {x: 786432, y: 16},
                            {x: 291504128, y: 16},
                            {x: 589299712, y: 32},
                            {x: 2415919104, y: 128},
                            {x: 7247757312, y: 256},
                            {x: 17179869184, y: 640},
                            ext2
                        ];
                    })()
                },
                {
                    algorithm: 1, // RING
                    protocol: 2,  // SIMPLE
                    vertices: (() => {
                        const ext = extendRegion({x: 7247757312, y: 256}, {x: 17179869184, y: 640}, {x: TUNER_MAX_SIZE, y: TUNER_MAX_RANKS});
                        return [
                            ext,
                            {x: 17179869184, y: 640},
                            {x: 7247757312, y: 256},
                            {x: 2415919104, y: 128},
                            {x: 589299712, y: 32},
                            {x: 291504128, y: 16},
                            {x: TUNER_MAX_SIZE, y: 16}
                        ];
                    })()
                }
            ]
        },
        '1rpn': {
            AllReduce: [
                {
                    algorithm: 0, // TREE
                    protocol: 0,  // LL
                    vertices: [
                        {x: 0, y: 2},
                        {x: 65536, y: 2},
                        {x: 65536, y: 64},
                        {x: 65536, y: TUNER_MAX_RANKS}
                    ]
                },
                {
                    algorithm: 0, // TREE
                    protocol: 1,  // LL128
                    vertices: (() => {
                        const ext = extendRegion({x: 524288, y: 8}, {x: 1048576, y: 96}, {x: TUNER_MAX_SIZE, y: TUNER_MAX_RANKS});
                        return [
                            {x: 65536, y: TUNER_MAX_RANKS},
                            {x: 65536, y: 64},
                            {x: 65536, y: 2},
                            {x: 262144, y: 2},
                            {x: 524288, y: 8},
                            {x: 1048576, y: 96},
                            ext
                        ];
                    })()
                },
                {
                    algorithm: 0, // TREE
                    protocol: 2,  // SIMPLE
                    vertices: (() => {
                        const ext1 = extendRegion({x: 524288, y: 8}, {x: 1048576, y: 96}, {x: TUNER_MAX_SIZE, y: TUNER_MAX_RANKS});
                        const ext2 = extendRegion({x: 8388608, y: 32}, {x: 33554432, y: 128}, {x: TUNER_MAX_SIZE, y: TUNER_MAX_RANKS});
                        return [
                            ext1,
                            {x: 1048576, y: 96},
                            {x: 524288, y: 8},
                            {x: 262144, y: 2},
                            {x: 8388608, y: 32},
                            {x: 33554432, y: 128},
                            ext2
                        ];
                    })()
                },
                {
                    algorithm: 1, // RING
                    protocol: 1,  // LL128
                    vertices: (() => {
                        const ext1 = extendRegion({x: 8388608, y: 32}, {x: 33554432, y: 128}, {x: TUNER_MAX_SIZE, y: TUNER_MAX_RANKS});
                        const ext2 = extendRegion({x: 50331648, y: 16}, {x: 301989888, y: 128}, {x: TUNER_MAX_SIZE, y: TUNER_MAX_RANKS});
                        return [
                            ext1,
                            {x: 33554432, y: 128},
                            {x: 8388608, y: 32},
                            {x: 262144, y: 2},
                            {x: 6291456, y: 2},
                            {x: 50331648, y: 16},
                            {x: 301989888, y: 128},
                            ext2
                        ];
                    })()
                },
                {
                    algorithm: 1, // RING
                    protocol: 2,  // SIMPLE
                    vertices: (() => {
                        const ext = extendRegion({x: 50331648, y: 16}, {x: 301989888, y: 128}, {x: TUNER_MAX_SIZE, y: TUNER_MAX_RANKS});
                        return [
                            ext,
                            {x: 301989888, y: 128},
                            {x: 50331648, y: 16},
                            {x: 6291456, y: 2},
                            {x: TUNER_MAX_SIZE, y: 2}
                        ];
                    })()
                }
            ],
            AllGather: [
                {
                    algorithm: 6, // PAT
                    protocol: 2,  // SIMPLE
                    vertices: (() => {
                        const ext = extendRegion({x: 50331648, y: 64}, {x: 117440512, y: 128}, {x: TUNER_MAX_SIZE, y: TUNER_MAX_RANKS});
                        return [
                            {x: 0, y: 2},
                            {x: 65536, y: 2},
                            {x: 1048576, y: 2},
                            {x: 16777216, y: 32},
                            {x: 50331648, y: 64},
                            {x: 117440512, y: 128},
                            ext,
                            {x: TUNER_MAX_SIZE, y: TUNER_MAX_RANKS},
                            {x: 65536, y: TUNER_MAX_RANKS},
                            {x: 0, y: TUNER_MAX_RANKS}
                        ];
                    })()
                },
                {
                    algorithm: 1, // RING
                    protocol: 1,  // LL128
                    vertices: (() => {
                        const ext1 = extendRegion({x: 50331648, y: 64}, {x: 117440512, y: 128}, {x: TUNER_MAX_SIZE, y: TUNER_MAX_RANKS});
                        const ext2 = extendRegion({x: 50331648, y: 16}, {x: 301989888, y: 128}, {x: TUNER_MAX_SIZE, y: TUNER_MAX_RANKS});
                        return [
                            ext1,
                            {x: 117440512, y: 128},
                            {x: 50331648, y: 64},
                            {x: 16777216, y: 32},
                            {x: 1048576, y: 2},
                            {x: 4194304, y: 2},
                            {x: 50331648, y: 16},
                            {x: 301989888, y: 128},
                            ext2
                        ];
                    })()
                },
                {
                    algorithm: 1, // RING
                    protocol: 2,  // SIMPLE
                    vertices: (() => {
                        const ext = extendRegion({x: 50331648, y: 16}, {x: 301989888, y: 128}, {x: TUNER_MAX_SIZE, y: TUNER_MAX_RANKS});
                        return [
                            ext,
                            {x: 301989888, y: 128},
                            {x: 50331648, y: 16},
                            {x: 4194304, y: 2},
                            {x: TUNER_MAX_SIZE, y: 2}
                        ];
                    })()
                }
            ],
            ReduceScatter: [
                {
                    algorithm: 6, // PAT
                    protocol: 2,  // SIMPLE
                    vertices: (() => {
                        const ext = extendRegion({x: 50331648, y: 64}, {x: 117440512, y: 128}, {x: TUNER_MAX_SIZE, y: TUNER_MAX_RANKS});
                        return [
                            {x: 0, y: 2},
                            {x: 65536, y: 2},
                            {x: 1048576, y: 2},
                            {x: 16777216, y: 32},
                            {x: 50331648, y: 64},
                            {x: 117440512, y: 128},
                            ext,
                            {x: TUNER_MAX_SIZE, y: TUNER_MAX_RANKS},
                            {x: 65536, y: TUNER_MAX_RANKS},
                            {x: 0, y: TUNER_MAX_RANKS}
                        ];
                    })()
                },
                {
                    algorithm: 1, // RING
                    protocol: 1,  // LL128
                    vertices: (() => {
                        const ext1 = extendRegion({x: 50331648, y: 64}, {x: 117440512, y: 128}, {x: TUNER_MAX_SIZE, y: TUNER_MAX_RANKS});
                        const ext2 = extendRegion({x: 50331648, y: 16}, {x: 301989888, y: 128}, {x: TUNER_MAX_SIZE, y: TUNER_MAX_RANKS});
                        return [
                            ext1,
                            {x: 117440512, y: 128},
                            {x: 50331648, y: 64},
                            {x: 16777216, y: 32},
                            {x: 1048576, y: 2},
                            {x: 4194304, y: 2},
                            {x: 50331648, y: 16},
                            {x: 301989888, y: 128},
                            ext2
                        ];
                    })()
                },
                {
                    algorithm: 1, // RING
                    protocol: 2,  // SIMPLE
                    vertices: (() => {
                        const ext = extendRegion({x: 50331648, y: 16}, {x: 301989888, y: 128}, {x: TUNER_MAX_SIZE, y: TUNER_MAX_RANKS});
                        return [
                            ext,
                            {x: 301989888, y: 128},
                            {x: 50331648, y: 16},
                            {x: 4194304, y: 2},
                            {x: TUNER_MAX_SIZE, y: 2}
                        ];
                    })()
                }
            ]
        }
    },
    p5: {
        // Placeholder for p5/p5e data - would be populated similarly
        '8rpn': {
            AllReduce: [],
            AllGather: [],
            ReduceScatter: []
        },
        '1rpn': {
            AllReduce: [],
            AllGather: [],
            ReduceScatter: []
        }
    }
};

// Canvas and state
let canvas, ctx;
let currentPlatform = 'p5en';
let currentCollective = 'AllReduce';
let currentConfig = '8rpn';
let testPointResult = null;

// Initialize
window.addEventListener('DOMContentLoaded', () => {
    canvas = document.getElementById('regionCanvas');
    ctx = canvas.getContext('2d');

    // Set up event listeners
    document.getElementById('platform').addEventListener('change', (e) => {
        currentPlatform = e.target.value;
        draw();
    });

    document.getElementById('collective').addEventListener('change', (e) => {
        currentCollective = e.target.value;
        draw();
    });

    document.getElementById('config').addEventListener('change', (e) => {
        currentConfig = e.target.value;
        draw();
    });

    canvas.addEventListener('mousemove', handleMouseMove);
    canvas.addEventListener('mouseleave', () => {
        document.getElementById('tooltip').classList.remove('show');
    });

    updateLegend();
    draw();
});

function handleMouseMove(e) {
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    // Convert canvas coordinates to data coordinates
    const point = canvasToData(x, y);

    // Find which region contains this point
    const regions = getRegions();
    const matchedRegion = findRegion(point, regions);

    // Update tooltip
    const tooltip = document.getElementById('tooltip');
    if (matchedRegion) {
        const algoName = ALGORITHMS[matchedRegion.algorithm];
        const protoName = PROTOCOLS[matchedRegion.protocol];
        tooltip.innerHTML = `
            <strong>${algoName} + ${protoName}</strong><br>
            Size: ${formatBytes(point.x)}<br>
            Ranks: ${Math.round(point.y)}
        `;
        tooltip.style.left = (e.clientX + 15) + 'px';
        tooltip.style.top = (e.clientY + 15) + 'px';
        tooltip.classList.add('show');
    } else {
        tooltip.classList.remove('show');
    }

    // Update info panel
    updateInfo(point, matchedRegion);
}

function testPoint() {
    const size = parseInt(document.getElementById('testSize').value);
    const ranks = parseInt(document.getElementById('testRanks').value);

    if (isNaN(size) || isNaN(ranks)) {
        alert('Please enter valid numbers');
        return;
    }

    const point = { x: size, y: ranks };
    const regions = getRegions();
    const matchedRegion = findRegion(point, regions);

    testPointResult = { point, matchedRegion };
    updateInfo(point, matchedRegion);
    draw();
}

function getRegions() {
    const data = REGION_DATA[currentPlatform];
    if (!data) return [];
    const config = data[currentConfig];
    if (!config) return [];
    return config[currentCollective] || [];
}

function findRegion(point, regions) {
    for (const region of regions) {
        if (isInsideRegion(point, region)) {
            return region;
        }
    }
    return null;
}

function isInsideRegion(point, region) {
    const vertices = region.vertices;
    const n = vertices.length;
    let inside = false;

    for (let i = 0, j = n - 1; i < n; j = i++) {
        const xi = vertices[i].x, yi = vertices[i].y;
        const xj = vertices[j].x, yj = vertices[j].y;

        const intersect = ((yi > point.y) !== (yj > point.y))
            && (point.x < (xj - xi) * (point.y - yi) / (yj - yi) + xi);
        if (intersect) inside = !inside;
    }

    return inside;
}

function updateLegend() {
    const legendGrid = document.getElementById('legendGrid');
    legendGrid.innerHTML = '';

    const combinations = new Set();
    for (const platform in REGION_DATA) {
        for (const config in REGION_DATA[platform]) {
            for (const collective in REGION_DATA[platform][config]) {
                const regions = REGION_DATA[platform][config][collective];
                regions.forEach(r => {
                    const key = `${ALGORITHMS[r.algorithm]}_${PROTOCOLS[r.protocol]}`;
                    combinations.add(key);
                });
            }
        }
    }

    Array.from(combinations).sort().forEach(combo => {
        if (COLOR_MAP[combo]) {
            const item = document.createElement('div');
            item.className = 'legend-item';
            item.innerHTML = `
                <div class="legend-color" style="background-color: ${COLOR_MAP[combo]}"></div>
                <div class="legend-label">${combo.replace('_', ' + ')}</div>
            `;
            legendGrid.appendChild(item);
        }
    });
}

function updateInfo(point, matchedRegion) {
    const infoContent = document.getElementById('infoContent');

    if (!matchedRegion) {
        infoContent.innerHTML = `
            <div class="info-item">
                <div class="info-label">Message Size</div>
                <div class="info-value">${formatBytes(point.x)}</div>
            </div>
            <div class="info-item">
                <div class="info-label">Rank Count</div>
                <div class="info-value">${Math.round(point.y)}</div>
            </div>
            <div class="info-item">
                <div class="info-label">Selected Algorithm</div>
                <div class="info-value" style="color: #f85149">No region match - NCCL default tuning</div>
            </div>
        `;
    } else {
        const algoName = ALGORITHMS[matchedRegion.algorithm];
        const protoName = PROTOCOLS[matchedRegion.protocol];
        const combo = `${algoName}_${protoName}`;
        const color = COLOR_MAP[combo] || '#58a6ff';

        infoContent.innerHTML = `
            <div class="info-item">
                <div class="info-label">Message Size</div>
                <div class="info-value">${formatBytes(point.x)}</div>
            </div>
            <div class="info-item">
                <div class="info-label">Rank Count</div>
                <div class="info-value">${Math.round(point.y)}</div>
            </div>
            <div class="info-item">
                <div class="info-label">Selected Algorithm</div>
                <div class="info-value highlight" style="color: ${color}">${algoName}</div>
            </div>
            <div class="info-item">
                <div class="info-label">Selected Protocol</div>
                <div class="info-value highlight" style="color: ${color}">${protoName}</div>
            </div>
        `;
    }
}

function draw() {
    const width = canvas.width;
    const height = canvas.height;
    const margin = { top: 40, right: 40, bottom: 60, left: 80 };
    const plotWidth = width - margin.left - margin.right;
    const plotHeight = height - margin.top - margin.bottom;

    // Clear canvas
    ctx.fillStyle = '#0d1117';
    ctx.fillRect(0, 0, width, height);

    // Get regions
    const regions = getRegions();
    if (regions.length === 0) {
        ctx.fillStyle = '#8b949e';
        ctx.font = '14px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('No region data available for this configuration', width / 2, height / 2);
        return;
    }

    // Calculate data bounds
    const bounds = calculateBounds(regions);

    // Draw regions
    ctx.save();
    ctx.translate(margin.left, margin.top);
    ctx.beginPath();
    ctx.rect(0, 0, plotWidth, plotHeight);
    ctx.clip();

    regions.forEach(region => {
        drawRegion(region, bounds, plotWidth, plotHeight);
    });

    // Draw test point if exists
    if (testPointResult) {
        const px = dataToCanvas(testPointResult.point.x, bounds.minX, bounds.maxX, plotWidth, true);
        const py = dataToCanvas(testPointResult.point.y, bounds.minY, bounds.maxY, plotHeight, false);

        ctx.strokeStyle = '#ffffff';
        ctx.fillStyle = testPointResult.matchedRegion ? '#238636' : '#f85149';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(px, py, 6, 0, 2 * Math.PI);
        ctx.fill();
        ctx.stroke();
    }

    ctx.restore();

    // Draw axes
    drawAxes(margin, plotWidth, plotHeight, bounds);
}

function calculateBounds(regions) {
    // Cap visualization at reasonable values
    const MAX_DISPLAY_SIZE = 16 * 1024 * 1024 * 1024;  // 16 GB
    const MAX_DISPLAY_RANKS = 2048;  // Reasonable max cluster size

    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;

    regions.forEach(region => {
        region.vertices.forEach(v => {
            const x = Math.min(v.x, MAX_DISPLAY_SIZE);
            const y = Math.min(v.y, MAX_DISPLAY_RANKS);
            minX = Math.min(minX, x);
            maxX = Math.max(maxX, x);
            minY = Math.min(minY, y);
            maxY = Math.max(maxY, y);
        });
    });

    // Use log scale for x-axis, linear for y-axis
    minX = Math.max(1, minX);
    minY = Math.max(1, minY);

    // Force maximum bounds to display limits
    maxX = MAX_DISPLAY_SIZE;
    maxY = MAX_DISPLAY_RANKS;

    return { minX, maxX, minY, maxY };
}

function drawRegion(region, bounds, width, height) {
    const algoName = ALGORITHMS[region.algorithm];
    const protoName = PROTOCOLS[region.protocol];
    const combo = `${algoName}_${protoName}`;
    const color = COLOR_MAP[combo] || '#58a6ff';

    ctx.beginPath();
    region.vertices.forEach((v, i) => {
        // Clamp vertices to display bounds for rendering
        const clampedX = Math.min(v.x, bounds.maxX);
        const clampedY = Math.min(v.y, bounds.maxY);

        const x = dataToCanvas(clampedX, bounds.minX, bounds.maxX, width, true);
        const y = dataToCanvas(clampedY, bounds.minY, bounds.maxY, height, false);
        if (i === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
    });
    ctx.closePath();

    ctx.fillStyle = color + '33'; // 20% opacity
    ctx.fill();
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.stroke();
}

function drawAxes(margin, width, height, bounds) {
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
    ctx.fillText('Message Size (bytes, log scale)', margin.left + width / 2, margin.top + height + 40);

    // Y-axis label
    ctx.save();
    ctx.translate(margin.left - 60, margin.top + height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Number of Ranks', 0, 0);
    ctx.restore();

    // X-axis ticks
    const xTicks = generateLogTicks(bounds.minX, bounds.maxX);
    xTicks.forEach(tick => {
        const x = margin.left + dataToCanvas(tick, bounds.minX, bounds.maxX, width, true);
        ctx.fillStyle = '#8b949e';
        ctx.font = '10px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(formatBytes(tick), x, margin.top + height + 20);

        // Tick mark
        ctx.strokeStyle = '#30363d';
        ctx.beginPath();
        ctx.moveTo(x, margin.top + height);
        ctx.lineTo(x, margin.top + height + 5);
        ctx.stroke();
    });

    // Y-axis ticks
    const yTicks = generateLinearTicks(bounds.minY, bounds.maxY);
    yTicks.forEach(tick => {
        const y = margin.top + dataToCanvas(tick, bounds.minY, bounds.maxY, height, false);
        ctx.fillStyle = '#8b949e';
        ctx.font = '10px sans-serif';
        ctx.textAlign = 'right';
        ctx.fillText(tick.toString(), margin.left - 10, y + 4);

        // Tick mark
        ctx.strokeStyle = '#30363d';
        ctx.beginPath();
        ctx.moveTo(margin.left - 5, y);
        ctx.lineTo(margin.left, y);
        ctx.stroke();
    });
}

function dataToCanvas(value, min, max, size, logScale) {
    if (logScale) {
        value = Math.max(1, value);
        min = Math.max(1, min);
        max = Math.max(1, max);
        const logValue = Math.log10(value);
        const logMin = Math.log10(min);
        const logMax = Math.log10(max);
        return (logValue - logMin) / (logMax - logMin) * size;
    } else {
        return size - ((value - min) / (max - min) * size);
    }
}

function canvasToData(canvasX, canvasY) {
    const margin = { top: 40, right: 40, bottom: 60, left: 80 };
    const width = canvas.width - margin.left - margin.right;
    const height = canvas.height - margin.top - margin.bottom;

    const regions = getRegions();
    const bounds = calculateBounds(regions);

    // Adjust for margins
    canvasX -= margin.left;
    canvasY -= margin.top;

    // X (log scale)
    const xRatio = canvasX / width;
    const logMin = Math.log10(Math.max(1, bounds.minX));
    const logMax = Math.log10(Math.max(1, bounds.maxX));
    const logX = logMin + xRatio * (logMax - logMin);
    const x = Math.pow(10, logX);

    // Y (linear scale, inverted)
    const yRatio = 1 - (canvasY / height);
    const y = bounds.minY + yRatio * (bounds.maxY - bounds.minY);

    return { x, y };
}

function generateLogTicks(min, max) {
    const ticks = [];
    const minLog = Math.floor(Math.log10(Math.max(1, min)));
    const maxLog = Math.ceil(Math.log10(Math.max(1, max)));

    for (let i = minLog; i <= maxLog; i++) {
        ticks.push(Math.pow(10, i));
    }

    return ticks;
}

function generateLinearTicks(min, max) {
    const ticks = [];
    const range = max - min;
    const step = Math.pow(10, Math.floor(Math.log10(range / 5)));
    const start = Math.ceil(min / step) * step;

    for (let val = start; val <= max; val += step) {
        ticks.push(val);
    }

    return ticks;
}

function formatBytes(bytes) {
    if (bytes === 0) return '0 B';
    if (bytes >= TUNER_MAX_SIZE) return 'MAX';

    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));

    return Math.round(bytes / Math.pow(k, i)) + ' ' + sizes[i];
}
