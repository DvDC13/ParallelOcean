import { OrbitCamera, mat4Multiply } from "./camera.js";
import { generateInitialSpectrum } from './spectrum.js';

// ===== Configuration =====
const N = 256;                              // FFT grid size
const CASCADE_PATCHES = [1000, 200, 50];    // 3 scales: large swells, medium waves, small ripples
const MAIN_PATCH = 1000;                    // tile size in meters

// LOD: high detail close, low detail far
const LOD_LEVELS = [
    { gridN: 256, maxDist: 2 },
    { gridN: 64,  maxDist: 5 },
    { gridN: 16,  maxDist: 10 },
];
const TILES_PER_SIDE = 21;

// Tweakable parameters
const params = {
    windSpeed: 40,
    windAngle: 45,
    amplitude0: 600,
    amplitude1: 150,
    amplitude2: 20,
    choppiness: 1.8,
};

// ===== Helper: create a flat grid mesh =====
function createOceanGrid(gridN, size) {
    const verts = gridN + 1;
    const vertices = new Float32Array(verts * verts * 3);
    const half = size / 2;
    const step = size / gridN;
    for (let z = 0; z < verts; z++) {
        for (let x = 0; x < verts; x++) {
            const i = (z * verts + x) * 3;
            vertices[i]     = -half + x * step;
            vertices[i + 1] = 0;
            vertices[i + 2] = -half + z * step;
        }
    }
    const indices = new Uint32Array(gridN * gridN * 6);
    let idx = 0;
    for (let z = 0; z < gridN; z++) {
        for (let x = 0; x < gridN; x++) {
            const tl = z * verts + x;
            const tr = tl + 1;
            const bl = tl + verts;
            const br = bl + 1;
            indices[idx++] = tl; indices[idx++] = bl; indices[idx++] = tr;
            indices[idx++] = tr; indices[idx++] = bl; indices[idx++] = br;
        }
    }
    return { vertices, indices };
}

// ===== Helper: assign tiles to LOD levels =====
function buildTileOffsets() {
    const half = Math.floor(TILES_PER_SIDE / 2);
    const lodTiles = LOD_LEVELS.map(() => []);
    for (let z = 0; z < TILES_PER_SIDE; z++) {
        for (let x = 0; x < TILES_PER_SIDE; x++) {
            const dist = Math.max(Math.abs(x - half), Math.abs(z - half));
            const tx = (x - half) * MAIN_PATCH;
            const tz = (z - half) * MAIN_PATCH;
            for (let i = 0; i < LOD_LEVELS.length; i++) {
                const prevMax = i > 0 ? LOD_LEVELS[i - 1].maxDist : -1;
                if (dist > prevMax && dist <= LOD_LEVELS[i].maxDist) {
                    lodTiles[i].push(tx, tz);
                    break;
                }
            }
        }
    }
    const allOffsets = [];
    const lodRanges = [];
    for (let i = 0; i < lodTiles.length; i++) {
        const start = allOffsets.length / 2;
        allOffsets.push(...lodTiles[i]);
        lodRanges.push({ firstInstance: start, instanceCount: lodTiles[i].length / 2 });
    }
    return { offsets: new Float32Array(allOffsets), lodRanges };
}

// ===== Main =====
async function main() {
    // --- Get the canvas and WebGPU device ---
    const canvas = document.getElementById('canvas');
    if (!navigator.gpu) { document.body.innerText = 'WebGPU not supported'; return; }

    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter.requestDevice({
        requiredLimits: {
            maxStorageBuffersPerShaderStage: 16,
        },
    });
    const context = canvas.getContext('webgpu');
    const format = navigator.gpu.getPreferredCanvasFormat();
    context.configure({ device, format, alphaMode: 'opaque' });

    // Resize canvas to fill screen
    function resize() {
        const dpr = window.devicePixelRatio || 1;
        canvas.width = Math.floor(canvas.clientWidth * dpr);
        canvas.height = Math.floor(canvas.clientHeight * dpr);
    }
    resize();
    window.addEventListener('resize', resize);

    // --- Camera ---
    const camera = new OrbitCamera(canvas);

    // --- Create LOD meshes (GPU buffers) ---
    const lodMeshes = LOD_LEVELS.map(({ gridN }) => {
        const { vertices, indices } = createOceanGrid(gridN, MAIN_PATCH);
        const vb = device.createBuffer({ size: vertices.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(vb, 0, vertices);
        const ib = device.createBuffer({ size: indices.byteLength, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(ib, 0, indices);
        return { vb, ib, indexCount: indices.length };
    });

    // --- Tile offsets buffer ---
    const { offsets: tileOffsetsData, lodRanges } = buildTileOffsets();
    const tileOffsetsBuffer = device.createBuffer({
        size: tileOffsetsData.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(tileOffsetsBuffer, 0, tileOffsetsData);

    // --- Create GPU buffers for 3 cascades ---
    const bufSize = N * N;
    const fftTemp = device.createBuffer({ size: bufSize * 8, usage: GPUBufferUsage.STORAGE });

    const cascades = CASCADE_PATCHES.map(() => ({
        h0Buf:       device.createBuffer({ size: bufSize * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }),
        omegaBuf:    device.createBuffer({ size: bufSize * 4,  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }),
        htBuf:       device.createBuffer({ size: bufSize * 8,  usage: GPUBufferUsage.STORAGE }),
        dxtBuf:      device.createBuffer({ size: bufSize * 8,  usage: GPUBufferUsage.STORAGE }),
        dztBuf:      device.createBuffer({ size: bufSize * 8,  usage: GPUBufferUsage.STORAGE }),
        timeParamBuf: device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }),
    }));

    // --- Generate initial spectrum on CPU and upload ---
    function regenerateSpectrum() {
        const amps = [params.amplitude0, params.amplitude1, params.amplitude2];
        for (let i = 0; i < 3; i++) {
            const { h0, omega } = generateInitialSpectrum(N, CASCADE_PATCHES[i], {
                windSpeed: params.windSpeed,
                windAngle: params.windAngle * Math.PI / 180,
                amplitude: amps[i],
            });
            device.queue.writeBuffer(cascades[i].h0Buf, 0, h0);
            device.queue.writeBuffer(cascades[i].omegaBuf, 0, omega);
        }
    }
    regenerateSpectrum();

    // --- Load and compile the shader ---
    const shaderCode = await (await fetch('ocean.wgsl')).text();
    const shaderModule = device.createShaderModule({ code: shaderCode });

    // --- Create compute pipelines ---
    const timeEvoPipeline = device.createComputePipeline({
        layout: 'auto',
        compute: { module: shaderModule, entryPoint: 'timeEvolution' },
    });
    const fftRowPipeline = device.createComputePipeline({
        layout: 'auto',
        compute: { module: shaderModule, entryPoint: 'fftRowPass' },
    });
    const fftColPipeline = device.createComputePipeline({
        layout: 'auto',
        compute: { module: shaderModule, entryPoint: 'fftColPass' },
    });

    // --- Bind groups for compute ---
    const timeEvoBGs = cascades.map(c => device.createBindGroup({
        layout: timeEvoPipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: c.h0Buf } },
            { binding: 1, resource: { buffer: c.omegaBuf } },
            { binding: 2, resource: { buffer: c.htBuf } },
            { binding: 3, resource: { buffer: c.dxtBuf } },
            { binding: 4, resource: { buffer: c.dztBuf } },
            { binding: 5, resource: { buffer: c.timeParamBuf } },
        ],
    }));

    function createFFTBindGroups(signalBuf) {
        return {
            rowBG: device.createBindGroup({
                layout: fftRowPipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: signalBuf } },
                    { binding: 1, resource: { buffer: fftTemp } },
                ],
            }),
            colBG: device.createBindGroup({
                layout: fftColPipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: fftTemp } },
                    { binding: 1, resource: { buffer: signalBuf } },
                ],
            }),
        };
    }

    const cascadeFFT = cascades.map(c => ({
        ht:  createFFTBindGroups(c.htBuf),
        dxt: createFFTBindGroups(c.dxtBuf),
        dzt: createFFTBindGroups(c.dztBuf),
    }));

    // --- Render pipeline (draws the ocean triangles) ---
    // Uniform buffer: 112 bytes (matches RenderUniforms in shader)
    const renderUniformBuf = device.createBuffer({
        size: 112,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const renderPipeline = device.createRenderPipeline({
        layout: 'auto',
        vertex: {
            module: shaderModule,
            entryPoint: 'vs',
            buffers: [{
                arrayStride: 12,  // 3 floats * 4 bytes
                attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }],
            }],
        },
        fragment: {
            module: shaderModule,
            entryPoint: 'fs',
            targets: [{ format }],
        },
        primitive: { topology: 'triangle-list', cullMode: 'none' },
        depthStencil: {
            format: 'depth24plus',
            depthWriteEnabled: true,
            depthCompare: 'less',
        },
    });

    const renderBG = device.createBindGroup({
        layout: renderPipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0,  resource: { buffer: renderUniformBuf } },
            { binding: 1,  resource: { buffer: cascades[0].htBuf } },
            { binding: 2,  resource: { buffer: cascades[0].dxtBuf } },
            { binding: 3,  resource: { buffer: cascades[0].dztBuf } },
            { binding: 4,  resource: { buffer: tileOffsetsBuffer } },
            { binding: 5,  resource: { buffer: cascades[1].htBuf } },
            { binding: 6,  resource: { buffer: cascades[1].dxtBuf } },
            { binding: 7,  resource: { buffer: cascades[1].dztBuf } },
            { binding: 8,  resource: { buffer: cascades[2].htBuf } },
            { binding: 9,  resource: { buffer: cascades[2].dxtBuf } },
            { binding: 10, resource: { buffer: cascades[2].dztBuf } },
        ],
    });

    // --- Depth texture (recreated on resize) ---
    let depthTexture = null;

    function ensureDepthTexture(w, h) {
        if (depthTexture && depthTexture.width === w && depthTexture.height === h) return;
        if (depthTexture) depthTexture.destroy();
        depthTexture = device.createTexture({
            size: [w, h],
            format: 'depth24plus',
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
        });
    }

    // --- Sun direction ---
    function sunDir() {
        const el = 56 * Math.PI / 180;
        const az = 31 * Math.PI / 180;
        return [
            Math.cos(el) * Math.sin(az),
            Math.sin(el),
            Math.cos(el) * Math.cos(az),
        ];
    }

    // ===== Frame loop =====
    let accTime = 0;
    let lastTime = performance.now();

    function frame() {
        const now = performance.now();
        const dt = (now - lastTime) / 1000;
        lastTime = now;
        accTime += dt;

        const w = canvas.width;
        const h = canvas.height;
        ensureDepthTexture(w, h);

        // Upload time params for each cascade
        for (let i = 0; i < 3; i++) {
            const buf = new ArrayBuffer(16);
            new Float32Array(buf, 0, 1)[0] = accTime;
            new Uint32Array(buf, 4, 1)[0] = N;
            new Float32Array(buf, 8, 1)[0] = CASCADE_PATCHES[i];
            new Float32Array(buf, 12, 1)[0] = params.choppiness;
            device.queue.writeBuffer(cascades[i].timeParamBuf, 0, buf);
        }

        // Upload render uniforms (112 bytes)
        const view = camera.viewMatrix();
        const proj = camera.projMatrix();
        const viewProj = mat4Multiply(proj, view);
        const eye = camera.eye;
        const sun = sunDir();

        const uniformData = new ArrayBuffer(112);
        const f = new Float32Array(uniformData);
        f.set(viewProj, 0);         // 0-15: viewProj matrix
        f[16] = eye[0];             // 16: eyePos.x
        f[17] = eye[1];             // 17: eyePos.y
        f[18] = eye[2];             // 18: eyePos.z
        f[19] = accTime;            // 19: time
        f[20] = sun[0];             // 20: sunDir.x
        f[21] = sun[1];             // 21: sunDir.y
        f[22] = sun[2];             // 22: sunDir.z
        f[23] = params.choppiness;  // 23: choppiness
        f[24] = CASCADE_PATCHES[0]; // 24: cascadePatch0
        f[25] = CASCADE_PATCHES[1]; // 25: cascadePatch1
        f[26] = CASCADE_PATCHES[2]; // 26: cascadePatch2
        f[27] = 0;                  // 27: pad
        device.queue.writeBuffer(renderUniformBuf, 0, uniformData);

        // --- GPU commands ---
        const encoder = device.createCommandEncoder();

        // Compute pass: time evolution + FFT for all 3 cascades
        const comp = encoder.beginComputePass();
        for (let i = 0; i < 3; i++) {
            // Time evolution
            comp.setPipeline(timeEvoPipeline);
            comp.setBindGroup(0, timeEvoBGs[i]);
            comp.dispatchWorkgroups(Math.ceil(N / 16), Math.ceil(N / 16));

            // FFT for ht, dxt, dzt (each needs row pass then column pass)
            const fft = cascadeFFT[i];
            for (const signal of [fft.ht, fft.dxt, fft.dzt]) {
                comp.setPipeline(fftRowPipeline);
                comp.setBindGroup(0, signal.rowBG);
                comp.dispatchWorkgroups(1, N);

                comp.setPipeline(fftColPipeline);
                comp.setBindGroup(0, signal.colBG);
                comp.dispatchWorkgroups(1, N);
            }
        }
        comp.end();

        // Render pass: draw the ocean
        const renderPass = encoder.beginRenderPass({
            colorAttachments: [{
                view: context.getCurrentTexture().createView(),
                loadOp: 'clear',
                storeOp: 'store',
                clearValue: { r: 0.5, g: 0.6, b: 0.75, a: 1 },  // sky-ish blue background
            }],
            depthStencilAttachment: {
                view: depthTexture.createView(),
                depthLoadOp: 'clear',
                depthStoreOp: 'store',
                depthClearValue: 1,
            },
        });

        renderPass.setPipeline(renderPipeline);
        renderPass.setBindGroup(0, renderBG);
        for (let i = 0; i < LOD_LEVELS.length; i++) {
            const mesh = lodMeshes[i];
            const range = lodRanges[i];
            if (range.instanceCount === 0) continue;
            renderPass.setVertexBuffer(0, mesh.vb);
            renderPass.setIndexBuffer(mesh.ib, 'uint32');
            renderPass.drawIndexed(mesh.indexCount, range.instanceCount, 0, 0, range.firstInstance);
        }
        renderPass.end();

        device.queue.submit([encoder.finish()]);
        requestAnimationFrame(frame);
    }

    requestAnimationFrame(frame);
}

main();
