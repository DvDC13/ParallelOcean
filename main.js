import { OrbitCamera, mat4Multiply, mat4Inverse } from "./camera.js";
import { generateInitialSpectrum } from './spectrum.js';
import GUI from 'lil-gui';

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
    timeScale: 1.0,
    paused: false,

    // Sun
    sunAzimuth: 31,
    sunElevation: 56,

    // Atmosphere
    fogStart: 800,
    fogEnd: 5000,

    // Clouds
    cloudCoverage: 0.5,
    cloudSpeed: 25,

    // Foam
    foamBias: 1.0,
    foamScale: 1.5,
    foamDecay: 0.5,

    // Day/Night
    autoCycle: false,
    cycleSpeed: 0.02,

    // Post-processing
    bloomStrength: 0.25,
    bloomThreshold: 1.5,
    exposure: 1.0,
    contrast: 1.1,
    saturation: 1.1,
    vignetteStrength: 0.3,
    godRayStrength: 0.8,
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

    // --- Load and compile shaders ---
    const shaderCode = await (await fetch('ocean.wgsl')).text();
    const shaderModule = device.createShaderModule({ code: shaderCode });

    const postShaderCode = await (await fetch('post.wgsl')).text();
    const postShaderModule = device.createShaderModule({ code: postShaderCode });

    const hdrFormat = 'rgba16float';

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

    // --- Foam accumulation ---
    const foamMapBuffer = device.createBuffer({
        size: N * N * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(foamMapBuffer, 0, new Float32Array(N * N));

    const foamParamsBuffer = device.createBuffer({
        size: 32, // FoamParams: 8 floats
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const foamAccumPipeline = device.createComputePipeline({
        layout: 'auto',
        compute: { module: shaderModule, entryPoint: 'foamAccumulate' },
    });

    const foamAccumBG = device.createBindGroup({
        layout: foamAccumPipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: cascades[0].dxtBuf } },
            { binding: 1, resource: { buffer: cascades[0].dztBuf } },
            { binding: 2, resource: { buffer: cascades[1].dxtBuf } },
            { binding: 3, resource: { buffer: cascades[1].dztBuf } },
            { binding: 4, resource: { buffer: cascades[2].dxtBuf } },
            { binding: 5, resource: { buffer: cascades[2].dztBuf } },
            { binding: 6, resource: { buffer: foamMapBuffer } },
            { binding: 7, resource: { buffer: foamParamsBuffer } },
        ],
    });

    // --- Render pipeline (draws the ocean triangles) ---
    // Uniform buffer: 128 bytes (matches RenderUniforms in shader)
    const renderUniformBuf = device.createBuffer({
        size: 128,
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
            targets: [{ format: hdrFormat }],
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
            { binding: 11, resource: { buffer: foamMapBuffer } },
        ],
    });

    // --- Sky pipeline ---
    const invViewProjBuf = device.createBuffer({
        size: 64,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const skyPipeline = device.createRenderPipeline({
        layout: 'auto',
        vertex: { module: shaderModule, entryPoint: 'skyVs' },
        fragment: { module: shaderModule, entryPoint: 'skyFs', targets: [{ format: hdrFormat }] },
        primitive: { topology: 'triangle-list' },
        depthStencil: {
            format: 'depth24plus',
            depthWriteEnabled: false,
            depthCompare: 'always',
        },
    });

    const skyBG = device.createBindGroup({
        layout: skyPipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: renderUniformBuf } },
            { binding: 1, resource: { buffer: invViewProjBuf } },
        ],
    });

    // --- Post-processing pipelines ---
    const postParamsBuffer = device.createBuffer({
        size: 48, // PostParams: 12 floats = 48 bytes
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const blurDirBuffer = device.createBuffer({
        size: 16, // vec4f
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const bloomExtractPipeline = device.createRenderPipeline({
        layout: 'auto',
        vertex: { module: postShaderModule, entryPoint: 'postVs' },
        fragment: { module: postShaderModule, entryPoint: 'bloomExtractFs', targets: [{ format: hdrFormat }] },
        primitive: { topology: 'triangle-list' },
    });

    const blurPipeline = device.createRenderPipeline({
        layout: 'auto',
        vertex: { module: postShaderModule, entryPoint: 'postVs' },
        fragment: { module: postShaderModule, entryPoint: 'blurFs', targets: [{ format: hdrFormat }] },
        primitive: { topology: 'triangle-list' },
    });

    const compositePipeline = device.createRenderPipeline({
        layout: 'auto',
        vertex: { module: postShaderModule, entryPoint: 'postVs' },
        fragment: { module: postShaderModule, entryPoint: 'compositeFs', targets: [{ format }] },
        primitive: { topology: 'triangle-list' },
    });

    const linearSampler = device.createSampler({
        magFilter: 'linear',
        minFilter: 'linear',
    });

    // --- Managed textures (recreated on resize) ---
    let hdrTexture = null, depthTexture = null, bloomTex0 = null, bloomTex1 = null;
    let bloomExtractBG = null, blurHBG = null, blurVBG = null, compositeBG = null;
    let lastW = 0, lastH = 0;

    function recreatePostTextures(w, h) {
        if (w === lastW && h === lastH) return;
        lastW = w; lastH = h;

        // Destroy old textures
        if (hdrTexture) hdrTexture.destroy();
        if (depthTexture) depthTexture.destroy();
        if (bloomTex0) bloomTex0.destroy();
        if (bloomTex1) bloomTex1.destroy();

        // HDR scene texture (full resolution)
        hdrTexture = device.createTexture({
            size: [w, h],
            format: hdrFormat,
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });

        // Depth buffer
        depthTexture = device.createTexture({
            size: [w, h],
            format: 'depth24plus',
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
        });

        // Bloom textures (half resolution for performance)
        const bloomW = Math.max(1, w >> 1);
        const bloomH = Math.max(1, h >> 1);
        bloomTex0 = device.createTexture({
            size: [bloomW, bloomH],
            format: hdrFormat,
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
        bloomTex1 = device.createTexture({
            size: [bloomW, bloomH],
            format: hdrFormat,
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });

        // Recreate bind groups that reference these textures
        bloomExtractBG = device.createBindGroup({
            layout: bloomExtractPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: hdrTexture.createView() },
                { binding: 1, resource: linearSampler },
                { binding: 2, resource: { buffer: postParamsBuffer } },
            ],
        });

        blurHBG = device.createBindGroup({
            layout: blurPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: bloomTex0.createView() },
                { binding: 1, resource: linearSampler },
                { binding: 2, resource: { buffer: blurDirBuffer } },
            ],
        });

        blurVBG = device.createBindGroup({
            layout: blurPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: bloomTex1.createView() },
                { binding: 1, resource: linearSampler },
                { binding: 2, resource: { buffer: blurDirBuffer } },
            ],
        });

        compositeBG = device.createBindGroup({
            layout: compositePipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: hdrTexture.createView() },
                { binding: 1, resource: bloomTex0.createView() },
                { binding: 2, resource: linearSampler },
                { binding: 3, resource: { buffer: postParamsBuffer } },
            ],
        });
    }

    // --- Sun direction (from params) ---
    function sunDir() {
        const el = params.sunElevation * Math.PI / 180;
        const az = params.sunAzimuth * Math.PI / 180;
        return [
            Math.cos(el) * Math.sin(az),
            Math.sin(el),
            Math.cos(el) * Math.cos(az),
        ];
    }

    // --- Compute sun screen position for god rays ---
    function sunScreenPos(viewProj, sun) {
        // Place sun far away in world space
        const sx = sun[0] * 10000;
        const sy = sun[1] * 10000;
        const sz = sun[2] * 10000;
        // Transform through viewProj
        const clipX = viewProj[0]*sx + viewProj[4]*sy + viewProj[8]*sz + viewProj[12];
        const clipY = viewProj[1]*sx + viewProj[5]*sy + viewProj[9]*sz + viewProj[13];
        const clipW = viewProj[3]*sx + viewProj[7]*sy + viewProj[11]*sz + viewProj[15];
        if (clipW <= 0) return null; // Sun behind camera
        const ndcX = clipX / clipW;
        const ndcY = clipY / clipW;
        // NDC to UV (0-1), flip Y
        return [ndcX * 0.5 + 0.5, -ndcY * 0.5 + 0.5];
    }

    // --- GUI ---
    const gui = new GUI({ title: 'Ocean Parameters' });

    const specFolder = gui.addFolder('Spectrum (regenerates)');
    specFolder.add(params, 'windSpeed', 1, 80).onChange(regenerateSpectrum);
    specFolder.add(params, 'windAngle', 0, 360).name('Wind Angle \u00B0').onChange(regenerateSpectrum);

    const ampFolder = gui.addFolder('Cascade Amplitudes');
    ampFolder.add(params, 'amplitude0', 0, 2000).name('Swell (1000m)').onChange(regenerateSpectrum);
    ampFolder.add(params, 'amplitude1', 0, 500).name('Medium (200m)').onChange(regenerateSpectrum);
    ampFolder.add(params, 'amplitude2', 0, 100).name('Ripple (50m)').onChange(regenerateSpectrum);

    const wavesFolder = gui.addFolder('Waves');
    wavesFolder.add(params, 'choppiness', 0, 4);
    wavesFolder.add(params, 'timeScale', 0, 3);

    const foamFolder = gui.addFolder('Foam');
    foamFolder.add(params, 'foamBias', 0, 2, 0.05).name('Bias (threshold)');
    foamFolder.add(params, 'foamScale', 0, 5, 0.1).name('Scale (intensity)');
    foamFolder.add(params, 'foamDecay', 0.1, 3, 0.05).name('Decay (fade speed)');
    wavesFolder.add(params, 'paused');

    const sunFolder = gui.addFolder('Sun');
    sunFolder.add(params, 'sunAzimuth', 0, 360).name('Azimuth \u00B0');
    sunFolder.add(params, 'sunElevation', -20, 90).name('Elevation \u00B0');

    const dayFolder = gui.addFolder('Day/Night');
    dayFolder.add(params, 'autoCycle').name('Auto Cycle');
    dayFolder.add(params, 'cycleSpeed', 0.001, 0.1);

    const atmosFolder = gui.addFolder('Atmosphere');
    atmosFolder.add(params, 'fogStart', 100, 2000);
    atmosFolder.add(params, 'fogEnd', 1000, 10000);

    const cloudFolder = gui.addFolder('Clouds');
    cloudFolder.add(params, 'cloudCoverage', 0, 1);
    cloudFolder.add(params, 'cloudSpeed', 0, 100);

    const postFolder = gui.addFolder('Post-Processing');
    postFolder.add(params, 'bloomStrength', 0, 1).name('Bloom Strength');
    postFolder.add(params, 'bloomThreshold', 0.5, 5).name('Bloom Threshold');
    postFolder.add(params, 'exposure', 0.1, 3).name('Exposure');
    postFolder.add(params, 'contrast', 0.5, 2).name('Contrast');
    postFolder.add(params, 'saturation', 0, 2).name('Saturation');
    postFolder.add(params, 'vignetteStrength', 0, 1).name('Vignette');
    postFolder.add(params, 'godRayStrength', 0, 2).name('God Rays');

    // ===== Frame loop =====
    let accTime = 0;
    let lastTime = performance.now();

    function frame() {
        const now = performance.now();
        const dt = (now - lastTime) / 1000;
        lastTime = now;
        if (!params.paused) {
            accTime += dt * params.timeScale;
        }

        // Day/night cycle
        if (params.autoCycle) {
            params.sunElevation += dt * params.cycleSpeed * 360;
            if (params.sunElevation > 180) params.sunElevation -= 360;
            if (params.sunElevation < -180) params.sunElevation += 360;
            sunFolder.controllers.forEach(c => c.updateDisplay());
        }

        const w = canvas.width;
        const h = canvas.height;
        recreatePostTextures(w, h);

        // Upload time params for each cascade
        for (let i = 0; i < 3; i++) {
            const buf = new ArrayBuffer(16);
            new Float32Array(buf, 0, 1)[0] = accTime;
            new Uint32Array(buf, 4, 1)[0] = N;
            new Float32Array(buf, 8, 1)[0] = CASCADE_PATCHES[i];
            new Float32Array(buf, 12, 1)[0] = params.choppiness;
            device.queue.writeBuffer(cascades[i].timeParamBuf, 0, buf);
        }

        // Upload render uniforms (128 bytes)
        const view = camera.viewMatrix();
        const proj = camera.projMatrix();
        const viewProj = mat4Multiply(proj, view);
        const invVP = mat4Inverse(viewProj);
        const eye = camera.eye;
        const sun = sunDir();

        const uniformData = new ArrayBuffer(128);
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
        f[27] = params.cloudCoverage; // 27: cloudCoverage
        f[28] = params.cloudSpeed;    // 28: cloudSpeed
        f[29] = 0;                  // 29: pad1
        f[30] = 0;                  // 30: pad2
        f[31] = 0;                  // 31: pad3
        device.queue.writeBuffer(renderUniformBuf, 0, uniformData);

        // Upload invViewProj for sky shader
        device.queue.writeBuffer(invViewProjBuf, 0, invVP);

        // Upload post-processing params (48 bytes = 12 floats)
        const sunUV = sunScreenPos(viewProj, sun);
        const postData = new Float32Array(12);
        postData[0] = w;                          // resolution.x
        postData[1] = h;                          // resolution.y
        postData[2] = params.bloomStrength;
        postData[3] = params.bloomThreshold;
        postData[4] = params.exposure;
        postData[5] = params.contrast;
        postData[6] = params.saturation;
        postData[7] = params.vignetteStrength;
        postData[8] = accTime;                    // time
        postData[9] = sunUV ? sunUV[0] : -10;    // sunScreenX (-10 = disabled)
        postData[10] = sunUV ? sunUV[1] : -10;   // sunScreenY
        postData[11] = sunUV ? params.godRayStrength : 0; // godRayStrength (0 if sun behind camera)
        device.queue.writeBuffer(postParamsBuffer, 0, postData);

        // --- GPU commands ---
        const encoder = device.createCommandEncoder();

        // Compute pass: time evolution + FFT for all 3 cascades
        const comp = encoder.beginComputePass();
        for (let i = 0; i < 3; i++) {
            comp.setPipeline(timeEvoPipeline);
            comp.setBindGroup(0, timeEvoBGs[i]);
            comp.dispatchWorkgroups(Math.ceil(N / 16), Math.ceil(N / 16));

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

        // Foam accumulation compute pass (runs after FFT)
        const foamData = new Float32Array(8);
        foamData[0] = dt;                   // dt
        foamData[1] = params.choppiness;    // choppiness
        foamData[2] = CASCADE_PATCHES[0];   // patch0
        foamData[3] = CASCADE_PATCHES[1];   // patch1
        foamData[4] = CASCADE_PATCHES[2];   // patch2
        foamData[5] = params.foamDecay;     // decay rate
        foamData[6] = params.foamBias;      // bias (threshold)
        foamData[7] = params.foamScale;     // scale (intensity)
        device.queue.writeBuffer(foamParamsBuffer, 0, foamData);

        const foamPass = encoder.beginComputePass();
        foamPass.setPipeline(foamAccumPipeline);
        foamPass.setBindGroup(0, foamAccumBG);
        foamPass.dispatchWorkgroups(Math.ceil(N / 16), Math.ceil(N / 16));
        foamPass.end();

        // ---- Pass 1: Render scene → HDR texture ----
        const renderPass = encoder.beginRenderPass({
            colorAttachments: [{
                view: hdrTexture.createView(),
                loadOp: 'clear',
                storeOp: 'store',
                clearValue: { r: 0, g: 0, b: 0, a: 1 },
            }],
            depthStencilAttachment: {
                view: depthTexture.createView(),
                depthLoadOp: 'clear',
                depthStoreOp: 'store',
                depthClearValue: 1,
            },
        });

        // Draw sky first
        renderPass.setPipeline(skyPipeline);
        renderPass.setBindGroup(0, skyBG);
        renderPass.draw(3);

        // Draw ocean on top
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

        // ---- Pass 2: Bloom extract (HDR → bloomTex0 at half res) ----
        const extractPass = encoder.beginRenderPass({
            colorAttachments: [{
                view: bloomTex0.createView(),
                loadOp: 'clear',
                storeOp: 'store',
                clearValue: { r: 0, g: 0, b: 0, a: 1 },
            }],
        });
        extractPass.setPipeline(bloomExtractPipeline);
        extractPass.setBindGroup(0, bloomExtractBG);
        extractPass.draw(3);
        extractPass.end();

        // ---- Pass 3: Blur horizontal (bloomTex0 → bloomTex1) ----
        const bloomW = Math.max(1, w >> 1);
        const bloomH = Math.max(1, h >> 1);
        device.queue.writeBuffer(blurDirBuffer, 0, new Float32Array([1.0 / bloomW, 0, 0, 0]));
        const blurHPass = encoder.beginRenderPass({
            colorAttachments: [{
                view: bloomTex1.createView(),
                loadOp: 'clear',
                storeOp: 'store',
                clearValue: { r: 0, g: 0, b: 0, a: 1 },
            }],
        });
        blurHPass.setPipeline(blurPipeline);
        blurHPass.setBindGroup(0, blurHBG);
        blurHPass.draw(3);
        blurHPass.end();

        // ---- Pass 4: Blur vertical (bloomTex1 → bloomTex0) ----
        device.queue.writeBuffer(blurDirBuffer, 0, new Float32Array([0, 1.0 / bloomH, 0, 0]));
        const blurVPass = encoder.beginRenderPass({
            colorAttachments: [{
                view: bloomTex0.createView(),
                loadOp: 'clear',
                storeOp: 'store',
                clearValue: { r: 0, g: 0, b: 0, a: 1 },
            }],
        });
        blurVPass.setPipeline(blurPipeline);
        blurVPass.setBindGroup(0, blurVBG);
        blurVPass.draw(3);
        blurVPass.end();

        // ---- Pass 5: Composite (HDR + bloom → canvas) ----
        const compositePass = encoder.beginRenderPass({
            colorAttachments: [{
                view: context.getCurrentTexture().createView(),
                loadOp: 'clear',
                storeOp: 'store',
                clearValue: { r: 0, g: 0, b: 0, a: 1 },
            }],
        });
        compositePass.setPipeline(compositePipeline);
        compositePass.setBindGroup(0, compositeBG);
        compositePass.draw(3);
        compositePass.end();

        device.queue.submit([encoder.finish()]);
        requestAnimationFrame(frame);
    }

    requestAnimationFrame(frame);
}

main();
