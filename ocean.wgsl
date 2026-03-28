// ===== Minimal Tessendorf Ocean — Compute + Render =====
const PI: f32 = 3.14159265359;
const N_SIZE: u32 = 256u;
const LOG2N: u32 = 8u;

// ==========================================
// Step 1: Time Evolution compute shader
// Takes the initial wave recipe (h0) and advances it in time
// ==========================================

struct TimeParams {
    time: f32,      // current time in seconds
    N: u32,         // grid size (256)
    patchSize: f32, // how big this patch of ocean is in meters
    lambda: f32,    // choppiness multiplier
};

@group(0) @binding(0) var<storage, read> h0: array<vec4f>;        // initial spectrum
@group(0) @binding(1) var<storage, read> omega: array<f32>;        // wave frequencies
@group(0) @binding(2) var<storage, read_write> ht: array<vec2f>;   // output: wave heights
@group(0) @binding(3) var<storage, read_write> dxt: array<vec2f>;  // output: X displacement
@group(0) @binding(4) var<storage, read_write> dzt: array<vec2f>;  // output: Z displacement
@group(0) @binding(5) var<uniform> tParams: TimeParams;

@compute @workgroup_size(16, 16)
fn timeEvolution(@builtin(global_invocation_id) id: vec3u) {
    let n = id.x;
    let m = id.y;
    let N = tParams.N;
    if (n >= N || m >= N) { return; }

    let idx = m * N + n;
    let w = omega[idx] * tParams.time;
    let cos_w = cos(w);
    let sin_w = sin(w);

    // h0k = initial wave, h0c = conjugate
    let h0k = vec2f(h0[idx].x, h0[idx].y);
    let h0c = vec2f(h0[idx].z, h0[idx].w);

    // Euler formula: rotate the wave by time
    let ht_val = vec2f(
        h0k.x * cos_w - h0k.y * sin_w + h0c.x * cos_w + h0c.y * sin_w,
        h0k.x * sin_w + h0k.y * cos_w - h0c.x * sin_w + h0c.y * cos_w,
    );
    ht[idx] = ht_val;

    // Wave direction for choppy displacement
    var kn = f32(n);
    if (n >= N / 2u) { kn -= f32(N); }
    var km = f32(m);
    if (m >= N / 2u) { km -= f32(N); }

    let kx = 2.0 * PI * kn / tParams.patchSize;
    let kz = 2.0 * PI * km / tParams.patchSize;
    let kLen = sqrt(kx * kx + kz * kz);

    var kNormX: f32 = 0.0;
    var kNormZ: f32 = 0.0;
    if (kLen > 1e-6) {
        kNormX = kx / kLen;
        kNormZ = kz / kLen;
    }

    // Horizontal displacement (makes waves lean and form sharp crests)
    dxt[idx] = vec2f(kNormX * ht_val.y, -kNormX * ht_val.x);
    dzt[idx] = vec2f(kNormZ * ht_val.y, -kNormZ * ht_val.x);
}

// ==========================================
// Step 2: GPU FFT (Fast Fourier Transform)
// Converts frequency data → spatial wave heights
// Uses butterfly algorithm with shared memory
// ==========================================

var<workgroup> s_re: array<f32, 256>;
var<workgroup> s_im: array<f32, 256>;

// Reverse the bits of an 8-bit number (needed for FFT reordering)
fn bitrev8(x: u32) -> u32 {
    var v = x;
    v = ((v >> 1u) & 0x55u) | ((v & 0x55u) << 1u);
    v = ((v >> 2u) & 0x33u) | ((v & 0x33u) << 2u);
    v = ((v >> 4u) & 0x0Fu) | ((v & 0x0Fu) << 4u);
    return v;
}

@group(0) @binding(0) var<storage, read> fftSrc: array<vec2f>;
@group(0) @binding(1) var<storage, read_write> fftDst: array<vec2f>;

// Process one row of the 2D grid
@compute @workgroup_size(256)
fn fftRowPass(@builtin(local_invocation_id) lid: vec3u,
              @builtin(workgroup_id) wid: vec3u) {
    let row = wid.y;
    let j = lid.x;

    // Load data in bit-reversed order
    let rev = bitrev8(j);
    let val = fftSrc[row * N_SIZE + rev];
    s_re[j] = val.x;
    s_im[j] = val.y;
    workgroupBarrier();

    // Butterfly passes (8 stages for 256 elements)
    for (var s = 0u; s < LOG2N; s++) {
        let m = 1u << s;
        let m2 = m << 1u;
        if ((j & m) == 0u) {
            let k = j & (m - 1u);
            let pairIdx = j + m;
            let angle = 2.0 * PI * f32(k) / f32(m2);
            let tw_re = cos(angle);
            let tw_im = sin(angle);
            let a_re = s_re[j]; let a_im = s_im[j];
            let b_re = s_re[pairIdx]; let b_im = s_im[pairIdx];
            let btw_re = b_re * tw_re - b_im * tw_im;
            let btw_im = b_re * tw_im + b_im * tw_re;
            s_re[j] = a_re + btw_re; s_im[j] = a_im + btw_im;
            s_re[pairIdx] = a_re - btw_re; s_im[pairIdx] = a_im - btw_im;
        }
        workgroupBarrier();
    }

    let invN = 1.0 / f32(N_SIZE);
    fftDst[row * N_SIZE + j] = vec2f(s_re[j] * invN, s_im[j] * invN);
}

// Process one column of the 2D grid
@compute @workgroup_size(256)
fn fftColPass(@builtin(local_invocation_id) lid: vec3u,
              @builtin(workgroup_id) wid: vec3u) {
    let col = wid.y;
    let j = lid.x;
    let rev = bitrev8(j);
    let val = fftSrc[rev * N_SIZE + col];
    s_re[j] = val.x;
    s_im[j] = val.y;
    workgroupBarrier();

    for (var s = 0u; s < LOG2N; s++) {
        let m = 1u << s;
        let m2 = m << 1u;
        if ((j & m) == 0u) {
            let k = j & (m - 1u);
            let pairIdx = j + m;
            let angle = 2.0 * PI * f32(k) / f32(m2);
            let tw_re = cos(angle);
            let tw_im = sin(angle);
            let a_re = s_re[j]; let a_im = s_im[j];
            let b_re = s_re[pairIdx]; let b_im = s_im[pairIdx];
            let btw_re = b_re * tw_re - b_im * tw_im;
            let btw_im = b_re * tw_im + b_im * tw_re;
            s_re[j] = a_re + btw_re; s_im[j] = a_im + btw_im;
            s_re[pairIdx] = a_re - btw_re; s_im[pairIdx] = a_im - btw_im;
        }
        workgroupBarrier();
    }

    let invN = 1.0 / f32(N_SIZE);
    fftDst[j * N_SIZE + col] = vec2f(s_re[j] * invN, s_im[j] * invN);
}

// ==========================================
// Step 2b: Foam accumulation
// Computes Jacobian (wave folding) and accumulates foam over time
// ==========================================

struct FoamParams {
    dt: f32,
    choppiness: f32,
    patch0: f32,
    patch1: f32,
    patch2: f32,
    decay: f32,
    bias: f32,      // Jacobian threshold (higher = more foam)
    scale: f32,     // foam intensity multiplier
};

@group(0) @binding(0) var<storage, read> foamDxt0: array<vec2f>;
@group(0) @binding(1) var<storage, read> foamDzt0: array<vec2f>;
@group(0) @binding(2) var<storage, read> foamDxt1: array<vec2f>;
@group(0) @binding(3) var<storage, read> foamDzt1: array<vec2f>;
@group(0) @binding(4) var<storage, read> foamDxt2: array<vec2f>;
@group(0) @binding(5) var<storage, read> foamDzt2: array<vec2f>;
@group(0) @binding(6) var<storage, read_write> foamMap: array<f32>;
@group(0) @binding(7) var<uniform> fParams: FoamParams;

@compute @workgroup_size(16, 16)
fn foamAccumulate(@builtin(global_invocation_id) id: vec3u) {
    let col = id.x;
    let row = id.y;
    if (col >= 256u || row >= 256u) { return; }

    let idx = row * 256u + col;
    let chop0 = fParams.choppiness;
    let chop1 = fParams.choppiness * 0.5;
    let chop2 = fParams.choppiness * 0.15;

    // Neighbor indices (wrapping)
    let l = row * 256u + ((col + 255u) % 256u);
    let r = row * 256u + ((col + 1u) % 256u);
    let u = ((row + 255u) % 256u) * 256u + col;
    let d = ((row + 1u) % 256u) * 256u + col;

    // Cascade 0 Jacobian — measures surface compression
    // When Jacobian < 0, the wave has folded over itself → foam
    let inv0 = 256.0 / (2.0 * fParams.patch0);
    let ddxdx0 = (foamDxt0[r].x * -chop0 - foamDxt0[l].x * -chop0) * inv0;
    let ddzdz0 = (foamDzt0[d].x * -chop0 - foamDzt0[u].x * -chop0) * inv0;
    let jac0 = (1.0 + ddxdx0) * (1.0 + ddzdz0);

    // Cascade 1 Jacobian
    let inv1 = 256.0 / (2.0 * fParams.patch1);
    let ddxdx1 = (foamDxt1[r].x * -chop1 - foamDxt1[l].x * -chop1) * inv1;
    let ddzdz1 = (foamDzt1[d].x * -chop1 - foamDzt1[u].x * -chop1) * inv1;
    let jac1 = (1.0 + ddxdx1) * (1.0 + ddzdz1);

    // Cascade 2 Jacobian
    let inv2 = 256.0 / (2.0 * fParams.patch2);
    let ddxdx2 = (foamDxt2[r].x * -chop2 - foamDxt2[l].x * -chop2) * inv2;
    let ddzdz2 = (foamDzt2[d].x * -chop2 - foamDzt2[u].x * -chop2) * inv2;
    let jac2 = (1.0 + ddxdx2) * (1.0 + ddzdz2);

    // Generate foam where Jacobian goes negative (wave folding)
    // bias controls threshold, scale controls intensity
    let foam0 = clamp((-jac0 + fParams.bias) * fParams.scale, 0.0, 1.0);
    let foam1 = clamp((-jac1 + fParams.bias * 0.9) * fParams.scale * 1.3, 0.0, 1.0) * 0.6;
    let foam2 = clamp((-jac2 + fParams.bias * 0.8) * fParams.scale * 1.6, 0.0, 1.0) * 0.3;
    let generated = min(foam0 + foam1 + foam2, 1.0);

    // Decay existing foam, keep maximum of decayed and new
    let existing = foamMap[idx];
    let decayed = existing * exp(-fParams.decay * fParams.dt);

    foamMap[idx] = max(decayed, generated);
}

// ==========================================
// Step 3: Render the ocean surface
// ==========================================

struct RenderUniforms {
    viewProj: mat4x4f,         // 0-63: camera transform
    eyePos: vec3f,             // 64-75: where the camera is
    time: f32,                 // 76-79: current time
    sunDir: vec3f,             // 80-91: sun direction (for basic lighting)
    choppiness: f32,           // 92-95
    cascadePatch0: f32,        // 96-99
    cascadePatch1: f32,        // 100-103
    cascadePatch2: f32,        // 104-107
    cloudCoverage: f32,        // 108-111
    cloudSpeed: f32,           // 112-115
    pad1: f32,                 // 116-119
    pad2: f32,                 // 120-123
    pad3: f32,                 // 124-127 (total 128 bytes)
};

// Cascade 0 — large swells
@group(0) @binding(0) var<uniform> render: RenderUniforms;
@group(0) @binding(1) var<storage, read> ht0: array<vec2f>;
@group(0) @binding(2) var<storage, read> dxt0: array<vec2f>;
@group(0) @binding(3) var<storage, read> dzt0: array<vec2f>;
@group(0) @binding(4) var<storage, read> tileOffsets: array<vec2f>;
// Cascade 1 — medium waves
@group(0) @binding(5) var<storage, read> ht1: array<vec2f>;
@group(0) @binding(6) var<storage, read> dxt1: array<vec2f>;
@group(0) @binding(7) var<storage, read> dzt1: array<vec2f>;
// Cascade 2 — small ripples
@group(0) @binding(8) var<storage, read> ht2: array<vec2f>;
@group(0) @binding(9) var<storage, read> dxt2: array<vec2f>;
@group(0) @binding(10) var<storage, read> dzt2: array<vec2f>;
// Accumulated foam map from compute pass
@group(0) @binding(11) var<storage, read> renderFoamMap: array<f32>;

// Look up FFT index from world position
fn cascadeIdx(wx: f32, wz: f32, patchSize: f32) -> u32 {
    let col = u32(fract(wx / patchSize) * 256.0) % 256u;
    let row = u32(fract(wz / patchSize) * 256.0) % 256u;
    return row * 256u + col;
}

// Look up index + 4 neighbors (for normal calculation)
struct CIdx { c: u32, l: u32, r: u32, u: u32, d: u32, };

fn cascadeIdx5(wx: f32, wz: f32, patchSize: f32) -> CIdx {
    let col = u32(fract(wx / patchSize) * 256.0) % 256u;
    let row = u32(fract(wz / patchSize) * 256.0) % 256u;
    var ci: CIdx;
    ci.c = row * 256u + col;
    ci.l = row * 256u + ((col + 255u) % 256u);
    ci.r = row * 256u + ((col + 1u) % 256u);
    ci.u = ((row + 255u) % 256u) * 256u + col;
    ci.d = ((row + 1u) % 256u) * 256u + col;
    return ci;
}

struct VsOut {
    @builtin(position) pos: vec4f,
    @location(0) worldPos: vec3f,
    @location(1) normal: vec3f,
    @location(2) distToEye: f32,
    @location(3) foam: f32,
    @location(4) waveHeight: f32,
};

@vertex
fn vs(@location(0) position: vec3f,
      @builtin(instance_index) instIdx: u32) -> VsOut {
    // Get tile offset for this instance
    let tileX = tileOffsets[instIdx].x;
    let tileZ = tileOffsets[instIdx].y;
    let worldX = position.x + tileX;
    let worldZ = position.z + tileZ;

    // Choppiness per cascade (smaller cascades get less chop)
    let lambda = render.choppiness;
    let chop0 = lambda;
    let chop1 = lambda * 0.5;
    let chop2 = lambda * 0.15;

    // Look up FFT data for all 3 cascades at this world position
    let ci0 = cascadeIdx5(worldX, worldZ, render.cascadePatch0);
    let ci1 = cascadeIdx5(worldX, worldZ, render.cascadePatch1);
    let ci2 = cascadeIdx5(worldX, worldZ, render.cascadePatch2);

    // Wave heights
    let h = ht0[ci0.c].x + ht1[ci1.c].x + ht2[ci2.c].x;

    // Horizontal displacements (choppiness)
    let dx = dxt0[ci0.c].x * -chop0 + dxt1[ci1.c].x * -chop1 + dxt2[ci2.c].x * -chop2;
    let dz = dzt0[ci0.c].x * -chop0 + dzt1[ci1.c].x * -chop1 + dzt2[ci2.c].x * -chop2;

    let displaced = vec3f(worldX + dx, h, worldZ + dz);

    // --- Normal calculation (with displacement gradients like OceanGpu) ---
    var totalDhDx: f32 = 0.0;
    var totalDhDz: f32 = 0.0;
    var totalDdxDx: f32 = 0.0;
    var totalDdzDz: f32 = 0.0;

    // Cascade 0 slopes + displacement gradients
    let s0 = render.cascadePatch0 / 256.0;
    let inv0 = 1.0 / (2.0 * s0);
    totalDhDx += (ht0[ci0.r].x - ht0[ci0.l].x) * inv0;
    totalDhDz += (ht0[ci0.d].x - ht0[ci0.u].x) * inv0;
    totalDdxDx += (dxt0[ci0.r].x * -chop0 - dxt0[ci0.l].x * -chop0) * inv0;
    totalDdzDz += (dzt0[ci0.d].x * -chop0 - dzt0[ci0.u].x * -chop0) * inv0;

    // Cascade 1
    let s1 = render.cascadePatch1 / 256.0;
    let inv1 = 1.0 / (2.0 * s1);
    totalDhDx += (ht1[ci1.r].x - ht1[ci1.l].x) * inv1;
    totalDhDz += (ht1[ci1.d].x - ht1[ci1.u].x) * inv1;
    totalDdxDx += (dxt1[ci1.r].x * -chop1 - dxt1[ci1.l].x * -chop1) * inv1;
    totalDdzDz += (dzt1[ci1.d].x * -chop1 - dzt1[ci1.u].x * -chop1) * inv1;

    // Cascade 2
    let s2 = render.cascadePatch2 / 256.0;
    let inv2 = 1.0 / (2.0 * s2);
    totalDhDx += (ht2[ci2.r].x - ht2[ci2.l].x) * inv2;
    totalDhDz += (ht2[ci2.d].x - ht2[ci2.u].x) * inv2;
    totalDdxDx += (dxt2[ci2.r].x * -chop2 - dxt2[ci2.l].x * -chop2) * inv2;
    totalDdzDz += (dzt2[ci2.d].x * -chop2 - dzt2[ci2.u].x * -chop2) * inv2;

    // Normal from tangent cross product (accounts for choppiness)
    let tangentX = vec3f(1.0 + totalDdxDx, totalDhDx, 0.0);
    let tangentZ = vec3f(0.0, totalDhDz, 1.0 + totalDdzDz);
    let normal = normalize(cross(tangentZ, tangentX));

    // Sample accumulated foam from compute pass
    let foamAmount = renderFoamMap[ci0.c];

    var out: VsOut;
    out.worldPos = displaced;
    out.normal = normal;
    out.distToEye = length(displaced.xz - render.eyePos.xz);
    out.foam = foamAmount;
    out.waveHeight = h;
    out.pos = render.viewProj * vec4f(displaced, 1.0);
    return out;
}

// ==========================================
// Underwater caustics
// ==========================================

fn causticsLayer(p: vec2f) -> f32 {
    // Create bright network pattern using tiled cells
    let cell = floor(p);
    let f = fract(p);

    // Voronoi-like distance field for caustic network
    var minDist = 1.0;
    for (var j = -1; j <= 1; j++) {
        for (var i = -1; i <= 1; i++) {
            let neighbor = vec2f(f32(i), f32(j));
            let offset = cell + neighbor;
            // Random point within each cell
            let h = fract(sin(vec2f(
                dot(offset, vec2f(127.1, 311.7)),
                dot(offset, vec2f(269.5, 183.3))
            )) * 43758.5453);
            let diff = neighbor + h - f;
            let d = length(diff);
            minDist = min(minDist, d);
        }
    }
    // Sharp bright lines where cells meet
    return pow(1.0 - smoothstep(0.0, 0.35, minDist), 3.0);
}

fn caustics(worldPos: vec3f, time: f32, sunDir: vec3f) -> f32 {
    // Project caustics from sun direction onto ocean floor
    // Simulate light refracting through waves
    let scale1 = 0.08;
    let scale2 = 0.12;
    let speed = time * 0.8;

    let p1 = worldPos.xz * scale1 + vec2f(speed * 0.3, speed * 0.2);
    let p2 = worldPos.xz * scale2 + vec2f(-speed * 0.2, speed * 0.35);

    // Two overlapping layers create more complex pattern
    let c1 = causticsLayer(p1);
    let c2 = causticsLayer(p2 * 1.3 + 3.7);

    // Combine: multiply for intersection pattern (realistic)
    let combined = (c1 + c2) * 0.5;

    // Stronger when sun is higher (sunDir.y), dimmer at low sun
    let sunFactor = clamp(sunDir.y * 1.5, 0.0, 1.0);

    return combined * sunFactor;
}

@fragment
fn fs(inp: VsOut) -> @location(0) vec4f {
    let N = normalize(inp.normal);
    let V = normalize(render.eyePos - inp.worldPos);
    let L = normalize(render.sunDir);
    let NdotV = max(dot(N, V), 0.001);
    let NdotL = max(dot(N, L), 0.0);

    // Sun light intensity
    let sunIntensity = vec3f(3.5, 3.2, 2.8);
    let ambientLight = vec3f(0.5, 0.6, 0.85);

    // ---------- Fresnel (Schlick) ----------
    let F0: f32 = 0.02;
    let fresnel = F0 + (1.0 - F0) * pow(1.0 - NdotV, 5.0);

    // ---------- Water absorption color ----------
    let absorpColor = vec3f(0.02, 0.08, 0.18);
    let scatterColor = vec3f(0.06, 0.28, 0.42);
    let depthFactor = clamp(NdotV * 0.8 + 0.1, 0.0, 1.0);
    var waterColor = mix(absorpColor, scatterColor, depthFactor);
    waterColor *= (ambientLight + sunIntensity * NdotL * 0.5);

    // ---------- Underwater caustics ----------
    // Visible when looking steeply into water (high NdotV)
    // and close to camera (fades with distance)
    let causticsVal = caustics(inp.worldPos, render.time, L);
    let causticsDepth = clamp(NdotV - 0.3, 0.0, 1.0);
    let causticsDist = smoothstep(500.0, 50.0, inp.distToEye);
    let causticsColor = vec3f(0.15, 0.35, 0.45) * causticsVal * causticsDepth * causticsDist;
    waterColor += causticsColor;

    // ---------- Subsurface scattering ----------
    // Light passing through thin wave crests creates a green/teal glow
    let sssDir = normalize(L + N * 0.4);
    let sssDot = pow(clamp(dot(V, -sssDir), 0.0, 1.0), 5.0);
    let sssHeight = clamp(inp.waveHeight * 0.1, 0.0, 1.0);
    let sssIntensity = sssDot * sssHeight * 0.3;
    let sssColor = vec3f(0.02, 0.18, 0.12) * sssIntensity * sunIntensity;

    // Forward scattering — glow when looking toward the sun
    let viewSunDot = max(dot(-V, L), 0.0);
    let forwardSSS = pow(viewSunDot, 6.0) * 0.03 * vec3f(0.02, 0.12, 0.08);

    // ---------- Sky / environment reflection ----------
    let R = reflect(-V, N);
    let envColor = skyColor(R, render.sunDir, render.time);

    // ---------- Sun specular (dual-lobe GGX) ----------
    let H = normalize(L + V);
    let NdotH = max(dot(N, H), 0.0);

    // Sharp lobe — tight sun disk reflection
    let roughSharp = 0.04;
    let a2s = roughSharp * roughSharp;
    let denomS = NdotH * NdotH * (a2s - 1.0) + 1.0;
    let Ds = a2s / (PI * denomS * denomS);
    let specSharp = Ds * fresnel * NdotL * 3.0;

    // Broad lobe — wide glitter around sun
    let roughBroad = 0.25;
    let a2b = roughBroad * roughBroad;
    let denomB = NdotH * NdotH * (a2b - 1.0) + 1.0;
    let Db = a2b / (PI * denomB * denomB);
    let specBroad = Db * fresnel * NdotL * 0.8;

    let sunColor = vec3f(1.0, 0.95, 0.85);
    let specTotal = sunColor * sunIntensity * (specSharp + specBroad);

    // ---------- Foam ----------
    let foam = inp.foam;
    let foamColor = vec3f(0.9, 0.92, 0.95) * (ambientLight + sunIntensity * max(NdotL, 0.3));
    let foamMask = smoothstep(0.0, 0.4, foam);

    // ---------- Composite ----------
    var color = mix(waterColor, envColor, fresnel) + sssColor + forwardSSS + specTotal;
    color = mix(color, foamColor, foamMask * 0.85);

    // ---------- Distance fog ----------
    let fogFactor = clamp((inp.distToEye - 800.0) / 4200.0, 0.0, 1.0);
    let viewDir = normalize(inp.worldPos - render.eyePos);
    let fogColor = skyColor(viewDir, render.sunDir, render.time);
    color = mix(color, fogColor, fogFactor * fogFactor);

    return vec4f(color, 1.0);
}

// ==========================================
// Step 4: Sky rendering
// ==========================================

// --- Noise helpers for clouds ---
fn hash31(p: vec3f) -> f32 {
    var q = fract(p * vec3f(443.897, 441.423, 437.195));
    q += dot(q, q.yzx + 19.19);
    return fract((q.x + q.y) * q.z);
}

fn valueNoise3D(p: vec3f) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    return mix(
        mix(mix(hash31(i + vec3f(0,0,0)), hash31(i + vec3f(1,0,0)), u.x),
            mix(hash31(i + vec3f(0,1,0)), hash31(i + vec3f(1,1,0)), u.x), u.y),
        mix(mix(hash31(i + vec3f(0,0,1)), hash31(i + vec3f(1,0,1)), u.x),
            mix(hash31(i + vec3f(0,1,1)), hash31(i + vec3f(1,1,1)), u.x), u.y),
        u.z
    );
}

fn fbm5(p: vec3f) -> f32 {
    var val = 0.0;
    var amp = 0.5;
    var pos = p;
    for (var i = 0; i < 5; i++) {
        val += amp * valueNoise3D(pos);
        pos = pos * 2.07 + vec3f(1.17, 0.83, 0.61);
        amp *= 0.48;
    }
    return val;
}

fn cloudDensity(worldPos: vec3f, time: f32, coverage: f32, speed: f32) -> f32 {
    let cloudBase = 2000.0;
    let cloudTop = 4500.0;
    let hFrac = (worldPos.y - cloudBase) / (cloudTop - cloudBase);
    if (hFrac < 0.0 || hFrac > 1.0) { return 0.0; }
    let heightMask = smoothstep(0.0, 0.25, hFrac) * smoothstep(1.0, 0.65, hFrac);
    let windDir = vec3f(speed, 0.0, speed * 0.6) * time;
    let p = worldPos * 0.00035 + windDir * 0.00035;
    let noise = fbm5(p);
    let threshold = 1.0 - coverage;
    let density = smoothstep(threshold, threshold + 0.25, noise) * heightMask;
    return density;
}

fn screenHash(uv: vec2f) -> f32 {
    return fract(sin(dot(uv, vec2f(12.9898, 78.233))) * 43758.5453);
}

fn raymarchClouds(rayOrigin: vec3f, rayDir: vec3f, sunDir: vec3f, time: f32, coverage: f32, speed: f32) -> vec4f {
    let cloudBase = 2000.0;
    let cloudTop = 4500.0;
    if (rayDir.y <= 0.005) { return vec4f(0.0); }
    let tBase = (cloudBase - rayOrigin.y) / rayDir.y;
    let tTop = (cloudTop - rayOrigin.y) / rayDir.y;
    if (tBase < 0.0) { return vec4f(0.0); }
    let tStart = max(tBase, 0.0);
    let tEnd = tTop;
    let maxDist = 40000.0;
    let tClamped = min(tEnd, tStart + maxDist);
    let numSteps = 32u;
    let stepSize = (tClamped - tStart) / f32(numSteps);
    let jitter = screenHash(rayDir.xz * 1000.0 + vec2f(time * 7.0)) * stepSize;
    var transmittance = 1.0;
    var lightColor = vec3f(0.0);
    let L = normalize(sunDir);
    let sunElev = L.y;
    let cloudDayFactor = clamp(sunElev * 5.0, 0.0, 1.0);
    let daySunCol = vec3f(1.0, 0.95, 0.85);
    let sunsetSunCol = vec3f(1.0, 0.6, 0.3);
    let sunCol = mix(sunsetSunCol, daySunCol, clamp(sunElev * 3.0, 0.0, 1.0)) * (1.0 + cloudDayFactor);
    let ambientCol = mix(vec3f(0.08, 0.1, 0.2), vec3f(0.5, 0.6, 0.8), cloudDayFactor);
    let cosAngle = dot(rayDir, L);
    let forwardScatter = pow(max(cosAngle, 0.0), 4.0) * 0.5;
    let backScatter = 0.2;
    let phase = forwardScatter + backScatter;
    for (var i = 0u; i < numSteps; i++) {
        let t = tStart + jitter + f32(i) * stepSize;
        if (t > tClamped) { break; }
        let pos = rayOrigin + rayDir * t;
        let density = cloudDensity(pos, time, coverage, speed);
        if (density > 0.005) {
            let ls1 = cloudDensity(pos + L * 120.0, time, coverage, speed);
            let ls2 = cloudDensity(pos + L * 300.0, time, coverage, speed);
            let lightOD = ls1 * 0.6 + ls2 * 0.4;
            let lightTransmit = exp(-lightOD * 4.5);
            let powder = 1.0 - exp(-density * 6.0);
            let lit = sunCol * lightTransmit * phase * powder + ambientCol * 0.4;
            let edge = exp(-density * 3.0) * pow(max(cosAngle, 0.0), 2.0) * 0.3;
            let cloudCol = lit + sunCol * edge;
            let absorption = density * stepSize * 0.004;
            lightColor += transmittance * cloudCol * absorption;
            transmittance *= exp(-absorption);
        }
        if (transmittance < 0.02) { break; }
    }
    let horizonFade = smoothstep(0.005, 0.1, rayDir.y);
    let alpha = (1.0 - transmittance) * horizonFade;
    return vec4f(lightColor * horizonFade, alpha);
}

// --- Stars ---
fn starHash(p: vec2f) -> f32 {
    return fract(sin(dot(p, vec2f(127.1, 311.7))) * 43758.5453);
}

fn stars(rayDir: vec3f, time: f32) -> f32 {
    let p = rayDir.xz / (rayDir.y + 0.001) * 80.0;
    let cell = floor(p);
    let f = fract(p);
    let rand = starHash(cell);
    if (rand > 0.04) { return 0.0; }
    let starPos = vec2f(fract(rand * 17.3) * 0.6 + 0.2, fract(rand * 31.7) * 0.6 + 0.2);
    let d = length(f - starPos);
    let size = 0.01 + fract(rand * 7.1) * 0.025;
    let twinkle = 0.7 + 0.3 * sin(rand * 100.0 + time * (2.0 + rand * 4.0));
    return smoothstep(size, 0.0, d) * twinkle;
}

// --- Analytical sky color ---
fn skyColor(rayDir: vec3f, sunDir: vec3f, time: f32) -> vec3f {
    let L = normalize(sunDir);
    let elevation = max(rayDir.y, 0.0);
    let sunElev = L.y;
    let dayFactor = clamp(sunElev * 5.0, 0.0, 1.0);
    let sunsetFactor = smoothstep(0.3, 0.05, sunElev) * smoothstep(-0.1, 0.02, sunElev);

    // Day sky
    let skyZenith  = vec3f(0.1, 0.3, 0.75);
    let skyMid     = vec3f(0.35, 0.6, 0.95);
    let skyHorizon = vec3f(0.65, 0.82, 0.98);
    var daySky: vec3f;
    if (elevation < 0.15) {
        daySky = mix(skyHorizon, skyMid, elevation / 0.15);
    } else {
        daySky = mix(skyMid, skyZenith, pow((elevation - 0.15) / 0.85, 0.6));
    }

    // Sunset sky
    let sunsetHorizon = vec3f(0.9, 0.4, 0.15);
    let sunsetMid = vec3f(0.7, 0.25, 0.3);
    let sunsetZenith = vec3f(0.15, 0.1, 0.35);
    var sunsetSky: vec3f;
    if (elevation < 0.15) {
        sunsetSky = mix(sunsetHorizon, sunsetMid, elevation / 0.15);
    } else {
        sunsetSky = mix(sunsetMid, sunsetZenith, (elevation - 0.15) / 0.85);
    }
    let sunHorizDot = max(dot(normalize(vec3f(rayDir.x, 0.0, rayDir.z)), normalize(vec3f(L.x, 0.0, L.z))), 0.0);
    let sunsetDirWeight = pow(sunHorizDot, 1.5) * 0.6 + 0.4;

    // Night sky
    let nightZenith = vec3f(0.005, 0.008, 0.025);
    let nightHorizon = vec3f(0.02, 0.025, 0.05);
    let nightSky = mix(nightHorizon, nightZenith, pow(elevation, 0.5));

    // Blend
    var sky = mix(nightSky, daySky, dayFactor);
    sky = mix(sky, sunsetSky * sunsetDirWeight, sunsetFactor);

    // Sun disk and glow
    let sunDot = max(dot(rayDir, L), 0.0);
    let sunCol = mix(vec3f(1.0, 0.4, 0.1), vec3f(1.0, 0.95, 0.85), clamp(sunElev * 4.0, 0.0, 1.0));
    let sunBrightness = max(sunElev * 2.0, 0.0);
    let sunDisk = pow(sunDot, 1500.0) * 25.0 * sunBrightness;
    let sunGlow = pow(sunDot, 80.0) * 0.6 * sunBrightness;
    let sunHalo = pow(sunDot, 16.0) * 0.12 * sunBrightness;
    sky += sunCol * (sunDisk + sunGlow + sunHalo);

    // Atmospheric scattering
    let scatter = pow(sunDot, 3.0) * 0.35;
    let scatterCol = mix(vec3f(0.6, 0.2, 0.05), vec3f(0.4, 0.25, 0.05), dayFactor);
    sky += scatterCol * scatter * (1.0 - elevation) * max(sunElev + 0.1, 0.0);
    let mie = pow(sunDot, 5.0) * 0.15;
    sky += vec3f(0.35, 0.3, 0.2) * mie * sunBrightness;

    // Stars
    if (dayFactor < 0.8 && rayDir.y > 0.01) {
        let starBrightness = stars(rayDir, time) * (1.0 - dayFactor);
        sky += vec3f(0.8, 0.85, 1.0) * starBrightness;
    }

    // Moon
    let moonDir = normalize(vec3f(-L.x, max(abs(L.y), 0.15), -L.z));
    let moonDot = max(dot(rayDir, moonDir), 0.0);
    let moonDisk = pow(moonDot, 3000.0) * 8.0;
    let moonGlow = pow(moonDot, 200.0) * 0.3;
    let moonVis = (1.0 - dayFactor) * smoothstep(-0.05, 0.1, moonDir.y);
    sky += vec3f(0.7, 0.75, 0.85) * (moonDisk + moonGlow) * moonVis;

    // Below horizon
    if (rayDir.y < 0.0) {
        let t = min(-rayDir.y * 4.0, 1.0);
        let belowCol = mix(nightHorizon, vec3f(0.35, 0.45, 0.55), dayFactor);
        sky = mix(sky, belowCol, t);
    }

    return sky;
}

// --- Sky vertex/fragment shaders ---
struct SkyVsOut {
    @builtin(position) pos: vec4f,
    @location(0) rayDir: vec3f,
};

@group(0) @binding(0) var<uniform> skyUniforms: RenderUniforms;
@group(0) @binding(1) var<uniform> invViewProj: mat4x4f;

@vertex
fn skyVs(@builtin(vertex_index) vid: u32) -> SkyVsOut {
    let uv = vec2f(f32((vid << 1u) & 2u), f32(vid & 2u));
    let clipPos = vec4f(uv * 2.0 - 1.0, 1.0, 1.0);
    var out: SkyVsOut;
    out.pos = clipPos;
    out.rayDir = vec3f(uv * 2.0 - 1.0, 0.0);
    return out;
}

@fragment
fn skyFs(inp: SkyVsOut) -> @location(0) vec4f {
    let clipPos = vec4f(inp.rayDir.xy, 1.0, 1.0);
    let worldPos4 = invViewProj * clipPos;
    let worldPos = worldPos4.xyz / worldPos4.w;
    let rayDir = normalize(worldPos - skyUniforms.eyePos);

    // Base sky color (analytical)
    var sky = skyColor(rayDir, skyUniforms.sunDir, skyUniforms.time);

    // Volumetric clouds
    let clouds = raymarchClouds(
        skyUniforms.eyePos,
        rayDir,
        skyUniforms.sunDir,
        skyUniforms.time,
        skyUniforms.cloudCoverage,
        skyUniforms.cloudSpeed,
    );
    sky = mix(sky, clouds.rgb, clouds.a);

    return vec4f(sky, 1.0);
}
