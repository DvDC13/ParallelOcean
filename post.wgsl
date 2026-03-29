// ==========================================
// Post-processing shaders for ParallelOcean
// ==========================================

struct PostParams {
    resolution: vec2f,       // 0-7
    bloomStrength: f32,      // 8-11
    bloomThreshold: f32,     // 12-15
    exposure: f32,           // 16-19
    contrast: f32,           // 20-23
    saturation: f32,         // 24-27
    vignetteStrength: f32,   // 28-31
    rainIntensity: f32,      // 32-35
    lightningFlash: f32,     // 36-39
    time: f32,               // 40-43
    cameraY: f32,            // 44-47
    waterLevel: f32,         // 48-51
    sunScreenX: f32,         // 52-55
    sunScreenY: f32,         // 56-59
    godRayStrength: f32,     // 60-63
};

struct RainCamera {
    invViewProj: mat4x4f,
    viewProj: mat4x4f,
    eyePos: vec4f,
};

// ---- Fullscreen triangle vertex shader (shared by all post passes) ----
struct PostVsOut {
    @builtin(position) pos: vec4f,
    @location(0) uv: vec2f,
};

@vertex
fn postVs(@builtin(vertex_index) vid: u32) -> PostVsOut {
    let uv = vec2f(f32((vid << 1u) & 2u), f32(vid & 2u));
    var out: PostVsOut;
    out.pos = vec4f(uv * 2.0 - 1.0, 0.0, 1.0);
    out.uv = vec2f(uv.x, 1.0 - uv.y);
    return out;
}

// ==========================================
// Pass 1: Bloom bright extract
// ==========================================
@group(0) @binding(0) var extractTex: texture_2d<f32>;
@group(0) @binding(1) var extractSampler: sampler;
@group(0) @binding(2) var<uniform> extractParams: PostParams;

@fragment
fn bloomExtractFs(inp: PostVsOut) -> @location(0) vec4f {
    let color = textureSample(extractTex, extractSampler, inp.uv).rgb;
    let threshold = extractParams.bloomThreshold;
    let brightness = max(color.r, max(color.g, color.b));
    let knee = threshold * 0.5;
    let soft = clamp(brightness - threshold + knee, 0.0, 2.0 * knee);
    let contribution = soft * soft / (4.0 * knee + 0.0001);
    var bloom = color * clamp(contribution / (brightness + 0.0001), 0.0, 1.0);
    bloom += max(color - vec3f(threshold), vec3f(0.0));
    return vec4f(bloom, 1.0);
}

// ==========================================
// Pass 2-3: Gaussian blur (separable H and V)
// ==========================================
@group(0) @binding(0) var blurTex: texture_2d<f32>;
@group(0) @binding(1) var blurSampler: sampler;
@group(0) @binding(2) var<uniform> blurDir: vec4f;

@fragment
fn blurFs(inp: PostVsOut) -> @location(0) vec4f {
    let dir = blurDir.xy;
    var color = vec3f(0.0);
    color += textureSample(blurTex, blurSampler, inp.uv + dir * -6.0).rgb * 0.0122;
    color += textureSample(blurTex, blurSampler, inp.uv + dir * -5.0).rgb * 0.0280;
    color += textureSample(blurTex, blurSampler, inp.uv + dir * -4.0).rgb * 0.0537;
    color += textureSample(blurTex, blurSampler, inp.uv + dir * -3.0).rgb * 0.0861;
    color += textureSample(blurTex, blurSampler, inp.uv + dir * -2.0).rgb * 0.1156;
    color += textureSample(blurTex, blurSampler, inp.uv + dir * -1.0).rgb * 0.1297;
    color += textureSample(blurTex, blurSampler, inp.uv).rgb             * 0.1493;
    color += textureSample(blurTex, blurSampler, inp.uv + dir *  1.0).rgb * 0.1297;
    color += textureSample(blurTex, blurSampler, inp.uv + dir *  2.0).rgb * 0.1156;
    color += textureSample(blurTex, blurSampler, inp.uv + dir *  3.0).rgb * 0.0861;
    color += textureSample(blurTex, blurSampler, inp.uv + dir *  4.0).rgb * 0.0537;
    color += textureSample(blurTex, blurSampler, inp.uv + dir *  5.0).rgb * 0.0280;
    color += textureSample(blurTex, blurSampler, inp.uv + dir *  6.0).rgb * 0.0122;
    return vec4f(color, 1.0);
}

// ==========================================
// 3D Rain effect — camera-aware with parallax
// ==========================================

fn rainHash(p: vec2f) -> f32 {
    return fract(sin(dot(p, vec2f(127.1, 311.7))) * 43758.5453);
}

fn rainLayer(
    uv: vec2f,
    eyePos: vec3f,
    time: f32,
    scaleX: f32,
    scaleY: f32,
    speed: f32,
    parallax: f32,
    tilt: f32,
    brightness: f32
) -> f32 {
    var p = vec2f(uv.x * scaleX, uv.y * scaleY);

    // Camera parallax — shifts layers at different rates for depth
    p.x += eyePos.x * parallax;
    p.y += eyePos.z * parallax * 0.5;

    // Falling
    p.y += time * speed;
    // Wind tilt
    p.x += p.y * tilt;

    let cell = floor(p);
    let f = fract(p);
    let rand = rainHash(cell);

    // ~40% of cells have drops
    if (rand > 0.4) { return 0.0; }

    // Drop position and shape
    let dropX = fract(rand * 7.3) * 0.4 + 0.3;
    let dx = abs(f.x - dropX);

    // Thin streak
    let width = 0.006 + fract(rand * 31.1) * 0.006;
    let streak = smoothstep(width, 0.0, dx);

    // Vertical extent — long tapered streak
    let dropLen = fract(rand * 13.7) * 0.35 + 0.4;
    let yOff = fract(rand * 5.7) * 0.2;
    let yMask = smoothstep(yOff, yOff + 0.05, f.y)
              * smoothstep(yOff + dropLen, yOff + dropLen - 0.15, f.y);

    // Per-drop brightness variation
    let bri = 0.5 + fract(rand * 17.3) * 0.5;

    return streak * yMask * brightness * bri;
}

fn rainEffect(uv: vec2f, time: f32, intensity: f32, eyePos: vec3f) -> vec3f {
    if (intensity < 0.01) { return vec3f(0.0); }

    var rain = 0.0;

    // 5 layers — near to far (bigger parallax = nearer)
    rain += rainLayer(uv, eyePos, time, 30.0,  3.5,  4.0,  0.030,  0.08,  0.35);
    rain += rainLayer(uv, eyePos, time, 45.0,  4.5,  5.5,  0.018,  0.06,  0.28);
    rain += rainLayer(uv, eyePos, time, 65.0,  6.0,  7.0,  0.010,  0.10,  0.22);
    rain += rainLayer(uv, eyePos, time, 90.0,  8.0,  9.0,  0.005,  0.07,  0.16);
    rain += rainLayer(uv, eyePos, time, 120.0, 10.0, 11.5, 0.002,  0.09,  0.10);

    let rainColor = vec3f(0.8, 0.83, 0.9);
    return rainColor * rain * intensity;
}

// ==========================================
// God rays — screen-space radial blur from sun
// ==========================================

fn godRays(uv: vec2f, sunUV: vec2f, strength: f32) -> vec3f {
    if (strength < 0.01) { return vec3f(0.0); }

    // Allow sun slightly off-screen, but not way off
    if (sunUV.x < -1.0 || sunUV.x > 2.0 || sunUV.y < -1.0 || sunUV.y > 2.0) {
        return vec3f(0.0);
    }

    let NUM_SAMPLES = 40;
    let decay = 0.97;
    let density = 0.5;
    let weight = 0.012;

    let deltaUV = (uv - sunUV) * density / f32(NUM_SAMPLES);
    var samplePos = uv;
    var illumination = 0.0;
    var currentDecay = 1.0;

    for (var i = 0; i < NUM_SAMPLES; i++) {
        samplePos -= deltaUV;
        let clamped = clamp(samplePos, vec2f(0.002), vec2f(0.998));
        // Sample from bloom texture — only bright areas contribute
        let bloomSample = textureSample(bloomTex, compositeSampler, clamped).rgb;
        let brightness = dot(bloomSample, vec3f(0.299, 0.587, 0.114));
        // Only accumulate from sky region (top of screen)
        let skyWeight = smoothstep(0.55, 0.3, clamped.y);
        illumination += brightness * currentDecay * weight * skyWeight;
        currentDecay *= decay;
    }

    let rayColor = vec3f(1.0, 0.9, 0.7);
    let sunDist = length(uv - sunUV);
    let falloff = smoothstep(1.2, 0.0, sunDist);

    return rayColor * illumination * strength * falloff;
}

// ==========================================
// Underwater effect
// ==========================================

fn underwaterHash(p: vec2f) -> f32 {
    return fract(sin(dot(p, vec2f(127.1, 311.7))) * 43758.5453);
}

fn underwaterNoise(p: vec2f) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let a = underwaterHash(i);
    let b = underwaterHash(i + vec2f(1.0, 0.0));
    let c = underwaterHash(i + vec2f(0.0, 1.0));
    let d = underwaterHash(i + vec2f(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// Sharp caustic network — thin bright lines like real refracted light
fn underwaterCaustics(uv: vec2f, time: f32) -> f32 {
    var c = 0.0;
    for (var layer = 0; layer < 2; layer++) {
        let fl = f32(layer);
        let scale = 15.0 + fl * 10.0;
        let speed = vec2f(0.3 + fl * 0.15, 0.2 - fl * 0.1);
        let p = uv * scale + time * speed + fl * vec2f(50.0, 30.0);

        // Voronoi edge detection — bright on cell boundaries
        let cell = floor(p);
        var minDist1 = 10.0;
        var minDist2 = 10.0;
        for (var dy = -1; dy <= 1; dy++) {
            for (var dx = -1; dx <= 1; dx++) {
                let neighbor = cell + vec2f(f32(dx), f32(dy));
                let point = neighbor + vec2f(
                    underwaterHash(neighbor + fl * 7.1),
                    underwaterHash(neighbor + vec2f(37.0, 17.0) + fl * 13.3)
                );
                let d = length(point - p);
                if (d < minDist1) {
                    minDist2 = minDist1;
                    minDist1 = d;
                } else if (d < minDist2) {
                    minDist2 = d;
                }
            }
        }
        let edge = minDist2 - minDist1;
        c += pow(smoothstep(0.15, 0.0, edge), 2.0);
    }
    return c * 0.5;
}

// Floating particles (plankton/dust)
fn underwaterParticles(uv: vec2f, time: f32) -> f32 {
    var total = 0.0;
    for (var i = 0; i < 4; i++) {
        let fi = f32(i);
        let scale = 20.0 + fi * 15.0;
        let speed = vec2f(0.02 + fi * 0.01, -0.05 - fi * 0.02);
        let p = uv * scale + time * speed + fi * vec2f(17.3, 31.7);
        let cell = floor(p);
        let f = fract(p);
        let rnd = vec2f(
            underwaterHash(cell + fi * 7.1),
            underwaterHash(cell + fi * 7.1 + vec2f(0.0, 53.0))
        );
        let dist = length(f - rnd);
        let size = 0.03 + underwaterHash(cell + fi * 3.3) * 0.06;
        let dot_val = smoothstep(size, size * 0.3, dist);
        let twinkle = sin(time * (2.0 + fi) + underwaterHash(cell) * 6.28) * 0.5 + 0.5;
        total += dot_val * twinkle * (0.4 + fi * 0.15);
    }
    return clamp(total, 0.0, 1.0);
}

fn underwaterEffect(color: vec3f, uv: vec2f, time: f32, depth: f32) -> vec3f {
    let depthFactor = clamp(depth * 0.04, 0.0, 1.0);
    let up = 1.0 - uv.y;

    // Strong vertical gradient: bright teal at top, deep blue at bottom
    let brightColor = vec3f(0.02, 0.18, 0.25);
    let darkColor = vec3f(0.005, 0.03, 0.06);
    var result = mix(darkColor, brightColor, pow(up, 0.8));
    result *= mix(1.0, 0.3, depthFactor);

    // Visible surface from below: bright wavy zone at top
    let surfaceZone = pow(smoothstep(0.5, 0.95, up), 1.5);
    let w1 = sin(uv.x * 15.0 + time * 1.5) * 0.5 + 0.5;
    let w2 = sin(uv.x * 25.0 - time * 1.0 + 2.3) * 0.5 + 0.5;
    let w3 = sin(uv.x * 8.0 + time * 0.7 + 1.1) * 0.5 + 0.5;
    let wavePattern = w1 * 0.4 + w2 * 0.3 + w3 * 0.3;
    let surfaceLight = surfaceZone * (0.6 + wavePattern * 0.4);
    result += vec3f(0.12, 0.35, 0.4) * surfaceLight * (1.0 - depthFactor * 0.5);
    let highlights = pow(w1 * w2, 1.5) * surfaceZone;
    result += vec3f(0.2, 0.4, 0.35) * highlights * (1.0 - depthFactor);

    // God rays: diagonal light shafts
    let rayAngle = 0.45;
    let cosA = cos(rayAngle);
    let sinA = sin(rayAngle);
    let ru = uv.x * cosA + (1.0 - uv.y) * sinA;
    var rayTotal = 0.0;
    for (var i = 0; i < 4; i++) {
        let fi = f32(i);
        let center = 0.15 + fi * 0.22 + sin(time * 0.1 + fi * 2.0) * 0.03;
        let width = 0.04 + fi * 0.01;
        let beam = exp(-pow((ru - center) / width, 2.0));
        rayTotal += beam * (0.7 + fi * 0.1);
    }
    rayTotal *= smoothstep(0.0, 0.7, up);
    let rn = underwaterNoise(vec2f(ru * 4.0 + time * 0.05, up * 3.0));
    rayTotal *= 0.7 + rn * 0.3;
    result += vec3f(0.05, 0.15, 0.18) * rayTotal * (1.0 - depthFactor * 0.5);

    // Caustics: subtle, only near surface
    let caustics = underwaterCaustics(uv, time);
    let causticsNearSurface = pow(up, 3.0) * (1.0 - depthFactor);
    result += vec3f(0.01, 0.02, 0.02) * caustics * causticsNearSurface;

    // Tiny floating particles
    let particles = underwaterParticles(uv, time);
    result += vec3f(0.08, 0.1, 0.09) * particles * 0.3;

    // Blend in some original scene (fish, objects)
    let lum = dot(color, vec3f(0.299, 0.587, 0.114));
    let sceneTinted = color * vec3f(0.3, 0.7, 0.8);
    let sceneVis = min(lum * 2.0, 1.0) * mix(0.6, 0.15, depthFactor);
    result = mix(result, result + sceneTinted * 0.4, sceneVis);

    // Soft vignette
    let vig = 1.0 - smoothstep(0.6, 1.4, length((uv - 0.5) * 1.5)) * 0.3;
    result *= vig;

    return result;
}

// ==========================================
// Pass 4: Final composite
// ==========================================
@group(0) @binding(0) var sceneTex: texture_2d<f32>;
@group(0) @binding(1) var bloomTex: texture_2d<f32>;
@group(0) @binding(2) var compositeSampler: sampler;
@group(0) @binding(3) var<uniform> postUniforms: PostParams;
@group(0) @binding(4) var<uniform> rainCamera: RainCamera;

@fragment
fn compositeFs(inp: PostVsOut) -> @location(0) vec4f {
    // Underwater detection
    let isUnderwater = postUniforms.cameraY < postUniforms.waterLevel;
    let underwaterDepth = select(0.0, postUniforms.waterLevel - postUniforms.cameraY, isUnderwater);

    // Underwater UV distortion — wavering view
    var sampleUV = inp.uv;
    if (isUnderwater) {
        let t = postUniforms.time;
        sampleUV.x += sin(inp.uv.y * 25.0 + t * 1.5) * 0.006;
        sampleUV.y += cos(inp.uv.x * 20.0 + t * 1.2) * 0.004;
    }

    var color = textureSample(sceneTex, compositeSampler, sampleUV).rgb;
    let bloom = textureSample(bloomTex, compositeSampler, sampleUV).rgb;

    if (isUnderwater) {
        // UNDERWATER: apply early, before bloom/tonemapping can wash it out
        color = underwaterEffect(color, inp.uv, postUniforms.time, underwaterDepth);
        color *= 2.5;
        color = color / (color + vec3f(1.0));
    } else {
        // Lightning flash — boost scene brightness
        let flash = postUniforms.lightningFlash;
        color += color * flash * 3.0;
        color += vec3f(0.6, 0.65, 0.8) * flash;

        // Add bloom
        color += bloom * postUniforms.bloomStrength;

        // God rays — radial light shafts from sun
        let sunUV = vec2f(postUniforms.sunScreenX, postUniforms.sunScreenY);
        let rays = godRays(inp.uv, sunUV, postUniforms.godRayStrength);
        color += rays;

        // Exposure
        color *= postUniforms.exposure;

        // ACES tonemap
        let a = 2.51; let b = 0.03; let c = 2.43; let d = 0.59; let e = 0.14;
        color = clamp((color * (a * color + b)) / (color * (c * color + d) + e), vec3f(0.0), vec3f(1.0));

        // Contrast
        let midpoint = vec3f(0.5);
        color = midpoint + (color - midpoint) * postUniforms.contrast;
        color = clamp(color, vec3f(0.0), vec3f(1.0));

        // Saturation
        let gray = dot(color, vec3f(0.2126, 0.7152, 0.0722));
        color = mix(vec3f(gray), color, postUniforms.saturation);

        // Rain
        let ri = postUniforms.rainIntensity;
        let mistGrey = vec3f(0.45, 0.48, 0.52);
        color = mix(color, mistGrey, ri * 0.18);
        let rainStreaks = rainEffect(inp.uv, postUniforms.time, ri, rainCamera.eyePos.xyz);
        color += rainStreaks;
        let rainDesat = 1.0 - ri * 0.1;
        let grayR = dot(color, vec3f(0.2126, 0.7152, 0.0722));
        color = mix(vec3f(grayR), color, rainDesat);
    }

    // Vignette (stronger during storm)
    let vuv = inp.uv * 2.0 - 1.0;
    let aspect = postUniforms.resolution.x / postUniforms.resolution.y;
    let dist = length(vuv * vec2f(1.0, 1.0 / aspect));
    let vigStr = postUniforms.vignetteStrength + select(postUniforms.rainIntensity * 0.2, 0.0, isUnderwater);
    let vignette = 1.0 - smoothstep(0.5, 1.5, dist) * vigStr;
    color *= vignette;

    // Lightning flash overlay — not underwater
    if (!isUnderwater) {
        let flashOverlay = postUniforms.lightningFlash;
        color = mix(color, vec3f(0.9, 0.92, 0.95), flashOverlay * 0.15);
    }

    return vec4f(clamp(color, vec3f(0.0), vec3f(1.0)), 1.0);
}
