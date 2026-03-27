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
    time: f32,               // 32-35
    sunScreenX: f32,         // 36-39
    sunScreenY: f32,         // 40-43
    godRayStrength: f32,     // 44-47
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
// Pass 4: Final composite
// ==========================================
@group(0) @binding(0) var sceneTex: texture_2d<f32>;
@group(0) @binding(1) var bloomTex: texture_2d<f32>;
@group(0) @binding(2) var compositeSampler: sampler;
@group(0) @binding(3) var<uniform> postUniforms: PostParams;

@fragment
fn compositeFs(inp: PostVsOut) -> @location(0) vec4f {
    var color = textureSample(sceneTex, compositeSampler, inp.uv).rgb;
    let bloom = textureSample(bloomTex, compositeSampler, inp.uv).rgb;

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

    // Vignette
    let vuv = inp.uv * 2.0 - 1.0;
    let aspect = postUniforms.resolution.x / postUniforms.resolution.y;
    let dist = length(vuv * vec2f(1.0, 1.0 / aspect));
    let vignette = 1.0 - smoothstep(0.5, 1.5, dist) * postUniforms.vignetteStrength;
    color *= vignette;

    return vec4f(clamp(color, vec3f(0.0), vec3f(1.0)), 1.0);
}
