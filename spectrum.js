// Generate initial Tessendorf spectrum h̃₀(k) on CPU
// Uses natural FFT order: k_n = n for n < N/2, k_n = n-N for n >= N/2

function gaussianRandom() {
    const u1 = Math.random();
    const u2 = Math.random();
    const r = Math.sqrt(-2.0 * Math.log(u1 + 1e-10));
    return [r * Math.cos(2 * Math.PI * u2), r * Math.sin(2 * Math.PI * u2)];
}

function phillips(kx, kz, windDir, windSpeed, A, g) {
    const kLen2 = kx * kx + kz * kz;
    if (kLen2 < 1e-12) return 0;

    const kLen4 = kLen2 * kLen2;
    const L = (windSpeed * windSpeed) / g;
    const L2 = L * L;
    const kLen = Math.sqrt(kLen2);
    const kNorm = [kx / kLen, kz / kLen];
    const wDot = kNorm[0] * windDir[0] + kNorm[1] * windDir[1];
    const dirFactor = wDot * wDot;
    const l = L * 0.001;
    const suppress = Math.exp(-kLen2 * l * l);

    return A * (Math.exp(-1.0 / (kLen2 * L2)) / kLen4) * dirFactor * suppress;
}

/**
 * Generate initial spectrum in natural FFT order.
 * Returns Float32Array of N*N*4 floats: [h0_re, h0_im, h0conj_re, h0conj_im]
 * and omega as Float32Array of N*N floats.
 */
export function generateInitialSpectrum(N, patchSize, params = {}) {
    const {
        windSpeed = 40,
        windAngle = Math.PI / 4,
        amplitude = 600,
        gravity = 9.81,
    } = params;

    const winDir = [Math.cos(windAngle), Math.sin(windAngle)];
    const h0 = new Float32Array(N * N * 4);
    const omega = new Float32Array(N * N);

    for (let m = 0; m < N; m++) {
        for (let n = 0; n < N; n++) {
            // Natural FFT order: frequency = n for n < N/2, n-N for n >= N/2
            const kn = n < N / 2 ? n : n - N;
            const km = m < N / 2 ? m : m - N;
            const kx = (2 * Math.PI * kn) / patchSize;
            const kz = (2 * Math.PI * km) / patchSize;

            const kLen = Math.sqrt(kx * kx + kz * kz);

            omega[m * N + n] = Math.sqrt(gravity * kLen);

            const ph = phillips(kx, kz, winDir, windSpeed, amplitude, gravity);
            const sqrtPh = Math.sqrt(ph * 0.5);
            const [gr, gi] = gaussianRandom();

            // Conjugate at (-kx, -kz)
            const ph_neg = phillips(-kx, -kz, winDir, windSpeed, amplitude, gravity);
            const sqrtPhNeg = Math.sqrt(ph_neg * 0.5);
            const [grNeg, giNeg] = gaussianRandom();

            const idx = (m * N + n) * 4;
            h0[idx] = gr * sqrtPh;
            h0[idx + 1] = gi * sqrtPh;
            h0[idx + 2] = grNeg * sqrtPhNeg;
            h0[idx + 3] = giNeg * sqrtPhNeg;
        }
    }

    return { h0, omega };
}