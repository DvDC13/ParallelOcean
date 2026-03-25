import { OrbitCamera } from "./camera.js";

function resize(canvas) {
    const dpr = window.devicePixelRatio || 1;
    const width = Math.floor(canvas.clientWidth * dpr);
    const height = Math.floor(canvas.clientHeight * dpr);

    if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
    }
}

async function main() {
    const canvas = document.getElementById('canvas');
    if (!navigator.gpu) {
        document.body.innerText = 'WebGPU not supported';
        return;
    }

    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter.requestDevice({
        requiredLimits: {
            maxStorageBuffersPerShaderStage: Math.min(adapter.limits.maxStorageBuffersPerShaderStage, 16),
        },
    });

    const context = canvas.getContext('webgpu');
    const format = navigator.gpu.getPreferredCanvasFormat();
    context.configure({
        device,
        format,
        alphaMode: 'opaque',
    });

    resize(canvas);
    window.addEventListener('resize', () => resize(canvas));

    const camera = new OrbitCamera(canvas);
    camera.phi = Math.atan2(0.3, 0.5);
    camera.theta = Math.PI * 0.35;
}