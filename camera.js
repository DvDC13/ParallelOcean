// Orbit camera with mouse controls
export class OrbitCamera {
    constructor(canvas) {
        this.canvas = canvas;
        this.distance = 150;
        this.theta = Math.PI * 0.3;   // up-down angle (0 = top, PI/2 = side)
        this.phi = Math.PI * 0.25;     // left-right angle
        this.target = [0, 0, 0];
        this.fov = Math.PI / 3;        // 60 degrees
        this.near = 0.1;
        this.far = 5000;

        this._dragging = false;
        this._lastX = 0;
        this._lastY = 0;

        canvas.addEventListener('mousedown', (e) => {
            this._dragging = true;
            this._lastX = e.clientX;
            this._lastY = e.clientY;
        });
        window.addEventListener('mouseup', () => { this._dragging = false; });
        window.addEventListener('mousemove', (e) => {
            if (!this._dragging) return;
            const dx = e.clientX - this._lastX;
            const dy = e.clientY - this._lastY;
            this._lastX = e.clientX;
            this._lastY = e.clientY;
            this.phi -= dx * 0.005;
            this.theta = Math.max(0.05, Math.min(Math.PI * 0.75, this.theta + dy * 0.005));
        });
        canvas.addEventListener('wheel', (e) => {
            e.preventDefault();
            this.distance *= 1 + e.deltaY * 0.001;
            this.distance = Math.max(5, Math.min(1000, this.distance));
        });
    }

    get eye() {
        return [
            this.target[0] + this.distance * Math.sin(this.theta) * Math.cos(this.phi),
            this.target[1] + this.distance * Math.cos(this.theta),
            this.target[2] + this.distance * Math.sin(this.theta) * Math.sin(this.phi),
        ];
    }

    viewMatrix() { return lookAt(this.eye, this.target, [0, 1, 0]); }
    projMatrix() { return perspective(this.fov, this.canvas.width / this.canvas.height, this.near, this.far); }
}

// ---- Math utilities ----

export function perspective(fov, aspect, near, far) {
    const f = 1.0 / Math.tan(fov / 2);
    const nf = 1 / (near - far);
    return new Float32Array([
        f / aspect, 0, 0, 0,
        0, f, 0, 0,
        0, 0, (far + near) * nf, -1,
        0, 0, (2 * far * near) * nf, 0
    ]);
}

export function lookAt(eye, center, up) {
    // Z axis = normalize(eye - center)
    const zx = eye[0] - center[0], zy = eye[1] - center[1], zz = eye[2] - center[2];
    let len = 1 / Math.hypot(zx, zy, zz);
    const z0 = zx * len, z1 = zy * len, z2 = zz * len;

    // X axis = normalize(cross(up, Z))
    const xx = up[1] * z2 - up[2] * z1;
    const xy = up[2] * z0 - up[0] * z2;
    const xz = up[0] * z1 - up[1] * z0;
    len = 1 / Math.hypot(xx, xy, xz);
    const x0 = xx * len, x1 = xy * len, x2 = xz * len;

    // Y axis = cross(Z, X)
    const y0 = z1 * x2 - z2 * x1;
    const y1 = z2 * x0 - z0 * x2;
    const y2 = z0 * x1 - z1 * x0;

    return new Float32Array([
        x0, y0, z0, 0,
        x1, y1, z1, 0,
        x2, y2, z2, 0,
        -(x0 * eye[0] + x1 * eye[1] + x2 * eye[2]),
        -(y0 * eye[0] + y1 * eye[1] + y2 * eye[2]),
        -(z0 * eye[0] + z1 * eye[1] + z2 * eye[2]),
        1
    ]);
}

export function mat4Multiply(a, b) {
    const out = new Float32Array(16);
    for (let i = 0; i < 4; i++) {
        for (let j = 0; j < 4; j++) {
            out[j * 4 + i] =
                a[i] * b[j * 4] +
                a[4 + i] * b[j * 4 + 1] +
                a[8 + i] * b[j * 4 + 2] +
                a[12 + i] * b[j * 4 + 3];
        }
    }
    return out;
}
