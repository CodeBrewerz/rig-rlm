//! HRR (Holographic Reduced Representation) math primitives.
//!
//! All functions operate on complex-valued vectors with unit magnitude.
//! Binding is element-wise complex multiplication; unbinding uses conjugate.
//! Zero external dependencies — pure f64 math.

use std::f64::consts::TAU; // 2π

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A complex-valued vector stored in split (SoA) layout for cache-friendly
/// SIMD-able iteration.
#[derive(Debug, Clone)]
pub struct ComplexVector {
    pub re: Vec<f64>,
    pub im: Vec<f64>,
}

impl ComplexVector {
    /// Create a zero-filled complex vector of dimension `d`.
    pub fn zeros(d: usize) -> Self {
        Self {
            re: vec![0.0; d],
            im: vec![0.0; d],
        }
    }

    /// Dimension (length of re/im arrays).
    pub fn dim(&self) -> usize {
        self.re.len()
    }
}

// ---------------------------------------------------------------------------
// Seeded PRNG — Mulberry32
// ---------------------------------------------------------------------------

/// Mulberry32 PRNG — fast, deterministic, matching the TypeScript implementation.
pub struct Mulberry32 {
    state: u32,
}

impl Mulberry32 {
    pub fn new(seed: u32) -> Self {
        Self { state: seed }
    }

    /// Returns a value in [0, 1).
    pub fn next_f64(&mut self) -> f64 {
        self.state = self.state.wrapping_add(0x6d2b79f5);
        let mut t = self.state ^ (self.state >> 15);
        t = t.wrapping_mul(1 | self.state);
        t = (t.wrapping_add(t ^ (t >> 7)).wrapping_mul(61 | t)) ^ t;
        let val = (t ^ (t >> 14)) as f64;
        val / 4294967296.0
    }
}

/// Derive a u32 seed from a string name (matches TypeScript `seedFromName`).
pub fn seed_from_name(name: &str) -> u32 {
    let bytes = name.as_bytes();
    let mut padded = [0u8; 8];
    let len = bytes.len().min(8);
    padded[..len].copy_from_slice(&bytes[..len]);
    // Little-endian u32 from first 4 bytes
    u32::from_le_bytes([padded[0], padded[1], padded[2], padded[3]])
}

// ---------------------------------------------------------------------------
// Key generation
// ---------------------------------------------------------------------------

/// Create `v` unit-magnitude complex key vectors of dimension `d`.
/// Each entry has magnitude 1 (phase-only): key = exp(i * phi)
/// where phi ~ Uniform(0, 2π).
pub fn make_vocab_keys(v: usize, d: usize, rng: &mut Mulberry32) -> Vec<ComplexVector> {
    let mut keys = Vec::with_capacity(v);
    for _ in 0..v {
        let mut re = vec![0.0; d];
        let mut im = vec![0.0; d];
        for i in 0..d {
            let phi = TAU * rng.next_f64();
            re[i] = phi.cos();
            im[i] = phi.sin();
        }
        keys.push(ComplexVector { re, im });
    }
    keys
}

/// Create `l` role/position keys via successive powers of a base root.
/// role[k] = base^k where base = exp(2πi * arange(D) / D).
pub fn make_role_keys(d: usize, l: usize) -> Vec<ComplexVector> {
    let mut keys = Vec::with_capacity(l);
    for k in 0..l {
        let mut re = vec![0.0; d];
        let mut im = vec![0.0; d];
        for i in 0..d {
            let angle = (k as f64 * TAU * i as f64) / d as f64;
            re[i] = angle.cos();
            im[i] = angle.sin();
        }
        keys.push(ComplexVector { re, im });
    }
    keys
}

// ---------------------------------------------------------------------------
// Orthogonalization
// ---------------------------------------------------------------------------

/// Gram-Schmidt-like decorrelation in R^{2D}, projected back to unit phase.
///
/// 1. Stack real/imag → R^{2D}
/// 2. Iteratively subtract correlated components
/// 3. Re-normalise and convert back to unit-magnitude complex
pub fn orthogonalize(keys: &[ComplexVector], iters: usize, step: f64) -> Vec<ComplexVector> {
    if iters == 0 || keys.is_empty() {
        return keys.to_vec();
    }

    let v = keys.len();
    let d = keys[0].dim();
    let d2 = d * 2;

    // Stack to [V, 2D] real matrix (flat row-major)
    let mut k_mat = vec![0.0; v * d2];
    for (vi, key) in keys.iter().enumerate() {
        let off = vi * d2;
        k_mat[off..off + d].copy_from_slice(&key.re);
        k_mat[off + d..off + d2].copy_from_slice(&key.im);
    }

    for _ in 0..iters {
        // G = K @ K^T — [V, V] Gram matrix
        let mut g = vec![0.0; v * v];
        for i in 0..v {
            for j in i..v {
                let mut dot = 0.0;
                let off_i = i * d2;
                let off_j = j * d2;
                for dd in 0..d2 {
                    dot += k_mat[off_i + dd] * k_mat[off_j + dd];
                }
                g[i * v + j] = dot;
                g[j * v + i] = dot;
            }
            // Zero diagonal
            g[i * v + i] = 0.0;
        }

        // correction = G @ K
        let mut correction = vec![0.0; v * d2];
        for i in 0..v {
            for j in 0..v {
                let gv = g[i * v + j];
                if gv == 0.0 {
                    continue;
                }
                let off_i = i * d2;
                let off_j = j * d2;
                for dd in 0..d2 {
                    correction[off_i + dd] += gv * k_mat[off_j + dd];
                }
            }
        }

        // K = K - step * correction / D2
        let scale = step / d2 as f64;
        for i in 0..v * d2 {
            k_mat[i] -= scale * correction[i];
        }

        // Row-normalise
        for vi in 0..v {
            let off = vi * d2;
            let mut norm = 0.0;
            for dd in 0..d2 {
                norm += k_mat[off + dd] * k_mat[off + dd];
            }
            let inv_norm = 1.0 / (norm.sqrt() + 1e-9);
            for dd in 0..d2 {
                k_mat[off + dd] *= inv_norm;
            }
        }
    }

    // Convert back to unit-phase complex
    let mut result = Vec::with_capacity(v);
    for vi in 0..v {
        let off = vi * d2;
        let mut re = vec![0.0; d];
        let mut im = vec![0.0; d];
        for dd in 0..d {
            let r = k_mat[off + dd];
            let i = k_mat[off + d + dd];
            let phase = i.atan2(r);
            re[dd] = phase.cos();
            im[dd] = phase.sin();
        }
        result.push(ComplexVector { re, im });
    }
    result
}

// ---------------------------------------------------------------------------
// Signal processing
// ---------------------------------------------------------------------------

/// Magnitude-sharpening nonlinearity.
///
/// z_out = z * (|z| + eps)^(p - 1)
/// - p > 1 → contrast-increasing (amplifies high-mag, suppresses noise)
/// - p < 1 → softening
/// - p == 1 → identity
pub fn sharpen(z: &ComplexVector, p: f64) -> ComplexVector {
    if (p - 1.0).abs() < f64::EPSILON {
        return z.clone();
    }
    let d = z.dim();
    let mut re = vec![0.0; d];
    let mut im = vec![0.0; d];
    let exp = p - 1.0;
    let eps = 1e-12;
    for i in 0..d {
        let mag = (z.re[i] * z.re[i] + z.im[i] * z.im[i]).sqrt();
        let scale = (mag + eps).powf(exp);
        re[i] = z.re[i] * scale;
        im[i] = z.im[i] * scale;
    }
    ComplexVector { re, im }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_product_avx2(a: &[f64], b: &[f64]) -> f64 {
    use std::arch::x86_64::*;
    
    // Unroll 4-ways to avoid pipeline dependency stalls (Latencies of FMA -> 4 cycles)
    // By having 4 accumulators, we hit 2 FMAs per cycle perfectly on Zen/Core architectures.
    let mut sum0 = _mm256_setzero_pd();
    let mut sum1 = _mm256_setzero_pd();
    let mut sum2 = _mm256_setzero_pd();
    let mut sum3 = _mm256_setzero_pd();
    
    let mut i = 0;
    while i + 16 <= a.len() {
        let va0 = _mm256_loadu_pd(a.as_ptr().add(i));
        let vb0 = _mm256_loadu_pd(b.as_ptr().add(i));
        let va1 = _mm256_loadu_pd(a.as_ptr().add(i + 4));
        let vb1 = _mm256_loadu_pd(b.as_ptr().add(i + 4));
        let va2 = _mm256_loadu_pd(a.as_ptr().add(i + 8));
        let vb2 = _mm256_loadu_pd(b.as_ptr().add(i + 8));
        let va3 = _mm256_loadu_pd(a.as_ptr().add(i + 12));
        let vb3 = _mm256_loadu_pd(b.as_ptr().add(i + 12));

        sum0 = _mm256_fmadd_pd(va0, vb0, sum0);
        sum1 = _mm256_fmadd_pd(va1, vb1, sum1);
        sum2 = _mm256_fmadd_pd(va2, vb2, sum2);
        sum3 = _mm256_fmadd_pd(va3, vb3, sum3);
        i += 16;
    }
    
    // Fallback for remaining blocks of 4
    while i + 4 <= a.len() {
        let va0 = _mm256_loadu_pd(a.as_ptr().add(i));
        let vb0 = _mm256_loadu_pd(b.as_ptr().add(i));
        sum0 = _mm256_fmadd_pd(va0, vb0, sum0);
        i += 4;
    }

    sum0 = _mm256_add_pd(sum0, sum1);
    sum2 = _mm256_add_pd(sum2, sum3);
    sum0 = _mm256_add_pd(sum0, sum2);

    let mut arr = [0.0; 4];
    _mm256_storeu_pd(arr.as_mut_ptr(), sum0);
    let mut dot = arr[0] + arr[1] + arr[2] + arr[3];
    while i < a.len() {
        dot += a[i] * b[i];
        i += 1;
    }
    dot
}

/// Compute the exact dot product between two f64 slices, utilizing AVX2/FMA if available.
pub fn dot_product(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        return unsafe { dot_product_avx2(a, b) };
    }

    let mut dot = 0.0;
    for i in 0..a.len() {
        dot += a[i] * b[i];
    }
    dot
}

/// Gentle magnitude limiter (CORVACS-lite).
///
/// z_out = z * tanh(a * |z|) / |z|
/// - a == 0 → identity
/// - a > 0 → soft saturation
pub fn corvacs_lite(z: &ComplexVector, a: f64) -> ComplexVector {
    if a <= 0.0 {
        return z.clone();
    }
    let d = z.dim();
    let mut re = vec![0.0; d];
    let mut im = vec![0.0; d];
    for i in 0..d {
        let mag = (z.re[i] * z.re[i] + z.im[i] * z.im[i]).sqrt() + 1e-12;
        let scale = (a * mag).tanh() / mag;
        re[i] = z.re[i] * scale;
        im[i] = z.im[i] * scale;
    }
    ComplexVector { re, im }
}

/// Temperature-scaled softmax over similarity logits.
pub fn softmax_temp(sims: &[f64], t: f64) -> Vec<f64> {
    let t = t.max(1e-6);
    let mut z: Vec<f64> = sims.iter().map(|&s| s / t).collect();

    let max = z.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let mut sum = 0.0;
    for v in z.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }

    sum += 1e-12;
    for v in z.iter_mut() {
        *v /= sum;
    }
    z
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convert [V, D] complex → [V, 2D] real with unit row norms.
/// Used for efficient cosine similarity: sims = vocab_norm @ query_2d.
pub fn stack_and_unit_norm(keys: &[ComplexVector]) -> Vec<Vec<f64>> {
    if keys.is_empty() {
        return vec![];
    }
    let d = keys[0].dim();
    let d2 = d * 2;

    keys.iter()
        .map(|key| {
            let mut row = vec![0.0; d2];
            row[..d].copy_from_slice(&key.re);
            row[d..d2].copy_from_slice(&key.im);
            let mut norm = 0.0;
            for v in row.iter() {
                norm += v * v;
            }
            let inv_norm = 1.0 / (norm.sqrt() + 1e-12);
            for v in row.iter_mut() {
                *v *= inv_norm;
            }
            row
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Bind / Unbind
// ---------------------------------------------------------------------------

/// Bind: element-wise complex product a * b.
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Bind: (a.re + a.im*i) * (b.re + b.im*i)
pub fn bind(a: &ComplexVector, b: &ComplexVector) -> ComplexVector {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        return unsafe { bind_avx2(a, b) };
    }
    bind_fallback(a, b)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn bind_avx2(a: &ComplexVector, b: &ComplexVector) -> ComplexVector {
    let d = a.dim();
    let mut re = vec![0.0; d];
    let mut im = vec![0.0; d];

    let mut i = 0;
    while i + 4 <= d {
        unsafe {
            let ar = _mm256_loadu_pd(a.re.as_ptr().add(i));
            let ai = _mm256_loadu_pd(a.im.as_ptr().add(i));
            let br = _mm256_loadu_pd(b.re.as_ptr().add(i));
            let bi = _mm256_loadu_pd(b.im.as_ptr().add(i));

            let r = _mm256_fmsub_pd(ar, br, _mm256_mul_pd(ai, bi));
            let m = _mm256_fmadd_pd(ar, bi, _mm256_mul_pd(ai, br));

            _mm256_storeu_pd(re.as_mut_ptr().add(i), r);
            _mm256_storeu_pd(im.as_mut_ptr().add(i), m);
        }
        i += 4;
    }
    while i < d {
        re[i] = a.re[i] * b.re[i] - a.im[i] * b.im[i];
        im[i] = a.re[i] * b.im[i] + a.im[i] * b.re[i];
        i += 1;
    }
    ComplexVector { re, im }
}

fn bind_fallback(a: &ComplexVector, b: &ComplexVector) -> ComplexVector {
    let d = a.dim();
    let mut re = vec![0.0; d];
    let mut im = vec![0.0; d];
    for i in 0..d {
        re[i] = a.re[i] * b.re[i] - a.im[i] * b.im[i];
        im[i] = a.re[i] * b.im[i] + a.im[i] * b.re[i];
    }
    ComplexVector { re, im }
}

/// Unbind: m * conj(key).
pub fn unbind(m: &ComplexVector, key: &ComplexVector) -> ComplexVector {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        return unsafe { unbind_avx2(m, key) };
    }
    unbind_fallback(m, key)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn unbind_avx2(m: &ComplexVector, key: &ComplexVector) -> ComplexVector {
    let d = m.dim();
    let mut re = vec![0.0; d];
    let mut im = vec![0.0; d];

    let mut i = 0;
    while i + 4 <= d {
        unsafe {
            let mr = _mm256_loadu_pd(m.re.as_ptr().add(i));
            let mi = _mm256_loadu_pd(m.im.as_ptr().add(i));
            let kr = _mm256_loadu_pd(key.re.as_ptr().add(i));
            let ki = _mm256_loadu_pd(key.im.as_ptr().add(i));

            // re = mr*kr + mi*ki
            let r = _mm256_fmadd_pd(mr, kr, _mm256_mul_pd(mi, ki));
            // im = mi*kr - mr*ki
            let m_mul = _mm256_fmsub_pd(mi, kr, _mm256_mul_pd(mr, ki));

            _mm256_storeu_pd(re.as_mut_ptr().add(i), r);
            _mm256_storeu_pd(im.as_mut_ptr().add(i), m_mul);
        }
        i += 4;
    }
    while i < d {
        re[i] = m.re[i] * key.re[i] + m.im[i] * key.im[i];
        im[i] = -m.re[i] * key.im[i] + m.im[i] * key.re[i];
        i += 1;
    }
    ComplexVector { re, im }
}

fn unbind_fallback(m: &ComplexVector, key: &ComplexVector) -> ComplexVector {
    let d = m.dim();
    let mut re = vec![0.0; d];
    let mut im = vec![0.0; d];
    for i in 0..d {
        re[i] = m.re[i] * key.re[i] + m.im[i] * key.im[i];
        im[i] = -m.re[i] * key.im[i] + m.im[i] * key.re[i];
    }
    ComplexVector { re, im }
}

/// Unbind exactly into an existing ComplexVector (zero memory allocation)
pub fn unbind_into(m: &ComplexVector, key: &ComplexVector, out: &mut ComplexVector) {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        unsafe { unbind_into_avx2(m, key, out); return; }
    }
    let d = m.dim();
    for i in 0..d {
        out.re[i] = m.re[i] * key.re[i] + m.im[i] * key.im[i];
        out.im[i] = -m.re[i] * key.im[i] + m.im[i] * key.re[i];
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn unbind_into_avx2(m: &ComplexVector, key: &ComplexVector, out: &mut ComplexVector) {
    let d = m.dim();
    let mut i = 0;
    
    // 4-way unrolled
    while i + 16 <= d {
        let mr0 = _mm256_loadu_pd(m.re.as_ptr().add(i));
        let mi0 = _mm256_loadu_pd(m.im.as_ptr().add(i));
        let kr0 = _mm256_loadu_pd(key.re.as_ptr().add(i));
        let ki0 = _mm256_loadu_pd(key.im.as_ptr().add(i));
        
        let mr1 = _mm256_loadu_pd(m.re.as_ptr().add(i + 4));
        let mi1 = _mm256_loadu_pd(m.im.as_ptr().add(i + 4));
        let kr1 = _mm256_loadu_pd(key.re.as_ptr().add(i + 4));
        let ki1 = _mm256_loadu_pd(key.im.as_ptr().add(i + 4));

        let mr2 = _mm256_loadu_pd(m.re.as_ptr().add(i + 8));
        let mi2 = _mm256_loadu_pd(m.im.as_ptr().add(i + 8));
        let kr2 = _mm256_loadu_pd(key.re.as_ptr().add(i + 8));
        let ki2 = _mm256_loadu_pd(key.im.as_ptr().add(i + 8));

        let mr3 = _mm256_loadu_pd(m.re.as_ptr().add(i + 12));
        let mi3 = _mm256_loadu_pd(m.im.as_ptr().add(i + 12));
        let kr3 = _mm256_loadu_pd(key.re.as_ptr().add(i + 12));
        let ki3 = _mm256_loadu_pd(key.im.as_ptr().add(i + 12));

        let r0 = _mm256_fmadd_pd(mr0, kr0, _mm256_mul_pd(mi0, ki0));
        let i0 = _mm256_fmsub_pd(mi0, kr0, _mm256_mul_pd(mr0, ki0));
        
        let r1 = _mm256_fmadd_pd(mr1, kr1, _mm256_mul_pd(mi1, ki1));
        let i1 = _mm256_fmsub_pd(mi1, kr1, _mm256_mul_pd(mr1, ki1));
        
        let r2 = _mm256_fmadd_pd(mr2, kr2, _mm256_mul_pd(mi2, ki2));
        let i2 = _mm256_fmsub_pd(mi2, kr2, _mm256_mul_pd(mr2, ki2));
        
        let r3 = _mm256_fmadd_pd(mr3, kr3, _mm256_mul_pd(mi3, ki3));
        let i3 = _mm256_fmsub_pd(mi3, kr3, _mm256_mul_pd(mr3, ki3));

        _mm256_storeu_pd(out.re.as_mut_ptr().add(i), r0);
        _mm256_storeu_pd(out.im.as_mut_ptr().add(i), i0);
        _mm256_storeu_pd(out.re.as_mut_ptr().add(i + 4), r1);
        _mm256_storeu_pd(out.im.as_mut_ptr().add(i + 4), i1);
        _mm256_storeu_pd(out.re.as_mut_ptr().add(i + 8), r2);
        _mm256_storeu_pd(out.im.as_mut_ptr().add(i + 8), i2);
        _mm256_storeu_pd(out.re.as_mut_ptr().add(i + 12), r3);
        _mm256_storeu_pd(out.im.as_mut_ptr().add(i + 12), i3);
        i += 16;
    }
    
    while i + 4 <= d {
        let mr = _mm256_loadu_pd(m.re.as_ptr().add(i));
        let mi = _mm256_loadu_pd(m.im.as_ptr().add(i));
        let kr = _mm256_loadu_pd(key.re.as_ptr().add(i));
        let ki = _mm256_loadu_pd(key.im.as_ptr().add(i));
        let r = _mm256_fmadd_pd(mr, kr, _mm256_mul_pd(mi, ki));
        let m_mul = _mm256_fmsub_pd(mi, kr, _mm256_mul_pd(mr, ki));
        _mm256_storeu_pd(out.re.as_mut_ptr().add(i), r);
        _mm256_storeu_pd(out.im.as_mut_ptr().add(i), m_mul);
        i += 4;
    }
    
    while i < d {
        out.re[i] = m.re[i] * key.re[i] + m.im[i] * key.im[i];
        out.im[i] = -m.re[i] * key.im[i] + m.im[i] * key.re[i];
        i += 1;
    }
}

/// Unbind exactly into a dual-stacked real f64 array [re, im] without allocating.
pub fn unbind_into_real(m: &ComplexVector, key: &ComplexVector, out: &mut [f64]) {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        unsafe { unbind_into_real_avx2(m, key, out); return; }
    }
    let d = m.dim();
    for i in 0..d {
        out[i] = m.re[i] * key.re[i] + m.im[i] * key.im[i];
        out[d + i] = -m.re[i] * key.im[i] + m.im[i] * key.re[i];
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn unbind_into_real_avx2(m: &ComplexVector, key: &ComplexVector, out: &mut [f64]) {
    let d = m.dim();
    let mut i = 0;
    while i + 4 <= d {
        let mr = _mm256_loadu_pd(m.re.as_ptr().add(i));
        let mi = _mm256_loadu_pd(m.im.as_ptr().add(i));
        let kr = _mm256_loadu_pd(key.re.as_ptr().add(i));
        let ki = _mm256_loadu_pd(key.im.as_ptr().add(i));

        let r = _mm256_fmadd_pd(mr, kr, _mm256_mul_pd(mi, ki));
        let m_mul = _mm256_fmsub_pd(mi, kr, _mm256_mul_pd(mr, ki));

        _mm256_storeu_pd(out.as_mut_ptr().add(i), r);
        _mm256_storeu_pd(out.as_mut_ptr().add(d + i), m_mul);
        i += 4;
    }
    while i < d {
        out[i] = m.re[i] * key.re[i] + m.im[i] * key.im[i];
        out[d + i] = -m.re[i] * key.im[i] + m.im[i] * key.re[i];
        i += 1;
    }
}

// ---------------------------------------------------------------------------
// RoPE-style inline role rotation — eliminates role_keys storage
// ---------------------------------------------------------------------------

/// Pre-computed sin/cos lookup table for RoPE base frequencies.
/// base_sin[j] = sin(2π·j/D), base_cos[j] = cos(2π·j/D)
/// For position k: sin(2π·k·j/D) and cos(2π·k·j/D) are derived via
/// Chebyshev recurrence (no per-element sin_cos calls).
pub struct RopeLut {
    base_sin: Vec<f64>,
    base_cos: Vec<f64>,
}

impl RopeLut {
    /// Build the LUT for dimension d.
    pub fn new(d: usize) -> Self {
        let inv_d = 1.0 / d as f64;
        let mut base_sin = vec![0.0; d];
        let mut base_cos = vec![0.0; d];
        for j in 0..d {
            let theta = TAU * (j as f64) * inv_d;
            let (s, c) = theta.sin_cos();
            base_sin[j] = s;
            base_cos[j] = c;
        }
        RopeLut { base_sin, base_cos }
    }

    /// Compute sin(2π·pos·j/D) and cos(2π·pos·j/D) for all j using
    /// de Moivre / Chebyshev recurrence: z^k = z^(k-1) * z^1
    /// where z = cos(θ) + i·sin(θ), θ = 2πj/D.
    ///
    /// Returns (sin_table, cos_table) for position `pos`.
    #[inline]
    pub fn for_position(&self, pos: usize) -> (Vec<f64>, Vec<f64>) {
        let d = self.base_sin.len();
        if pos == 0 {
            return (vec![0.0; d], vec![1.0; d]);
        }
        if pos == 1 {
            return (self.base_sin.clone(), self.base_cos.clone());
        }
        // Use binary exponentiation: z^pos = product of z^(2^k) for bits of pos
        // This avoids pos×d sin_cos calls, doing only O(log(pos))×d multiplies
        let mut result_s = vec![0.0; d];
        let mut result_c = vec![1.0; d]; // z^0 = 1
        let mut base_s = self.base_sin.clone();
        let mut base_c = self.base_cos.clone();
        let mut p = pos;
        while p > 0 {
            if p & 1 == 1 {
                // result *= base  (complex multiply)
                for j in 0..d {
                    let rc = result_c[j];
                    let rs = result_s[j];
                    result_c[j] = rc * base_c[j] - rs * base_s[j];
                    result_s[j] = rc * base_s[j] + rs * base_c[j];
                }
            }
            // base *= base (square)
            for j in 0..d {
                let bc = base_c[j];
                let bs = base_s[j];
                base_c[j] = bc * bc - bs * bs;
                base_s[j] = 2.0 * bc * bs;
            }
            p >>= 1;
        }
        (result_s, result_c)
    }
}

/// Bind a vector with role[position] inline (RoPE-style).
/// Uses sin_cos LUT when available, falls back to direct computation.
pub fn bind_role_inline(v: &ComplexVector, pos: usize) -> ComplexVector {
    let d = v.dim();
    let mut re = vec![0.0; d];
    let mut im = vec![0.0; d];
    let inv_d = 1.0 / d as f64;
    for j in 0..d {
        let theta = TAU * (pos as f64) * (j as f64) * inv_d;
        let (st, ct) = theta.sin_cos();
        re[j] = ct * v.re[j] - st * v.im[j];
        im[j] = ct * v.im[j] + st * v.re[j];
    }
    ComplexVector { re, im }
}

/// Bind using pre-computed RoPE LUT (zero sin_cos calls).
pub fn bind_role_lut(v: &ComplexVector, pos: usize, lut: &RopeLut) -> ComplexVector {
    let d = v.dim();
    let (sin_k, cos_k) = lut.for_position(pos);
    let mut re = vec![0.0; d];
    let mut im = vec![0.0; d];
    for j in 0..d {
        re[j] = cos_k[j] * v.re[j] - sin_k[j] * v.im[j];
        im[j] = cos_k[j] * v.im[j] + sin_k[j] * v.re[j];
    }
    ComplexVector { re, im }
}

/// Unbind role[position] inline (RoPE-style, no LUT).
pub fn unbind_role_inline(v: &ComplexVector, pos: usize) -> ComplexVector {
    let d = v.dim();
    let mut re = vec![0.0; d];
    let mut im = vec![0.0; d];
    let inv_d = 1.0 / d as f64;
    for j in 0..d {
        let theta = TAU * (pos as f64) * (j as f64) * inv_d;
        let (st, ct) = theta.sin_cos();
        re[j] = v.re[j] * ct + v.im[j] * st;
        im[j] = v.im[j] * ct - v.re[j] * st;
    }
    ComplexVector { re, im }
}

/// Unbind using pre-computed RoPE LUT (zero sin_cos calls).
pub fn unbind_role_lut(v: &ComplexVector, pos: usize, lut: &RopeLut) -> ComplexVector {
    let d = v.dim();
    let (sin_k, cos_k) = lut.for_position(pos);
    let mut re = vec![0.0; d];
    let mut im = vec![0.0; d];
    for j in 0..d {
        re[j] = v.re[j] * cos_k[j] + v.im[j] * sin_k[j];
        im[j] = v.im[j] * cos_k[j] - v.re[j] * sin_k[j];
    }
    ComplexVector { re, im }
}

/// Unbind role[position] inline into an existing ComplexVector (zero alloc).
pub fn unbind_role_inline_into(v: &ComplexVector, pos: usize, out: &mut ComplexVector) {
    let d = v.dim();
    let inv_d = 1.0 / d as f64;
    for j in 0..d {
        let theta = TAU * (pos as f64) * (j as f64) * inv_d;
        let (st, ct) = theta.sin_cos();
        out.re[j] = v.re[j] * ct + v.im[j] * st;
        out.im[j] = v.im[j] * ct - v.re[j] * st;
    }
}

// ---------------------------------------------------------------------------
// Lazy codebook regeneration — eliminates vocab_keys storage
// ---------------------------------------------------------------------------

/// Regenerate a single codebook vector by index, without materializing the
/// full vocabulary. Skips the PRNG to position `index` then generates one vector.
///
/// This is **losslessly identical** to `make_vocab_keys(V, d, &mut rng)[index]`.
pub fn regenerate_vocab_key(seed: u32, index: usize, d: usize) -> ComplexVector {
    let mut rng = Mulberry32::new(seed);
    // Skip previous vectors: each vector consumes d random values
    for _ in 0..(index * d) {
        rng.next_f64();
    }
    // Generate the target vector
    let mut re = vec![0.0; d];
    let mut im = vec![0.0; d];
    for i in 0..d {
        let phi = TAU * rng.next_f64();
        re[i] = phi.cos();
        im[i] = phi.sin();
    }
    ComplexVector { re, im }
}

/// Regenerate the sent_key (sentence key). It's the first vector generated
/// after all V vocab keys have been consumed from the PRNG stream.
pub fn regenerate_sent_key(seed: u32, vocab_size: usize, d: usize) -> ComplexVector {
    regenerate_vocab_key(seed, vocab_size, d)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn magnitude(v: &ComplexVector, d: usize) -> f64 {
        (v.re[d] * v.re[d] + v.im[d] * v.im[d]).sqrt()
    }

    fn norm_2d(v: &ComplexVector) -> f64 {
        let mut sum = 0.0;
        for i in 0..v.dim() {
            sum += v.re[i] * v.re[i] + v.im[i] * v.im[i];
        }
        sum.sqrt()
    }

    #[test]
    fn mulberry32_deterministic() {
        let mut a = Mulberry32::new(42);
        let mut b = Mulberry32::new(42);
        for _ in 0..100 {
            assert_eq!(a.next_f64(), b.next_f64());
        }
    }

    #[test]
    fn mulberry32_range() {
        let mut rng = Mulberry32::new(12345);
        for _ in 0..1000 {
            let v = rng.next_f64();
            assert!(v >= 0.0 && v < 1.0, "value out of range: {v}");
        }
    }

    #[test]
    fn seed_from_name_consistent() {
        assert_eq!(seed_from_name("test"), seed_from_name("test"));
        assert_ne!(seed_from_name("a"), seed_from_name("b"));
    }

    #[test]
    fn make_vocab_keys_unit_magnitude() {
        let mut rng = Mulberry32::new(0);
        let keys = make_vocab_keys(5, 64, &mut rng);
        assert_eq!(keys.len(), 5);
        for key in &keys {
            assert_eq!(key.re.len(), 64);
            assert_eq!(key.im.len(), 64);
            for d in 0..64 {
                let mag = magnitude(key, d);
                assert!((mag - 1.0).abs() < 1e-10, "mag = {mag}");
            }
        }
    }

    #[test]
    fn make_role_keys_positions() {
        let keys = make_role_keys(64, 10);
        assert_eq!(keys.len(), 10);
        // role[0] should be all ones (base^0 = 1)
        for d in 0..64 {
            assert!(
                (keys[0].re[d] - 1.0).abs() < 1e-10,
                "re[{d}] = {}",
                keys[0].re[d]
            );
            assert!(keys[0].im[d].abs() < 1e-10, "im[{d}] = {}", keys[0].im[d]);
        }
    }

    #[test]
    fn bind_unbind_recovers() {
        let mut rng = Mulberry32::new(99);
        let keys = make_vocab_keys(2, 128, &mut rng);
        let a = &keys[0];
        let b = &keys[1];

        let bound = bind(a, b);
        let recovered = unbind(&bound, a);

        // cosine similarity between recovered and b
        let mut dot = 0.0;
        for d in 0..128 {
            dot += recovered.re[d] * b.re[d] + recovered.im[d] * b.im[d];
        }
        let sim = dot / (norm_2d(&recovered) * norm_2d(b));
        assert!(sim > 0.99, "similarity = {sim}");
    }

    #[test]
    fn orthogonalize_unit_magnitude() {
        let mut rng = Mulberry32::new(7);
        let keys = make_vocab_keys(5, 128, &mut rng);
        let orth = orthogonalize(&keys, 2, 0.4);
        assert_eq!(orth.len(), 5);
        for key in &orth {
            for d in 0..128 {
                let mag = magnitude(key, d);
                assert!((mag - 1.0).abs() < 1e-4, "mag = {mag}");
            }
        }
    }

    #[test]
    fn orthogonalize_zero_iters_noop() {
        let mut rng = Mulberry32::new(7);
        let keys = make_vocab_keys(3, 32, &mut rng);
        let result = orthogonalize(&keys, 0, 0.4);
        assert_eq!(result.len(), keys.len());
        // Values should be identical
        for (a, b) in result.iter().zip(keys.iter()) {
            for d in 0..32 {
                assert_eq!(a.re[d], b.re[d]);
                assert_eq!(a.im[d], b.im[d]);
            }
        }
    }

    #[test]
    fn sharpen_identity_at_p1() {
        let v = ComplexVector {
            re: vec![1.0, 2.0, 3.0],
            im: vec![4.0, 5.0, 6.0],
        };
        let result = sharpen(&v, 1.0);
        for i in 0..3 {
            assert_eq!(result.re[i], v.re[i]);
            assert_eq!(result.im[i], v.im[i]);
        }
    }

    #[test]
    fn sharpen_amplifies_high_mag() {
        let v = ComplexVector {
            re: vec![0.1, 1.0],
            im: vec![0.0, 0.0],
        };
        let result = sharpen(&v, 2.0);
        let mag0 = result.re[0].abs();
        let mag1 = result.re[1].abs();
        // ratio should be > 10 (more contrast than the original 10x)
        assert!(mag1 / mag0 > 10.0, "ratio = {}", mag1 / mag0);
    }

    #[test]
    fn corvacs_lite_identity_at_zero() {
        let v = ComplexVector {
            re: vec![1.0, 2.0, 3.0],
            im: vec![4.0, 5.0, 6.0],
        };
        let result = corvacs_lite(&v, 0.0);
        for i in 0..3 {
            assert_eq!(result.re[i], v.re[i]);
            assert_eq!(result.im[i], v.im[i]);
        }
    }

    #[test]
    fn softmax_temp_valid_probs() {
        let sims = [1.0, 2.0, 3.0];
        let probs = softmax_temp(&sims, 1.0);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-8, "sum = {sum}");
        for &p in &probs {
            assert!(p > 0.0);
        }
    }

    #[test]
    fn softmax_temp_higher_temp_more_uniform() {
        let sims = [1.0, 2.0, 3.0];
        let low = softmax_temp(&sims, 0.5);
        let high = softmax_temp(&sims, 5.0);
        let max_low = low.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let max_high = high.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(max_high < max_low);
    }

    #[test]
    fn stack_and_unit_norm_produces_unit_vectors() {
        let mut rng = Mulberry32::new(0);
        let keys = make_vocab_keys(3, 32, &mut rng);
        let normed = stack_and_unit_norm(&keys);
        assert_eq!(normed.len(), 3);
        for row in &normed {
            assert_eq!(row.len(), 64);
            let norm: f64 = row.iter().map(|v| v * v).sum::<f64>().sqrt();
            assert!((norm - 1.0).abs() < 1e-8, "norm = {norm}");
        }
    }
}
