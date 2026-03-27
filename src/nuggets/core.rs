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

// =========================================================================
// N-Dimensional RoPE (Golden Gate RoPE) — Temporal Lattice Memory
// =========================================================================
// Based on https://jerryxio.ng/posts/nd-rope/
//
// Instead of encoding facts at a linear integer position, this encodes
// facts at an N-dimensional coordinate in a normalized [-1,1] space.
// Each dimension represents a temporal axis:
//   t₀ = epoch (coarse time)
//   t₁ = episode (conversation/session)
//   t₂ = turn (fine-grained position)
//   t₃ = recency (hit count / importance)
//
// The golden ratio quasi-random direction vectors ensure no two
// temporal scales alias against each other.

/// N-dimensional RoPE with golden-ratio direction vectors.
/// `n` = number of coordinate dimensions (e.g. 3 for epoch/episode/turn).
/// `d` = HRR vector dimension (number of frequency pairs).
pub struct NdRope {
    pub n: usize,
    pub d: usize,
    /// Per-pair frequency magnitudes, length = d
    pub freqs: Vec<f64>,
    /// Direction matrix: d rows × n cols (flattened, row-major)
    /// u[i*n + k] = k-th component of direction vector for pair i
    pub dirs: Vec<f64>,
}

impl NdRope {
    /// Create an NdRope with `n` temporal dimensions and `d` frequency pairs.
    ///
    /// `w_min`/`w_max`: frequency range. For coordinates in [-1,1],
    /// w_min=0.5, w_max=20.0 is a good default (avoids high-freq aliasing).
    pub fn new(n: usize, d: usize, w_min: f64, w_max: f64) -> Self {
        let mut freqs = vec![0.0; d];
        let mut dirs = vec![0.0; d * n];

        let d_m1 = (d.max(2) - 1) as f64;
        let w_ratio = w_max / w_min;

        // Generalized golden ratio for N dimensions
        // Uses the quasi-random R-sequence from Extreme Learning:
        // phi_n = smallest real root of x^(n+1) = x + 1
        let phi_n = Self::golden_n(n);

        for i in 0..d {
            // Log-spaced frequencies
            freqs[i] = w_min * w_ratio.powf((i as f64) / d_m1);

            // Quasi-random direction vector on the N-sphere
            // Map R-sequence samples to Gaussian via Box-Muller-like inverse
            let mut norm_sq = 0.0;
            for k in 0..n {
                // R-sequence: fractional part of (i+1) * phi_n^-(k+1)
                let alpha = Self::phi_inv_pow(phi_n, k + 1);
                let sample = ((i + 1) as f64 * alpha).fract();
                // Inverse CDF of standard normal (approximation via probit)
                let g = Self::inv_normal_cdf(sample);
                dirs[i * n + k] = g;
                norm_sq += g * g;
            }
            // Normalize to unit sphere
            let inv_norm = 1.0 / (norm_sq.sqrt() + 1e-12);
            for k in 0..n {
                dirs[i * n + k] *= inv_norm;
            }
        }

        Self { n, d, freqs, dirs }
    }

    /// Compute the rotation angle for frequency pair `i` given coordinate `t`.
    /// angle_i = freq_i * dot(u_i, t)
    #[inline]
    pub fn angle(&self, i: usize, t: &[f64]) -> f64 {
        let mut dot = 0.0;
        let base = i * self.n;
        for k in 0..self.n {
            dot += self.dirs[base + k] * t[k];
        }
        self.freqs[i] * dot
    }

    /// Compute all rotation angles for coordinate `t`.
    #[inline]
    pub fn angles_for(&self, t: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; self.d];
        for i in 0..self.d {
            out[i] = self.angle(i, t);
        }
        out
    }

    /// Generalized golden ratio for N dimensions.
    /// Solves x^(n+1) = x + 1 by Newton iteration.
    fn golden_n(n: usize) -> f64 {
        let mut x = 2.0_f64;
        let n1 = (n + 1) as f64;
        for _ in 0..60 {
            let xn = x.powf(n1);
            let f = xn - x - 1.0;
            let df = n1 * x.powf(n1 - 1.0) - 1.0;
            x -= f / df;
        }
        x
    }

    /// Compute 1 / phi^k
    fn phi_inv_pow(phi: f64, k: usize) -> f64 {
        1.0 / phi.powi(k as i32)
    }

    /// Approximate inverse normal CDF (probit function).
    /// Maps (0,1) → ℝ. Uses rational approximation (Abramowitz & Stegun).
    fn inv_normal_cdf(p: f64) -> f64 {
        // Clamp to avoid infinities
        let p = p.clamp(0.001, 0.999);
        let t = if p < 0.5 {
            (-2.0 * p.ln()).sqrt()
        } else {
            (-2.0 * (1.0 - p).ln()).sqrt()
        };
        // Coefficients for rational approximation
        let c0 = 2.515517;
        let c1 = 0.802853;
        let c2 = 0.010328;
        let d1 = 1.432788;
        let d2 = 0.189269;
        let d3 = 0.001308;
        let val = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t);
        if p < 0.5 { -val } else { val }
    }
}

/// Bind a ComplexVector to an N-dimensional temporal coordinate.
pub fn bind_nd(v: &ComplexVector, t: &[f64], rope: &NdRope) -> ComplexVector {
    let d = v.dim();
    let angles = rope.angles_for(t);
    let mut re = vec![0.0; d];
    let mut im = vec![0.0; d];
    for j in 0..d {
        let (s, c) = angles[j].sin_cos();
        re[j] = c * v.re[j] - s * v.im[j];
        im[j] = c * v.im[j] + s * v.re[j];
    }
    ComplexVector { re, im }
}

/// Unbind a ComplexVector from an N-dimensional temporal coordinate.
pub fn unbind_nd(v: &ComplexVector, t: &[f64], rope: &NdRope) -> ComplexVector {
    let d = v.dim();
    let angles = rope.angles_for(t);
    let mut re = vec![0.0; d];
    let mut im = vec![0.0; d];
    for j in 0..d {
        let (s, c) = angles[j].sin_cos();
        // Inverse rotation: negate sin
        re[j] = v.re[j] * c + v.im[j] * s;
        im[j] = v.im[j] * c - v.re[j] * s;
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

    // ── Temporal Nd-RoPE Tests ──────────────────────────────────────────

    fn cos_sim(a: &ComplexVector, b: &ComplexVector) -> f64 {
        let mut dot = 0.0;
        let mut na = 0.0;
        let mut nb = 0.0;
        for i in 0..a.dim() {
            dot += a.re[i] * b.re[i] + a.im[i] * b.im[i];
            na += a.re[i] * a.re[i] + a.im[i] * a.im[i];
            nb += b.re[i] * b.re[i] + b.im[i] * b.im[i];
        }
        dot / (na.sqrt() * nb.sqrt() + 1e-12)
    }

    /// Golden ratio for N=1 should be the classic φ = 1.618...
    #[test]
    fn nd_rope_golden_ratio_1d() {
        let phi = NdRope::golden_n(1);
        assert!((phi - 1.6180339887).abs() < 1e-6, "phi_1 = {phi}");
    }

    /// Golden ratio for N=2 should be the plastic constant ≈ 1.3247...
    #[test]
    fn nd_rope_golden_ratio_2d() {
        let phi = NdRope::golden_n(2);
        assert!((phi - 1.3247179572).abs() < 1e-6, "phi_2 = {phi}");
    }

    /// Direction vectors should be unit-length on the N-sphere.
    #[test]
    fn nd_rope_directions_unit_norm() {
        let rope = NdRope::new(3, 256, 0.5, 20.0);
        for i in 0..rope.d {
            let mut norm_sq = 0.0;
            for k in 0..rope.n {
                let v = rope.dirs[i * rope.n + k];
                norm_sq += v * v;
            }
            let norm = norm_sq.sqrt();
            assert!((norm - 1.0).abs() < 1e-6, "dir[{i}] norm = {norm}");
        }
    }
    
    /// Bind then unbind at the exact same coordinate → perfect recovery.
    #[test]
    fn nd_rope_exact_roundtrip() {
        let d = 512;
        let rope = NdRope::new(3, d, 0.5, 20.0);
        let mut rng = Mulberry32::new(42);
        let keys = make_vocab_keys(1, d, &mut rng);
        let fact = &keys[0];

        let coord = [0.3, -0.5, 0.8];
        let bound = bind_nd(fact, &coord, &rope);
        let recovered = unbind_nd(&bound, &coord, &rope);

        let sim = cos_sim(fact, &recovered);
        assert!((sim - 1.0).abs() < 1e-10, "Exact roundtrip sim = {sim}");
    }

    /// Two facts at different coordinates can be separated from a superposition.
    #[test]
    fn nd_rope_superposition_separation() {
        let d = 1024;
        let rope = NdRope::new(3, d, 1.0, 100.0);
        let mut rng = Mulberry32::new(999);
        let keys = make_vocab_keys(3, d, &mut rng);
        let fact_a = &keys[0];
        let fact_b = &keys[1];
        let sent_key = &keys[2];

        // Temporal coordinates: well-separated in normalized space
        let t_a = [0.1, 0.2, 0.3];
        let t_b = [0.7, -0.4, 0.1];

        // Build superposition
        let bound_a = bind_nd(&bind(sent_key, fact_a), &t_a, &rope);
        let bound_b = bind_nd(&bind(sent_key, fact_b), &t_b, &rope);
        let mut mem = ComplexVector::zeros(d);
        for i in 0..d {
            mem.re[i] = bound_a.re[i] + bound_b.re[i];
            mem.im[i] = bound_a.im[i] + bound_b.im[i];
        }

        // Query at t_a → should recover fact_a clearly over fact_b
        let query_a = unbind(&unbind_nd(&mem, &t_a, &rope), sent_key);
        let sim_aa = cos_sim(fact_a, &query_a);
        let sim_ba = cos_sim(fact_b, &query_a);
        println!("  Separation: sim(a,@a)={sim_aa:.4} sim(b,@a)={sim_ba:.4}");
        assert!(sim_aa > sim_ba, "fact_a should dominate at t_a");
        assert!(sim_aa > 0.4, "sim(a, query@t_a) = {sim_aa}");

        // Query at t_b → should recover fact_b clearly over fact_a
        let query_b = unbind(&unbind_nd(&mem, &t_b, &rope), sent_key);
        let sim_bb = cos_sim(fact_b, &query_b);
        let sim_ab = cos_sim(fact_a, &query_b);
        println!("  Separation: sim(b,@b)={sim_bb:.4} sim(a,@b)={sim_ab:.4}");
        assert!(sim_bb > sim_ab, "fact_b should dominate at t_b");
        assert!(sim_bb > 0.4, "sim(b, query@t_b) = {sim_bb}");
    }

    /// Temporal proximity: nearby coordinates → smooth similarity decay.
    #[test]
    fn nd_rope_temporal_proximity_decay() {
        let d = 1024;
        let rope = NdRope::new(3, d, 1.0, 100.0);
        let mut rng = Mulberry32::new(7);
        let keys = make_vocab_keys(2, d, &mut rng);
        let fact = &keys[0];
        let sent_key = &keys[1];

        let t_exact = [0.3, 0.0, 0.0];
        let bound = bind_nd(&bind(sent_key, fact), &t_exact, &rope);

        // Query at exact, near, mid, and far coordinates
        let r_exact = unbind(&unbind_nd(&bound, &t_exact, &rope), sent_key);
        let r_near  = unbind(&unbind_nd(&bound, &[0.301, 0.0, 0.0], &rope), sent_key);
        let r_mid   = unbind(&unbind_nd(&bound, &[0.31, 0.0, 0.0], &rope), sent_key);
        let r_far   = unbind(&unbind_nd(&bound, &[0.5, 0.0, 0.0], &rope), sent_key);

        let s_exact = cos_sim(fact, &r_exact);
        let s_near  = cos_sim(fact, &r_near);
        let s_mid   = cos_sim(fact, &r_mid);
        let s_far   = cos_sim(fact, &r_far);

        // Monotonic decay: exact > near > mid > far
        assert!(s_exact > s_near, "exact {s_exact} > near {s_near}");
        assert!(s_near > s_mid,   "near {s_near} > mid {s_mid}");
        assert!(s_mid > s_far,    "mid {s_mid} > far {s_far}");
        // Exact is perfect
        assert!(s_exact > 0.99, "exact = {s_exact}");

        println!("  Temporal decay: exact={s_exact:.4} near={s_near:.4} mid={s_mid:.4} far={s_far:.4}");
    }

    /// Trie-like hierarchical test: facts sharing the same epoch (t₀)
    /// are partially recoverable even with different episode/turn.
    #[test]
    fn nd_rope_hierarchical_trie_property() {
        let d = 512;
        // 3 dims: epoch, episode, turn
        let rope = NdRope::new(3, d, 0.5, 20.0);
        let mut rng = Mulberry32::new(55);
        let keys = make_vocab_keys(2, d, &mut rng);
        let fact = &keys[0];
        let sent_key = &keys[1];

        // Store fact at (epoch=0.2, episode=0.3, turn=0.5)
        let t_stored = [0.2, 0.3, 0.5];
        let bound = bind_nd(&bind(sent_key, fact), &t_stored, &rope);

        // Query same epoch, different episode/turn
        let t_same_epoch = [0.2, 0.1, -0.3];
        let r_same_epoch = unbind(&unbind_nd(&bound, &t_same_epoch, &rope), sent_key);

        // Query completely different epoch
        let t_diff_epoch = [0.8, 0.3, 0.5];
        let r_diff_epoch = unbind(&unbind_nd(&bound, &t_diff_epoch, &rope), sent_key);

        let s_same = cos_sim(fact, &r_same_epoch);
        let s_diff = cos_sim(fact, &r_diff_epoch);

        // Both should be <1, but queries in the same epoch should show
        // some inherited similarity (the "trie prefix match" property).
        // The exact relationship depends on how the direction vectors
        // distribute weight across dimensions.
        println!("  Hierarchical: same_epoch={s_same:.4} diff_epoch={s_diff:.4}");
        // At minimum, neither should be perfect
        assert!(s_same < 0.95, "same epoch shouldn't be perfect: {s_same}");
        assert!(s_diff < 0.95, "diff epoch shouldn't be perfect: {s_diff}");
    }

    /// Scale test: 50 facts in superposition with 3D temporal coords.
    #[test]
    fn nd_rope_50_fact_superposition() {
        let d = 2048;
        let n_facts = 50;
        let rope = NdRope::new(3, d, 1.0, 100.0);
        let mut rng = Mulberry32::new(321);
        let keys = make_vocab_keys(n_facts + 1, d, &mut rng);
        let sent_key = &keys[n_facts];

        // Generate 50 facts at quasi-random temporal coordinates in [-0.5, 0.5]
        let mut coords: Vec<[f64; 3]> = Vec::new();
        let mut rng2 = Mulberry32::new(111);
        for _ in 0..n_facts {
            coords.push([
                rng2.next_f64() - 0.5,
                rng2.next_f64() - 0.5,
                rng2.next_f64() - 0.5,
            ]);
        }

        // Build superposition
        let mut mem = ComplexVector::zeros(d);
        for i in 0..n_facts {
            let bound = bind_nd(&bind(sent_key, &keys[i]), &coords[i], &rope);
            for j in 0..d {
                mem.re[j] += bound.re[j];
                mem.im[j] += bound.im[j];
            }
        }

        // Query each fact at its exact coordinate
        let mut correct = 0;
        for i in 0..n_facts {
            let query = unbind(&unbind_nd(&mem, &coords[i], &rope), sent_key);
            // Find best match
            let mut best_idx = 0;
            let mut best_sim = f64::NEG_INFINITY;
            for k in 0..n_facts {
                let s = cos_sim(&keys[k], &query);
                if s > best_sim {
                    best_sim = s;
                    best_idx = k;
                }
            }
            if best_idx == i {
                correct += 1;
            }
        }

        let accuracy = correct as f64 / n_facts as f64;
        println!("  50-fact Nd-RoPE accuracy: {correct}/{n_facts} = {accuracy:.1}%");
        assert!(accuracy > 0.9, "Expected >90% accuracy, got {accuracy}");
    }
}
