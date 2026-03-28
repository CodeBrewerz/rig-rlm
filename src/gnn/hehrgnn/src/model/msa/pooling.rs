//! Chunk-wise Mean Pooling for KV Cache Compression.
//!
//! MSA Paper §3.2.1: "We segment each document into multiple fixed-length chunks
//! and perform chunk-wise mean pooling, denoted as φ(·), to compress these states
//! into latent representations."
//!
//! This yields compressed matrices: K̄ = φ(K), V̄ = φ(V), K̄^R = φ(K^R)
//! With a default chunk size P=64, this achieves 64× compression.

use burn::prelude::*;

/// Perform chunk-wise mean pooling on a 2D tensor.
///
/// Splits the sequence dimension into chunks of size `chunk_size` and
/// averages within each chunk. If the sequence length is not evenly
/// divisible, the last chunk covers the remaining tokens.
///
/// # Arguments
/// * `input` - Tensor of shape [seq_len, dim]
/// * `chunk_size` - Number of tokens per chunk (P=64 in paper)
///
/// # Returns
/// * Compressed tensor of shape [num_chunks, dim]
pub fn chunk_mean_pool<B: Backend>(input: Tensor<B, 2>, chunk_size: usize) -> Tensor<B, 2> {
    let [seq_len, dim] = input.dims();

    if seq_len == 0 {
        return input;
    }

    if seq_len <= chunk_size {
        // Single chunk: mean of entire sequence
        return input.mean_dim(0); // [1, dim]
    }

    let num_full_chunks = seq_len / chunk_size;
    let remainder = seq_len % chunk_size;
    let num_chunks = if remainder > 0 {
        num_full_chunks + 1
    } else {
        num_full_chunks
    };

    let device = input.device();
    let mut chunk_means: Vec<Tensor<B, 2>> = Vec::with_capacity(num_chunks);

    for i in 0..num_full_chunks {
        let start = i * chunk_size;
        let end = start + chunk_size;
        let chunk = input.clone().slice([start..end, 0..dim]); // [chunk_size, dim]
        chunk_means.push(chunk.mean_dim(0)); // [1, dim]
    }

    // Handle remainder chunk
    if remainder > 0 {
        let start = num_full_chunks * chunk_size;
        let chunk = input.clone().slice([start..seq_len, 0..dim]); // [remainder, dim]
        chunk_means.push(chunk.mean_dim(0)); // [1, dim]
    }

    Tensor::cat(chunk_means, 0) // [num_chunks, dim]
}

/// Perform chunk-wise mean pooling on a 3D tensor (multi-head).
///
/// # Arguments
/// * `input` - Tensor of shape [seq_len, num_heads, head_dim]
/// * `chunk_size` - Number of tokens per chunk
///
/// # Returns
/// * Compressed tensor [num_chunks, num_heads, head_dim]
pub fn chunk_mean_pool_3d<B: Backend>(input: Tensor<B, 3>, chunk_size: usize) -> Tensor<B, 3> {
    let [seq_len, num_heads, head_dim] = input.dims();

    if seq_len == 0 || seq_len <= chunk_size {
        return input.mean_dim(0); // [1, num_heads, head_dim]
    }

    // Reshape to [seq_len, num_heads * head_dim], pool, then reshape back
    let flat = input.reshape([seq_len, num_heads * head_dim]);
    let pooled = chunk_mean_pool(flat, chunk_size);
    let [num_chunks, _] = pooled.dims();
    pooled.reshape([num_chunks, num_heads, head_dim])
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray;

    #[test]
    fn test_chunk_mean_pool_exact_division() {
        let device = <B as Backend>::Device::default();
        // 128 tokens, chunk_size=64 → 2 chunks
        let input = Tensor::<B, 2>::ones([128, 32], &device);
        let pooled = chunk_mean_pool(input, 64);
        assert_eq!(pooled.dims(), [2, 32]);
        println!("✅ Exact division: [128, 32] → [2, 32] with P=64");
    }

    #[test]
    fn test_chunk_mean_pool_with_remainder() {
        let device = <B as Backend>::Device::default();
        // 150 tokens, chunk_size=64 → 3 chunks (64+64+22)
        let input = Tensor::<B, 2>::ones([150, 16], &device);
        let pooled = chunk_mean_pool(input, 64);
        assert_eq!(pooled.dims(), [3, 16]);
        println!("✅ Remainder: [150, 16] → [3, 16] with P=64");
    }

    #[test]
    fn test_chunk_mean_pool_small_input() {
        let device = <B as Backend>::Device::default();
        // 30 tokens < chunk_size=64 → 1 chunk
        let input = Tensor::<B, 2>::ones([30, 8], &device);
        let pooled = chunk_mean_pool(input, 64);
        assert_eq!(pooled.dims(), [1, 8]);
        println!("✅ Small input: [30, 8] → [1, 8] with P=64");
    }

    #[test]
    fn test_chunk_mean_pool_values() {
        let device = <B as Backend>::Device::default();
        // 4 tokens, chunk_size=2 → 2 chunks
        // [[1,1], [3,3], [5,5], [7,7]] → mean([1,3])=[2,2], mean([5,7])=[6,6]
        let data = vec![1.0f32, 1.0, 3.0, 3.0, 5.0, 5.0, 7.0, 7.0];
        let input = Tensor::<B, 1>::from_data(data.as_slice(), &device).reshape([4, 2]);
        let pooled = chunk_mean_pool(input, 2);
        assert_eq!(pooled.dims(), [2, 2]);

        let result: Vec<f32> = pooled.into_data().as_slice::<f32>().unwrap().to_vec();
        assert!((result[0] - 2.0).abs() < 1e-5);
        assert!((result[1] - 2.0).abs() < 1e-5);
        assert!((result[2] - 6.0).abs() < 1e-5);
        assert!((result[3] - 6.0).abs() < 1e-5);
        println!("✅ Value correctness verified for chunk mean pooling");
    }

    #[test]
    fn test_chunk_mean_pool_3d() {
        let device = <B as Backend>::Device::default();
        // 128 tokens, 4 heads, 16 dim per head → 2 chunks
        let input = Tensor::<B, 3>::ones([128, 4, 16], &device);
        let pooled = chunk_mean_pool_3d(input, 64);
        assert_eq!(pooled.dims(), [2, 4, 16]);
        println!("✅ 3D pool: [128, 4, 16] → [2, 4, 16] with P=64");
    }
}
