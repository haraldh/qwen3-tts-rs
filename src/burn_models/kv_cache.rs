//! KV cache for autoregressive generation (Burn).
//!
//! Pre-allocated fixed-size buffer with `slice_assign` writes.
//! Eliminates per-step tensor allocation during decode.

use burn::prelude::*;

/// Pre-allocated KV cache for efficient autoregressive generation.
///
/// Allocates fixed-size buffers of shape `[batch, num_heads, max_seq_len, head_dim]`
/// on construction, then writes new K/V entries via `slice_assign` on dim 2.
/// Returns a narrowed view of the valid portion for attention.
pub struct KVCache<B: Backend> {
    /// Using Option to allow take() for move semantics with slice_assign.
    k: Option<Tensor<B, 4>>,
    v: Option<Tensor<B, 4>>,
    /// Number of valid positions written so far.
    len: usize,
    /// Maximum sequence length this cache can hold.
    max_seq: usize,
}

impl<B: Backend> KVCache<B> {
    /// Create a pre-allocated cache.
    ///
    /// # Arguments
    /// * `batch` — batch size (always 1 for TTS)
    /// * `num_heads` — number of KV heads (may differ from Q heads in GQA)
    /// * `max_seq` — maximum sequence length to accommodate
    /// * `head_dim` — dimension per head
    /// * `device` — target device
    pub fn new(
        batch: usize,
        num_heads: usize,
        max_seq: usize,
        head_dim: usize,
        device: &B::Device,
    ) -> Self {
        let k = Tensor::zeros([batch, num_heads, max_seq, head_dim], device);
        let v = Tensor::zeros([batch, num_heads, max_seq, head_dim], device);
        Self {
            k: Some(k),
            v: Some(v),
            len: 0,
            max_seq,
        }
    }

    /// Write new K/V entries and return the valid cached portion.
    ///
    /// `k` and `v` have shape `[batch, num_heads, new_seq, head_dim]`.
    pub fn update(&mut self, k: Tensor<B, 4>, v: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let new_seq = k.dims()[2];
        let end = self.len + new_seq;
        debug_assert!(
            end <= self.max_seq,
            "KV cache overflow: {} + {} > {}",
            self.len,
            new_seq,
            self.max_seq
        );

        // Take ownership to avoid clone — slice_assign consumes self
        let k_buf = self.k.take().unwrap();
        let v_buf = self.v.take().unwrap();

        let k_buf = k_buf.slice_assign(
            [
                0..k.dims()[0],
                0..k.dims()[1],
                self.len..end,
                0..k.dims()[3],
            ],
            k,
        );
        let v_buf = v_buf.slice_assign(
            [
                0..v.dims()[0],
                0..v.dims()[1],
                self.len..end,
                0..v.dims()[3],
            ],
            v,
        );
        self.len = end;

        // Return narrow view of valid portion
        let k_valid = k_buf.clone().narrow(2, 0, self.len);
        let v_valid = v_buf.clone().narrow(2, 0, self.len);
        self.k = Some(k_buf);
        self.v = Some(v_buf);
        (k_valid, v_valid)
    }

    /// Reset the cache for a new generation session.
    ///
    /// Does NOT reallocate — just resets the write position.
    pub fn reset(&mut self) {
        self.len = 0;
    }

    /// Current cached sequence length.
    pub fn seq_len(&self) -> usize {
        self.len
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray;

    #[test]
    fn test_kv_cache_new() {
        let device = Default::default();
        let cache = KVCache::<B>::new(1, 2, 64, 16, &device);
        assert!(cache.is_empty());
        assert_eq!(cache.seq_len(), 0);
    }

    #[test]
    fn test_kv_cache_update() {
        let device = Default::default();
        let mut cache = KVCache::<B>::new(1, 2, 64, 16, &device);

        let k1 = Tensor::<B, 4>::ones([1, 2, 4, 16], &device);
        let v1 = Tensor::<B, 4>::ones([1, 2, 4, 16], &device);
        let (k_out, v_out) = cache.update(k1, v1);
        assert_eq!(k_out.dims(), [1, 2, 4, 16]);
        assert_eq!(v_out.dims(), [1, 2, 4, 16]);
        assert_eq!(cache.seq_len(), 4);

        let k2 = Tensor::<B, 4>::ones([1, 2, 3, 16], &device);
        let v2 = Tensor::<B, 4>::ones([1, 2, 3, 16], &device);
        let (k_out, _) = cache.update(k2, v2);
        assert_eq!(k_out.dims(), [1, 2, 7, 16]); // 4 + 3 = 7
        assert_eq!(cache.seq_len(), 7);
    }

    #[test]
    fn test_kv_cache_values_preserved() {
        let device = Default::default();
        let mut cache = KVCache::<B>::new(1, 1, 64, 2, &device);

        // Write [1.0, 2.0] at position 0
        let k1 = Tensor::<B, 4>::from_floats([[[[1.0, 2.0]]]], &device);
        let v1 = Tensor::<B, 4>::from_floats([[[[3.0, 4.0]]]], &device);
        cache.update(k1, v1);

        // Write [5.0, 6.0] at position 1
        let k2 = Tensor::<B, 4>::from_floats([[[[5.0, 6.0]]]], &device);
        let v2 = Tensor::<B, 4>::from_floats([[[[7.0, 8.0]]]], &device);
        let (k_out, v_out) = cache.update(k2, v2);

        // Verify both positions are correct
        let k_data: Vec<f32> = k_out.into_data().to_vec().unwrap();
        assert_eq!(k_data, [1.0, 2.0, 5.0, 6.0]);
        let v_data: Vec<f32> = v_out.into_data().to_vec().unwrap();
        assert_eq!(v_data, [3.0, 4.0, 7.0, 8.0]);
    }

    #[test]
    fn test_kv_cache_reset() {
        let device = Default::default();
        let mut cache = KVCache::<B>::new(1, 2, 64, 16, &device);

        let k = Tensor::<B, 4>::zeros([1, 2, 4, 16], &device);
        let v = Tensor::<B, 4>::zeros([1, 2, 4, 16], &device);
        cache.update(k, v);
        assert!(!cache.is_empty());

        cache.reset();
        assert!(cache.is_empty());
        assert_eq!(cache.seq_len(), 0);
    }
}
