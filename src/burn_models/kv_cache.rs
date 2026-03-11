//! KV cache for autoregressive generation (Burn).
//!
//! Concat-based implementation that works on all backends.
//! Pre-allocated in-place cache can be added later via CubeCL if profiling
//! shows concat is a bottleneck.

use burn::prelude::*;

/// Concat-based KV cache for efficient autoregressive generation.
///
/// Stores key and value tensors of shape `[batch, num_heads, seq_len, head_dim]`
/// and grows by concatenation along the sequence dimension (dim 2).
pub struct KVCache<B: Backend> {
    k: Option<Tensor<B, 4>>,
    v: Option<Tensor<B, 4>>,
}

impl<B: Backend> Default for KVCache<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> KVCache<B> {
    pub fn new() -> Self {
        Self { k: None, v: None }
    }

    /// Append new K values along the sequence dimension and return the full K.
    pub fn update_k(&mut self, k: Tensor<B, 4>) -> Tensor<B, 4> {
        let k = if let Some(prev_k) = self.k.take() {
            Tensor::cat(vec![prev_k, k], 2)
        } else {
            k
        };
        self.k = Some(k.clone());
        k
    }

    /// Append new V values along the sequence dimension and return the full V.
    pub fn update_v(&mut self, v: Tensor<B, 4>) -> Tensor<B, 4> {
        let v = if let Some(prev_v) = self.v.take() {
            Tensor::cat(vec![prev_v, v], 2)
        } else {
            v
        };
        self.v = Some(v.clone());
        v
    }

    /// Update both K and V and return the full (K, V).
    pub fn update(&mut self, k: Tensor<B, 4>, v: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let k = self.update_k(k);
        let v = self.update_v(v);
        (k, v)
    }

    /// Reset the cache (e.g. between generation sessions).
    pub fn reset(&mut self) {
        self.k = None;
        self.v = None;
    }

    /// Current cached sequence length, or 0 if empty.
    pub fn seq_len(&self) -> usize {
        self.k.as_ref().map(|k| k.dims()[2]).unwrap_or(0)
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.k.is_none()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray;

    #[test]
    fn test_kv_cache_new() {
        let cache = KVCache::<B>::new();
        assert!(cache.is_empty());
        assert_eq!(cache.seq_len(), 0);
    }

    #[test]
    fn test_kv_cache_update() {
        let device = Default::default();
        let mut cache = KVCache::<B>::new();

        let k1 = Tensor::<B, 4>::zeros([1, 2, 4, 16], &device);
        let k_out = cache.update_k(k1);
        assert_eq!(k_out.dims(), [1, 2, 4, 16]);
        assert_eq!(cache.seq_len(), 4);

        let k2 = Tensor::<B, 4>::zeros([1, 2, 3, 16], &device);
        let k_out = cache.update_k(k2);
        assert_eq!(k_out.dims(), [1, 2, 7, 16]); // 4 + 3 = 7
        assert_eq!(cache.seq_len(), 7);
    }

    #[test]
    fn test_kv_cache_update_both() {
        let device = Default::default();
        let mut cache = KVCache::<B>::new();

        let k = Tensor::<B, 4>::zeros([1, 2, 5, 16], &device);
        let v = Tensor::<B, 4>::zeros([1, 2, 5, 16], &device);
        let (k_out, v_out) = cache.update(k, v);
        assert_eq!(k_out.dims(), [1, 2, 5, 16]);
        assert_eq!(v_out.dims(), [1, 2, 5, 16]);
    }

    #[test]
    fn test_kv_cache_reset() {
        let device = Default::default();
        let mut cache = KVCache::<B>::new();

        let k = Tensor::<B, 4>::zeros([1, 2, 4, 16], &device);
        cache.update_k(k);
        assert!(!cache.is_empty());

        cache.reset();
        assert!(cache.is_empty());
        assert_eq!(cache.seq_len(), 0);
    }
}
