//! Raw HIP/hipBLAS code predictor bypass.
//!
//! Replaces CubeCL kernel dispatch with direct HIP API calls for the
//! code predictor's autoregressive loop. Reduces kernel launch overhead
//! from ~7071 dispatches/frame to ~1200 by using custom fused kernels.

pub mod code_predictor;
mod kernels;
