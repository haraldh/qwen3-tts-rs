//! Minimal test: does the Burn WGPU (SPIR-V/Vulkan) backend initialize and run?
//!
//! Requires the `wgpu` feature: cargo run --features wgpu --bin vulkan_test

#[cfg(not(feature = "wgpu"))]
fn main() {
    eprintln!("This binary requires --features wgpu");
    std::process::exit(1);
}

#[cfg(feature = "wgpu")]
use burn::backend::wgpu::{Wgpu, WgpuDevice};
#[cfg(feature = "wgpu")]
use burn::tensor::Tensor;
#[cfg(feature = "wgpu")]
use std::time::Instant;

#[cfg(feature = "wgpu")]
fn main() {
    println!("Step 1: Creating device handle...");
    let t = Instant::now();
    let device = WgpuDevice::DefaultDevice;
    println!("  Done in {:?}", t.elapsed());

    println!("Step 2: Creating a small tensor (triggers runtime init)...");
    let t = Instant::now();
    let a: Tensor<Wgpu, 2> = Tensor::ones([4, 4], &device);
    println!("  Tensor created in {:?}", t.elapsed());

    println!("Step 3: Simple add...");
    let t = Instant::now();
    let b: Tensor<Wgpu, 2> = Tensor::ones([4, 4], &device);
    let c = a + b;
    let data = c.to_data();
    println!("  Add completed in {:?}", t.elapsed());
    println!("  Result[0,0] = {:?}", data.as_slice::<f32>().unwrap()[0]);

    println!("Step 4: Small matmul...");
    let t = Instant::now();
    let x: Tensor<Wgpu, 2> = Tensor::ones([32, 64], &device);
    let y: Tensor<Wgpu, 2> = Tensor::ones([64, 16], &device);
    let z = x.matmul(y);
    let data = z.to_data();
    println!("  Matmul completed in {:?}", t.elapsed());
    println!("  Result[0,0] = {:?} (expected 64.0)", data.as_slice::<f32>().unwrap()[0]);

    println!("All steps passed!");
}
