{
  description = "qwen3-tts-rs dev shell with AMD ROCm";

  inputs = {
    # Pinned to the exact revision nixcfg builds against, so the dev shell and
    # the nixcfg package (pkgs.metacfg.qwen3-tts-rs) use one ROCm. Drifting
    # apart means developing against a different HIP than we deploy: this rev
    # ships clr 7.2.3 / HIP patch 53211, whereas nixpkgs-unstable was on 7.2.0
    # / 53210, and cubecl-hip-sys selects its bindings by that patch number.
    # Bump in lockstep with nixcfg's flake.lock `nixpkgs` entry.
    unstable.url = "github:NixOS/nixpkgs/21ea275a7c46aef9d4d6ddc962e6d562e9d94183";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "unstable";
    };
  };

  outputs = { self, unstable, flake-utils, rust-overlay }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import unstable {
          inherit system;
          config.allowUnfree = true;
          config.rocmSupport = true;
        };

        rust-bin = rust-overlay.lib.mkRustBin { } pkgs;
        rustToolchain = rust-bin.stable.latest.default.override {
          extensions = [ "rust-src" "rust-analyzer" "clippy" ];
        };

        rocmPkgs = pkgs.rocmPackages;

        # openblas-src crate probes pkg-config for "openblas", but NixOS
        # ships blas.pc instead. This derivation creates a proper openblas.pc.
        openblasLib = pkgs.openblas;
        openblasDev = pkgs.openblas.dev;
        openblasPkgConfig = pkgs.writeTextDir "lib/pkgconfig/openblas.pc" ''
          Name: openblas
          Description: OpenBLAS (NixOS wrapper for openblas-src crate)
          Version: ${pkgs.openblas.version}
          Libs: -L${openblasLib}/lib -lopenblas
          Cflags: -I${openblasDev}/include
        '';
      in
      {
        devShells.default = pkgs.mkShell {
          name = "qwen3-tts-rocm";

          nativeBuildInputs = with pkgs; [
            rustToolchain
            pkg-config
            cmake
            ninja
            python3
            mold              # Fast linker
          ];

          buildInputs = with pkgs; [
            # ROCm toolchain
            rocmPkgs.clr          # HIP runtime (libamdhip64)
            rocmPkgs.hipcc        # HIP compiler driver
            rocmPkgs.rocm-core
            rocmPkgs.rocm-runtime
            rocmPkgs.rocm-device-libs
            rocmPkgs.rocm-comgr
            rocmPkgs.rocblas
            rocmPkgs.rocsolver
            rocmPkgs.rocfft
            rocmPkgs.rocrand
            rocmPkgs.hipblas
            rocmPkgs.hipsparse
            rocmPkgs.miopen
            rocmPackages.rocm-smi
            rocmPkgs.rocprofiler    # rocprof CLI for kernel tracing
            rocmPkgs.roctracer      # HIP/HSA tracing (rocprofiler dependency)

            # Vulkan
            vulkan-loader
            vulkan-headers
            vulkan-tools        # vulkaninfo
            vulkan-validation-layers

            # System libs
            openssl
            alsa-lib
            openblas.dev         # Multi-threaded BLAS for CPU code predictor
            openblasPkgConfig    # pkg-config wrapper so openblas-src can find it
          ];

          shellHook = ''
            export ROCM_PATH="${rocmPkgs.clr}"
            export HIP_PATH="${rocmPkgs.clr}"
            export HIP_DEVICE_LIB_PATH="${rocmPkgs.rocm-device-libs}/amdgcn/bitcode"

            # Vulkan: use Mesa RADV ICD from this flake's nixpkgs
            export LD_LIBRARY_PATH="${pkgs.vulkan-loader}/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
            export VK_ICD_FILENAMES="${pkgs.mesa}/share/vulkan/icd.d/radeon_icd.x86_64.json"

            # Strix Halo iGPU = gfx1151 (RDNA 3.5). The raw HIP kernels in
            # src/burn_models/hip are compiled at RUNTIME through HIPRTC, which
            # does not infer the target from the current device — without this
            # the kernels fail to compile at first synthesis, not at build time.
            # Anything running outside this shell (a systemd unit, a container)
            # must set it too.
            export HIPRTC_COMPILE_OPTIONS_APPEND="--offload-arch=gfx1151"

            # If ROCm doesn't recognize the exact GFX version, override to nearest supported:
            # export HSA_OVERRIDE_GFX_VERSION="11.5.1"

            echo "ROCm $(${rocmPkgs.rocm-core}/bin/rocm_agent_enumerator 2>/dev/null | tail -1 || echo '7.2.0') dev shell ready"
            echo "Vulkan: $(vulkaninfo --summary 2>/dev/null | grep 'deviceName' | head -1 || echo 'checking...')"
          '';

          RUST_BACKTRACE = 1;
        };
      }
    );
}
