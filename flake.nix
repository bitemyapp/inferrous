{
  description = "inferrous: pinned Rust, CUDA, and local GPU validation tools";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    { nixpkgs, rust-overlay, ... }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        overlays = [ rust-overlay.overlays.default ];
        config.allowUnfree = true;
      };
      rust = pkgs.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml;
      llvm = pkgs.llvmPackages_22;
      # The pinned cuda-bindings requires CUDA >= 13.0, including curand.h.
      cuda = pkgs.symlinkJoin {
        name = "inferrous-cuda";
        paths = with pkgs.cudaPackages_13; [
          cuda_nvcc
          cuda_cudart
          cuda_crt
          cccl
          libnvvm
          libnvjitlink.lib
          libcurand.include
        ];
        postBuild = ''
          # The CUDA compiler loaders expect the conventional toolkit layout.
          ln -s lib "$out/lib64"
        '';
      };
      python = pkgs.python3.withPackages (
        ps: with ps; [
          numpy
          safetensors
          tokenizers
          huggingface-hub
          # Preparation and independent reference fixtures run on the CPU.
          torch
        ]
      );
    in
    {
      formatter.${system} = pkgs.nixfmt;
      devShells.${system}.default = pkgs.mkShell {
        packages = [
          rust
          cuda
          llvm.clang
          llvm.libclang
          python
          pkgs.git
          pkgs.shellcheck
        ];
        # Transitive native dependencies when linking the rustc backend dylib.
        buildInputs = with pkgs; [
          libffi
          libxml2
          zstd
          zlib
        ];
        CUDA_HOME = cuda;
        CUDA_PATH = cuda;
        CUDA_TOOLKIT_PATH = cuda;
        LIBCLANG_PATH = "${llvm.libclang.lib}/lib";
        # Also selects the Nix toolchain for any rustup proxy on the host PATH.
        RUSTUP_TOOLCHAIN = rust;
        shellHook = ''
          export LD_LIBRARY_PATH="/run/opengl-driver/lib:${cuda}/lib:${cuda}/nvvm/lib64''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        '';
      };
    };
}
