{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    naersk = {
      url = "github:nix-community/naersk";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, flake-utils, naersk, rust-overlay }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [ (import rust-overlay) ];
        };
        naersk' = pkgs.callPackage naersk {};

        # Python for PyO3
        python = pkgs.python314;

        # Rust toolchain with rust-src for IDE support
        rustToolchain = pkgs.rust-bin.stable.latest.default.override {
          extensions = [ "rust-src" "rust-analyzer" "clippy" ];
        };

        # Common build inputs for rig-rlm
        commonBuildInputs = {
          nativeBuildInputs = with pkgs; [ pkg-config protobuf ];
          buildInputs = with pkgs; [ openssl python ];
          OPENSSL_NO_VENDOR = "1";
          OPENSSL_DIR = "${pkgs.openssl.dev}";
          OPENSSL_LIB_DIR = "${pkgs.openssl.out}/lib";
          OPENSSL_INCLUDE_DIR = "${pkgs.openssl.dev}/include";
          PKG_CONFIG_PATH = "${pkgs.openssl.dev}/lib/pkgconfig";
          PYO3_PYTHON = "${python}/bin/python3";
        };
      in {
        # ── rig-rlm packages ───────────────────────────────────────

        packages.default = naersk'.buildPackage (commonBuildInputs // {
          src = ./.;
        });

        packages.rig-rlm = naersk'.buildPackage (commonBuildInputs // {
          src = ./.;
          cargoBuildOptions = x: x ++ [ "--bin" "rig-rlm" ];
        });

        packages.a2a-server = naersk'.buildPackage (commonBuildInputs // {
          src = ./.;
          cargoBuildOptions = x: x ++ [ "--bin" "a2a-server" ];
        });

        packages.mcp-server = naersk'.buildPackage (commonBuildInputs // {
          src = ./.;
          cargoBuildOptions = x: x ++ [ "--bin" "mcp-server" ];
        });

        packages.restate-server = naersk'.buildPackage (commonBuildInputs // {
          src = ./.;
          cargoBuildOptions = x: x ++ [ "--bin" "restate-server" ];
        });

        # ── Dev shell (rig-rlm + agentgateway) ─────────────────────

        devShells.default = pkgs.mkShell {
          nativeBuildInputs = with pkgs; [
            rustToolchain
            pkg-config
            openssl
            sccache
            wild  # Fast linker for iterative development
            clang # Required for wild linker integration
            lld   # Fallback linker

            # Python (PyO3)
            python

            # Protobuf (agentgateway uses prost-build / tonic codegen)
            protobuf
            buf   # agentgateway uses buf for proto API generation

            # Node.js (MCP inspector, npx tools)
            nodejs_22
          ];

          OPENSSL_NO_VENDOR = "1";
          OPENSSL_DIR = "${pkgs.openssl.dev}";
          OPENSSL_LIB_DIR = "${pkgs.openssl.out}/lib";
          OPENSSL_INCLUDE_DIR = "${pkgs.openssl.dev}/include";
          PKG_CONFIG_PATH = "${pkgs.openssl.dev}/lib/pkgconfig";

          # PyO3 configuration
          PYO3_PYTHON = "${python}/bin/python3";

          # Configure Wild linker for faster builds
          CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER = "clang";
          CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUSTFLAGS = "-C link-arg=-fuse-ld=wild";

          shellHook = ''
            DEVENV_CACHE="$HOME/.cache/rig-rlm-devenv"
            export SCCACHE_DIR="$HOME/.cache/sccache"
            export RUSTC_WRAPPER="sccache"
            mkdir -p "$SCCACHE_DIR"
            chmod 700 "$SCCACHE_DIR"

            # Ensure Python shared library is on LD_LIBRARY_PATH for PyO3
            export LD_LIBRARY_PATH="${python}/lib:$LD_LIBRARY_PATH"

            # Agentgateway path alias
            export AGENTGATEWAY_DIR="$HOME/dev-stuff/agentgateway"

            if [ ! -d "$DEVENV_CACHE/profile" ]; then
              mkdir -p "$DEVENV_CACHE/profile"/{bin,lib/rustlib/src}
              ln -sfn ${rustToolchain}/bin/* "$DEVENV_CACHE/profile/bin/"
              ln -sfn ${rustToolchain}/lib/rustlib/src/rust "$DEVENV_CACHE/profile/lib/rustlib/src/rust"

              echo ""
              echo "🦀 rig-rlm development environment initialized"
              echo "   Toolchain: $DEVENV_CACHE/profile/bin"
              echo "   Python:    ${python}/bin/python3"
              echo ""
              echo "🌐 agentgateway support:"
              echo "   cd $AGENTGATEWAY_DIR && cargo build --release"
              echo "   cd $AGENTGATEWAY_DIR && cargo test --all-targets"
              echo ""
            fi
          '';
        };
      }
    );
}
