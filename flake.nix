{
  description = "TrustCheck: Automated Clinical AI Stress-Testing & Hospital Deployment Readiness Platform";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
          config.cudaSupport = true;
        };

        libPath = pkgs.lib.makeLibraryPath [
          pkgs.stdenv.cc.cc.lib
          pkgs.zlib
          pkgs.glib
          pkgs.libGL
          pkgs.libx11
          pkgs.libxext
          pkgs.libxrender
          pkgs.cudatoolkit
        ];
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            pkgs.python311
            pkgs.uv
            pkgs.nodejs
            pkgs.cudatoolkit
            pkgs.git
          ];

          shellHook = ''
            export VIRTUAL_ENV="/home/lev/pro/drishya/.venv"
            export PATH="$VIRTUAL_ENV/bin:$PATH"
            export LD_LIBRARY_PATH="${libPath}:/run/opengl-driver/lib:$LD_LIBRARY_PATH"
            export CUDA_PATH="${pkgs.cudatoolkit}"
            echo "🛡️ TrustCheck Clinical AI Safety Evaluation Harness Loaded"
            echo "Python: $(python3 --version 2>/dev/null) | Node: $(node --version 2>/dev/null)"
          '';
        };
      }
    );
}
