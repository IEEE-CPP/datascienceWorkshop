{
  description = "A very basic flake";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
  };

  outputs =
    { self, nixpkgs }@inputs:
    let
      system = "x86_64-linux";
      pkgs = inputs.nixpkgs.legacyPackages.${system};
      env =
        script:
        (pkgs.buildFHSEnv {
          name = "python-env";
          targetPkgs =
            pkgs:
            (with pkgs; [
              python313
              python313Packages.pip
              python313Packages.virtualenv
              # Support binary wheels from PyPI
              pythonManylinuxPackages.manylinux2014Package
              # Enable building from sdists
              cmake
              ninja
              gcc
              pre-commit
            ]);
          runScript = "${
            pkgs.writeShellScriptBin "runScript" (
              ''
                    set -e
                    test -d env || ${pkgs.python313.interpreter} -m venv env
                source env/bin/activate
                set +e
              ''
              + script
            )
          }/bin/runScript";
        }).env;
    in
    {
      devShells.${system}.default = env "zsh";
    };
}
