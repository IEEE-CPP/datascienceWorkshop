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
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        buildInputs = with pkgs; [
          zsh
          basedpyright
          (python313FreeThreading.withPackages (p: [
            p.jupyter
            p.jupytext
            p.ipython
            p.pandas
            p.matplotlib
            p.seaborn
          ]))
        ];
        shellHook = ''
          zsh
          exit
        '';
      };
    };
}
