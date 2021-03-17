{ sources ? import ../nix/sources.nix,
  pkgs ? import sources.nixpkgs {} }:

pkgs.mkShell {
  buildInputs = [
    (pkgs.texlive.combine {
      inherit (pkgs.texlive) scheme-minimal
      # LaTeX packages
      collection-langenglish
      collection-fontsextra
      collection-fontsrecommended
      collection-latex
      collection-latexrecommended
      collection-latexextra
      metafont # mf commandline util for fonts
      # Build tools
      arara
      ;
    })
    pkgs.glibcLocales
  ];
}
