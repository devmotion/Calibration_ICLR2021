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
      collection-luatex
      collection-latexrecommended
      collection-latexextra
      siunitx
      biblatex
      metafont # mf commandline util for fonts
      metapost
      # Build tools
      arara
      biber
      ;
    })
    pkgs.curl
    pkgs.gawk
    pkgs.glibcLocales
  ];
}
