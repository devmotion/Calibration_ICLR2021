# Slides

This folder contains the LaTeX source code of the slides for
the video presentation of the paper
["Calibration tests beyond classification"](https://openreview.net/forum?id=-bxf89v3Nx)
by [David Widmann](http://www.it.uu.se/katalog/davwi492),
[Fredrik Lindsten](https://liu.se/en/employee/freli29), and
[Dave Zachariah](https://www.it.uu.se/katalog/davza513) at
[ICLR 2021](https://iclr.cc/Conferences/2021).

## Compilation

### Run experiment (optional)

You can run the experiment `friedman` to regenerate the data, if desired. To do so, please
follow [the instructions in the `experiments` folder](../experiments/README.md).

### Install requirements

The paper was compiled with TeXLive 2020. You can follow the
[official installation instructions](https://www.tug.org/texlive/acquire-netinstall.html).
Additionally, [`curl`](https://curl.se/), [`GNU awk`](https://www.gnu.org/software/gawk/manual/gawk.html), and
[`sort`](https://www.gnu.org/software/coreutils/manual/html_node/sort-invocation.html#sort-invocation)
have to be installed.

For increased reproducibility, we also provide a `nix` environment with a pinned software setup:
1. Install [nix](https://github.com/NixOS/nix#installation).
2. Navigate to this folder and activate the environment by running
   ```shell
   nix-shell
   ```
   in a terminal. Alternatively, if you use [lorri](https://github.com/target/lorri), you can
   activate the environment by executing
   ```shell
   direnv allow
   ```
   in a terminal.

### Create PDF

Finally, inside this folder run
```shell
arara --verbose main
```
in a terminal.
