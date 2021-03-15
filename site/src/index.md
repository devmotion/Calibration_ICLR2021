# Home

This webpage accompanies the paper
["Calibration tests beyond classification"](https://openreview.net/forum?id=-bxf89v3Nx)
by [David Widmann](http://www.it.uu.se/katalog/davwi492),
[Fredrik Lindsten](https://liu.se/en/employee/freli29), and
[Dave Zachariah](https://www.it.uu.se/katalog/davza513), which will be presented at
[ICLR 2021](https://iclr.cc/Conferences/2021).

![](generated/figures/friedman/statsplot_zoom.svg)

The source code for the paper, the experiments therein, and also this webpage are
available on [Github](https://github.com/devmotion/Calibration_ICLR2021/).

## Abstract

```@example
using Markdown: Markdown                      #hide
function abstract(file)                       #hide
    buffer = IOBuffer()                       #hide
    isabstract = false                        #hide
    for line in eachline(file)                #hide
        if line == "\\begin{abstract}"        #hide
            isabstract = true                 #hide
        elseif line == "\\end{abstract}"      #hide
            break                             #hide
        elseif isabstract                     #hide
            println(buffer, "> ", line)       #hide
        end                                   #hide
    end                                       #hide
    return Markdown.parse(seekstart(buffer))  #hide
end                                           #hide
                                              #hide
abstract(joinpath(@__DIR__, "..", "..", "paper", "main.tex")) #hide
```
