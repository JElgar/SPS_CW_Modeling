(TeX-add-style-hook
 "report"
 (lambda ()
   (TeX-run-style-hooks
    "latex2e"
    "article"
    "art10"
    "amsmath"
    "graphicx"
    "multicol")
   (LaTeX-add-labels
    "fig:noise1"
    "fig:noise2"
    "fig:adv3"))
 :latex)

