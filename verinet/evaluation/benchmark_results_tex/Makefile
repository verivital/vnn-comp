TEX_NAME = results
LATEX = pdflatex
BIBTEX = bibtex
all:
	${LATEX} ${TEX_NAME}.tex
	${BIBTEX} ${TEX_NAME}
	${LATEX} ${TEX_NAME}.tex
	${LATEX} ${TEX_NAME}.tex
	rm ${TEX_NAME}.log  ${TEX_NAME}.aux ./tables/*.aux ${TEX_NAME}.blg ${TEX_NAME}.bbl
