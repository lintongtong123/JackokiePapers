:: Copyright (c) 2012-2016 hzz

@echo off

:: compile the tex file

cd exampleandimage

call compiler

cd ..

call gbtclear

xelatex.exe --synctex=-1 biblatex-solution-to-latex-bibliography.tex

biber biblatex-solution-to-latex-bibliography

xelatex.exe --synctex=-1 biblatex-solution-to-latex-bibliography.tex

xelatex.exe --synctex=-1 biblatex-solution-to-latex-bibliography.tex

call gbtclear



