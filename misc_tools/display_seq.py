##  Mike Phillips, 6/13/2023
##  Script for displaying sequence in a nice way.
##  - e.g. to make figures / tables for publication
##  * adapted 2/24/2025, now using 'Sequence' object
##  * extended 6/11/2025: now printing LaTeX formatting similar to image formatting (roughly)
"""
command line arguments:
    (1) direct string of aminos _or_ sequence name to load from CSV file
    (2) filename for loading sequence from file, or string flag for default file
    (3) name heading, if using CSV file
    (4) sequence heading, if using CSV file
    (5) output directory, to automatically save figure (PDF) and LaTeX formatting (TXT)
"""

import Sequence as S
import numpy as np
import matplotlib.pyplot as plt
import myPlotOptions as mpo
import os
import sys

argv = sys.argv

seq_in = 'EK'*25        # input: aminos, or sequence name
if len(argv) > 1:
    seq_in = argv.pop(1)

file_in = None          # input: CSV filename/path (if loading), or any other string to use default file
if len(argv) > 1:
    file_in = argv.pop(1)
    if file_in.lower() == 'none':
        file_in = None

name_h = 'NAME'         # input: name heading, for CSV
if len(argv) > 1:
    name_h = argv.pop(1)
seq_h = 'SEQUENCE'      # input: name heading, for CSV
if len(argv) > 1:
    seq_h = argv.pop(1)

OUTDIR = None           # input: out directory, for automatically saving PDF of sequence image, and LaTeX formatting
if len(argv) > 1:
    OUTDIR = argv.pop(1)

# characterize sequence, store aminos
if file_in and ('.csv' == file_in.lower()[-4:]):
    ob = S.Sequence(name=seq_in, file=file_in, headName=name_h, headSeq=seq_h, info=True)
elif file_in:
    ob = S.Sequence(name=seq_in, info=True)
else:
    ob = S.Sequence(name=seq_in, aminos=seq_in, info=True)
seq = ob.aminos     # grab primary sequence

#labels = True       # include numerical labels with sequence ?
labels = False
label_skip = 10     # number of characters to skip between labels (if shown)
label_end = False   # show ending label, i.e. total length of sequence ?
shift_end = 1       # extra horizontal space for ending label

# define types of aminos for special display (like color, bold, etc.) using MatPlotLib
neg = ("E", "D")
pos = ("R", "K")
special = ('C',)

neg_color = 'r'
#neg_color = "red"
#neg_color = 'orangered'
#neg_wt = "bold"
neg_wt = "normal"

pos_color = 'b'
#pos_color = "blue"
#pos_color = 'dodgerblue'
#pos_wt = "bold"
pos_wt = "normal"

special_color = "tab:green"
#special_color = 'limegreen'
special_wt = "bold"

def_color = "black"
def_wt = "normal"
#def_wt = "bold"

#def_font = "courier"
def_font = "courier new"
def_size = 14

# font dictionary for actual sequence (modified for each letter)
fdict = {"fontfamily":def_font, "size":def_size, "ha":"left", "fontweight":def_wt, "color":def_color}

# font dictionary for labels (should be static)
ldict = fdict.copy()
ldict.update( {"color":"gray", "size":10} )

pt = [0, 0.8]   # beginning point for main sequence string
pt = [0, 1]

# general figure settings
dpi = 100
width = 6.6
#height = 1.5
#height = 1.1
height = 0.7

width = 9.0     # allowing for ~60 residues per line
height = 0.18*(np.ceil(len(seq)/60))

facy = 1.2      # factor to leave a little vertical space after newline

dpi_f = fdict['size']/dpi   # fraction of inch used by each character
dx = (dpi_f) / width        # width spacing fraction (translating to figure coords.)
dy = (dpi_f*facy) / height  # height ...

newx = 0        # where to start after a newline
#newx = dx
#newy = 2.25*dy  # how much vertial space to skip after newline
newy = 2.25*dy if labels else dy
lbly = dy       # where to show label (i.e. above or below)
shifty = dy*0.9 # extra shift between lines
#xmax = 0.9      # maximum x position; begin new line if coordinate surpasses this
xmax = 0.985

fig = plt.figure(figsize=(width,height), dpi=dpi)
for i in range(len(seq)):
    s = seq[i]
    if s in neg:
        fdict.update( {"color":neg_color, "fontweight":neg_wt} )
    elif s in pos:
        fdict.update( {"color":pos_color, "fontweight":pos_wt} )
    elif s in special:
        fdict.update( {"color":special_color, "fontweight":special_wt} )
    else:
        fdict.update( {"color":def_color, "fontweight":def_wt} )
    fig.text(pt[0], pt[1]-shifty, s, fdict)
    if labels:
        if ((i % label_skip) == 0):
            fig.text(pt[0], pt[1]-shifty+lbly, str(i+1), ldict)
        if label_end and (i == (len(seq)-1)):
            fig.text(pt[0]+shift_end*dx, pt[1]-shifty+lbly, str(i+1), ldict)
        if ((i % label_skip) == (label_skip-1)):
            pt[0] += dx
    pt[0] += dx
    if pt[0] > xmax:
        pt[0] = newx
        pt[1] -= newy
if OUTDIR:
    fig.savefig(os.path.join(OUTDIR, f'{seq_in}.pdf'))
plt.show()
plt.close()


####    ####    ####    ####    ####    ####    ####    ####    ####    ####

# LaTeX formatting option for direct input, without using images
# define types of aminos for special display (like color, bold, etc.) in LaTeX
##  requires packages:
##      \usepackage[dvipsnames]{xcolor}
##      \usepackage[T1]{fontenc}

neg = ("E", "D")
pos = ("R", "K")
special = ('C',)

neg_texcolor = 'red'
#neg_texwt = "bold"
neg_texwt = "normal"

pos_texcolor = 'blue'
#pos_texwt = "bold"
pos_texwt = "normal"

special_texcolor = "ForestGreen"
special_texwt = "bold"
#special_texwt = "normal"

def_texcolor = "black"
def_texwt = "normal"
#def_texwt = "bold"

la_string = ''
for i in range(len(seq)):
    s = seq[i]
    if s in neg:
        if neg_texwt == 'bold':
            s = r'\textbf{' + s + r'}'
        fmt_s = r'{\textcolor{' + neg_texcolor + r'}{' + s + r'}}'
        wt = neg_texwt
    elif s in pos:
        if pos_texwt == 'bold':
            s = r'\textbf{' + s + r'}'
        fmt_s = r'{\textcolor{' + pos_texcolor + r'}{' + s + r'}}'
    elif s in special:
        if special_texwt == 'bold':
            s = r'\textbf{' + s + r'}'
        fmt_s = r'{\textcolor{' + special_texcolor + r'}{' + s + r'}}'
    else:
        if def_texwt == 'bold':
            s = r'\textbf{' + s + r'}'
        if def_texcolor != 'black':
            fmt_s = r'{\textcolor{' + def_texcolor + r'}{' + s + r'}}'
        else:
            fmt_s = s
    la_string += fmt_s

extra_la_string = r'{\fontfamily{lmtt}\fontseries{l}\fontsize{10}{\baselineskip}\selectfont ' + la_string + r'}'
aa_la_string = r'\AAseq{' + la_string + r'}'    # \AAseq  as alias for  \seqsplit  (from package 'seqsplit', for multiple lines)

print('\n' + extra_la_string + '\n')
print('\n' + aa_la_string + '\n')

if OUTDIR:
    with open(os.path.join(OUTDIR, f'{seq_in}_latex.txt'), 'w') as lf:
        lf.write('\n' + extra_la_string + '\n')
        lf.write('\n' + aa_la_string + '\n')
