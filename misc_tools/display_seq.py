##  Mike Phillips, 6/13/2023
##  Script for displaying sequence in a nice way.
##  - e.g. to make figures / tables for publication
##  * adapted 2/24/2025, now using 'Sequence' object
##  command line arguments:
##      (1) direct string of aminos _or_ sequence name to load from CSV file
##      (2) filename for loading sequence from file, or string flag for default file
##      (3) name heading, if using CSV file
##      (4) sequence heading, if using CSV file

import Sequence as S
import matplotlib.pyplot as plt
import myPlotOptions as mpo
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

# characterize sequence, store aminos
if file_in and ('.csv' == file_in.lower()[-4:]):
    ob = S.Sequence(name=seq_in, file=file_in, headName=name_h, headSeq=seq_h, info=True)
elif file_in:
    ob = S.Sequence(name=seq_in, info=True)
else:
    ob = S.Sequence(name=seq_in, aminos=seq_in, info=True)
seq = ob.aminos     # grab primary sequence

labels = True       # include numerical labels with sequence ?
label_skip = 10     # number of characters to skip between labels (if shown)
label_end = False   # show ending label, i.e. total length of sequence ?
shift_end = 1       # extra horizontal space for ending label

# define types of aminos for special display (like color, bold, etc.)
neg = ("E", "D")
pos = ("R", "K")

neg_color = "red"
neg_wt = "bold"

pos_color = "blue"
pos_wt = "bold"

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

# general figure settings
dpi = 100
width = 6.6
#height = 1.5
#height = 1.1
height = 0.7

facy = 1.2      # factor to leave a little vertical space after newline

dpi_f = fdict['size']/dpi   # fraction of inch used by each character
dx = (dpi_f) / width        # width spacing fraction (translating to figure coords.)
dy = (dpi_f*facy) / height  # height ...

newx = 0        # where to start after a newline
#newx = dx
newy = 2.25*dy  # how much vertial space to skip after newline
lbly = dy       # where to show label (i.e. above or below)
shifty = dy*0.9 # extra shift between lines

fig = plt.figure(figsize=(width,height), dpi=dpi)
for i in range(len(seq)):
    s = seq[i]
    if s in neg:
        fdict.update( {"color":neg_color, "fontweight":neg_wt} )
    elif s in pos:
        fdict.update( {"color":pos_color, "fontweight":pos_wt} )
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
    if pt[0] > 0.9:
        pt[0] = newx
        pt[1] -= newy
plt.show()
plt.close()

