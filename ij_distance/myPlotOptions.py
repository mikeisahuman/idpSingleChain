##  Mike Phillips, 10/30/2021
##  Nice file for saving typical plot options

import matplotlib.pyplot as plt

FINAL = True
#FINAL = False

##  Plot styles
#plt.style.use("dark_background")
#plt.style.use("seaborn-notebook")
plt.style.use("seaborn-v0_8-notebook")


##  Specifications of font sizes, linewidths, ticks
if FINAL:
    SMALL_SIZE = 22
    MEDIUM_SIZE = 40
    LARGE_SIZE = 40
    LEG_SIZE = 30       # different font size for legend (e.g. if too crowded)
    LW = 4.0            # line width for plot elements
    LW_AX = 3.0         # line width for axes (and ticks)
    TICK = 8            # tick size (length)
else:
    SMALL_SIZE = 18
    MEDIUM_SIZE = 20
    LARGE_SIZE = 24
    LEG_SIZE = 20       # different font size for legend (e.g. if too crowded)
    LW = 2.0            # line width for plot elements
    LW_AX = 1.5         # line width for axes (and ticks)
    TICK = 4            # tick size (length)


##  Global settings with RC params
plt.rc('font', size=MEDIUM_SIZE)        # controls default text sizes
plt.rc('axes', titlesize=LARGE_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=LARGE_SIZE)    # fontsize of the x and y labels
#plt.rc('axes', labelpad=10)             # padding between axes and corresponding labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=LEG_SIZE)     # legend fontsize
plt.rc('figure', titlesize=LARGE_SIZE)  # fontsize of the figure title
plt.rc('legend', edgecolor="inherit")   # ensure legend inherits color by default
plt.rc('xtick', bottom=True, top=True, direction="in")  # ensure ticks are on all sides
plt.rc('ytick', left=True, right=True, direction="in")  #
plt.rc('xtick.major', width=LW_AX, size=TICK)           # set size (length) and weight (width) of tickmarks
plt.rc('ytick.major', width=LW_AX, size=TICK)           #
plt.rc('xtick.minor', width=LW_AX*0.8, size=TICK*0.6)   # " minor tickmarks
plt.rc('ytick.minor', width=LW_AX*0.8, size=TICK*0.6)   #
plt.rc('axes', linewidth=LW_AX)         # adjust AXES linewidth
plt.rc('lines', linewidth=LW)           # adjust all plotted linewidths
#plt.rc('mathtext', fontset='stix')      # math font

