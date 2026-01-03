##  Mike Phillips, 3/6/2023
##  File for holding amino acid specific hydropathy paramters.

##  cf. "Hydropathy Patterning Complements Charge Patterning to Describe Conformational Preferences of Disordered Proteins"
##      Wenwei Zheng, Gregory Dignon, Matthew Brown, Young C. Kim, and Jeetain Mittal;  JPCL 2020  [SI]

LAMBDA_M = {
    "A": 0.730,
    "R": 0,
    "N": 0.432,
    "D": 0.378,
    "C": 0.595,
    "Q": 0.514,
    "E": 0.459,
    "G": 0.649,
    "H": 0.514,
    "I": 0.973,
    "L": 0.973,
    "K": 0.514,
    "M": 0.838,
    "F": 1.000,
    "P": 1.000,
    "S": 0.595,
    "T": 0.676,
    "W": 0.946,
    "Y": 0.865,
    "V": 0.892  }


##  Following set based on CALVADOS 2, optimized for both Rg and LLPS
##  cf. "Improved predictions of phase behaviour of intrinsically disordered proteins by tuning the interaction range"
##  Tesei & Lindorff-Larsen, Open Research Europe 2023 (SI)

LAMBDA_L = {
    'R' : 0.730762477,
    'D' : 0.041604048,
    'N' : 0.425585901,
    'E' : 0.000693546,
    'K' : 0.179021174,
    'H' : 0.466366729,
    'Q' : 0.393431855,
    'S' : 0.462541681,
    'C' : 0.56154351,
    'G' : 0.705884373,
    'T' : 0.371316298,
    'A' : 0.274329797,
    'M' : 0.530848113,
    'Y' : 0.977461145,
    'V' : 0.208376961,
    'W' : 0.989376474,
    'L' : 0.644000501,
    'I' : 0.542362361,
    'P' : 0.359312658,
    'F' : 0.867235898    }

SIGMA_L = {
    'R' : 6.56,
    'D' : 5.58,
    'N' : 5.68,
    'E' : 5.92,
    'K' : 6.36,
    'H' : 6.08,
    'Q' : 6.02,
    'S' : 5.18,
    'C' : 5.48,
    'G' : 4.50,
    'T' : 5.62,
    'A' : 5.04,
    'M' : 6.18,
    'Y' : 6.46,
    'V' : 5.86,
    'W' : 6.78,
    'L' : 6.18,
    'I' : 6.18,
    'P' : 5.56,
    'F' : 6.36   }

