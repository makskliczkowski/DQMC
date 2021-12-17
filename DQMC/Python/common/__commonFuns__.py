import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib

plt.style.use(['science', 'ieee', 'no-latex'])

import numpy as np
import itertools
import os
import seaborn as sns
import pandas as pd
import math
import random
import imageio
import scipy.stats as stats
import warnings

warnings.filterwarnings('ignore')

from scipy.optimize import curve_fit
from joblib import Parallel, delayed

# ------------- DEFINITIONS

markers = itertools.cycle(['o', 's', 'v', '+'])
boundary_conditions = {0: "PBC", 1: "OBC"}
colors_ls = list(mcolors.TABLEAU_COLORS)[:30]
colors = itertools.cycle(sns.color_palette()[:3])
TWOPI = math.pi * 2
kPSep = os.sep

IsingPath = f"D:{kPSep}Uni{kPSep}SEMESTERS{kPSep}PRACE{kPSep}CONDENSED_GROUP_CLOUD_UNI{kPSep}Transverse_Ising{kPSep}Transverse_Ising_ETH{kPSep}" + \
            f"IsignTransverse_ETH"
WavefunctionsPath = f"D:{kPSep}Uni{kPSep}SEMESTERS{kPSep}PRACE{kPSep}CONDENSED_GROUP_CLOUD_UNI{kPSep}EIGENVECTORS_SHRINK{kPSep}"
# ------------- LAMBDAS
dat = lambda x: x.endswith('.dat')

# ------------- FUNCTIONS FOR FILES

'''
Function that given a list of directories creates each
'''


def createFolder(directories, silent=False):
    for folder in directories:
        try:
            if not os.path.isdir(folder):
                os.makedirs(folder)
                if not silent:
                    print("Created a directory : ", folder)
        except OSError:
            print("Creation of the directory %s failed" % folder)
            raise


# Guard against race condition
# except OSError as exc:
#    if exc.errno != errno.EEXIST:
#        raise

'''
Functions that given a directory can return a random file from it
Given without folder, it can return only the name of the file, without folder
'''


def readRandomFile(folder, cond, withoutFolder = False):
    choice = random.choice(os.listdir(folder))
    #print(choice)
    maxlen = len(os.listdir(folder))
    counter = 0
    while not cond(choice):
        choice = random.choice(os.listdir(folder))
        if counter > maxlen:
            raise
        counter += 1
    if withoutFolder:
        return choice
    else:
        return folder + choice


'''
Function that can print a list of elements creating indents
The separator also can be used to clean the indents.
Width is governing the width of each column. 
endlne, if true, puts an endline after last element of the list
'''


def justPrinter(file, sep="\t", elements=[], width=8, endline=True):
    for item in elements:
        file.write((str(item) + sep).ljust(width))
    if endline:
        file.write("\n")


# ------------- FUNCTIONS FOR PLOTS

'''
Returns the list of axes
'''


def listOfAxis(num, figsize=(10, 10), dpi=100):
    fig, ax = plt.subplots(num, figsize=(10, 10), dpi=100)
    sns.set_style("ticks")
    # set axis for it to always be a list
    axis = []
    if num > 1:  # if we have many columns to plot
        axis = [ax[i] for i in range(num)]
    else:
        axis = [ax]
    return fig, axis


# -------------- HELPERS

def theirFidelity(probaA, probaB):
    # first state
    bins = np.histogram_bin_edges(probaB)  # [mini + i * h for i in range(int((maxi - mini) / h))]

    f1, b1 = np.histogram(probaA, density=True, bins=bins)
    nans = np.isnan(f1)
    for i in range(len(nans)):
        if nans[i]:
            f1[i] = 0

    # plt.hist(probaA, density=True, bins=b1)
    #print("f1=",f1)
    # second state
    f2, b2 = np.histogram(probaB, density=True, bins=bins)
    # plt.hist(probaB, density=True, bins=b1)
    # print("\nf2=",f2,"\nbins=",bins)
    sum1 = np.sum(f1)
    if sum1 != 0:
        f1 /= sum1#f1.sum()
    sum2 = np.sum(f2)
    if sum2 != 0:
        f2 /= sum2#f1.sum()

    returner = np.multiply(f1, f2)
    # print(returner)
    # for i in range(len(f1)):
    returner = np.sqrt(returner)
    # print("sqrt=", returner)

    #    returner += math.sqrt(f1[i] * f2[i])
    return np.sum(returner)
