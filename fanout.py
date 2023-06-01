import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from scipy.interpolate import interpolate


def draw_resource():
    filename = "dataset/fanout/cpu.csv"
    colnames = ['O 2M', 'C 2M', 'W 2M', 'O 30M', 'C 30M', 'W 30M', 'O 100M', 'C 100M', 'W 100M']
    groups = ['Openfaas', 'Cwasi', 'Wasmedge']
    df = pd.read_csv(filename, names=colnames, header=None, sep=",", float_precision='high').astype(float)
    fig = plt.figure()
    legend_elements = []
    df.loc[:, ['O 2M', 'O 30M', 'O 100M']].boxplot(grid=True, figsize=(15, 10), positions=[1, 8, 15],
                                                  widths=0.8, fontsize=15, whis=[0, 100],
                                                  medianprops=dict(color="white", linewidth=1.5),
                                                  whiskerprops=dict(color="black"),
                                                  patch_artist=True)
    legend_elements.append(Patch(label=groups[0]))
    df.loc[:, ['C 2M', 'C 30M', 'C 100M']].boxplot(grid=True, figsize=(15, 10), positions=[2, 9, 16],
                                                  widths=0.8, fontsize=15, patch_artist=True, whis=[0, 100],
                                                  boxprops=dict(facecolor="orange"),
                                                  medianprops=dict(color="black", linewidth=1),
                                                  whiskerprops=dict(color="black"))
    legend_elements.append(Patch(facecolor='orange', label=groups[1]))
    ax: object = df.loc[:, ['W 2M', 'W 30M', 'W 100M']].boxplot(grid=True, figsize=(15, 10), positions=[3, 10, 17],
                                                       widths=0.8, boxprops=dict(facecolor="yellowgreen"),
                                                       medianprops=dict(color="black", linewidth=1),
                                                       whiskerprops=dict(color="black"),
                                                       fontsize=15, patch_artist=True, whis=[0, 100])
    legend_elements.append(Patch(facecolor='yellowgreen', label=groups[2]))
    #ax.set_xticklabels([1, 2, 3,4,5,6,7,8,9,],
    #           ['2MB','','','', '15MB', '30MB','','','',])
    ax.set_xticks([])
    plt.ylabel("CPU %", fontsize=15)

    ax.set_xlabel("File Sizes in MB", fontsize=15)

    plt.twinx(ax)
    filename = "dataset/fanout/ram.csv"
    df = pd.read_csv(filename, names=colnames, header=None, sep=",", float_precision='high').astype(float)
    # df.head()
    groups = ['Openfaas', 'Cwasi', 'Wasmedge']
    # fig = plot.figure()
    legend_elements = []
    c="brown"
    df.loc[:, ['O 2M', 'O 30M', 'O 100M']].boxplot(grid=True, figsize=(15, 10), positions=[4, 11, 18],
                                                  widths=0.8, fontsize=15, whis=[0, 100],
                                                  medianprops=dict(color=c, linewidth=1),
                                                  boxprops=dict(color=c),
                                                  whiskerprops=dict(color=c),
                                                  capprops=dict(color=c),
                                                  patch_artist=True)
    legend_elements.append(Patch(label=groups[0]))
    df.loc[:, ['C 2M', 'C 30M', 'C 100M']].boxplot(grid=True, figsize=(15, 10), positions=[5, 12, 19],
                                                  widths=0.8, fontsize=15, patch_artist=True, whis=[0, 100],
                                                  boxprops=dict(facecolor="orange",color=c),
                                                  capprops=dict(color=c),
                                                  medianprops=dict(color=c, linewidth=1),
                                                  whiskerprops=dict(color=c))
    legend_elements.append(Patch(facecolor='orange', label=groups[1]))
    ax = df.loc[:, ['W 2M', 'W 30M', 'W 100M']].boxplot(grid=True, figsize=(15, 10), positions=[6, 13, 20],
                                                       widths=0.8, boxprops=dict(facecolor="yellowgreen",color=c),
                                                       medianprops=dict(color=c, linewidth=1),
                                                       whiskerprops=dict(color=c),
                                                       capprops=dict(color=c),
                                                       fontsize=15, patch_artist=True, whis=[0, 100])
    legend_elements.append(Patch(facecolor='yellowgreen', label=groups[2]))
    ax.grid(False)
    ax.set_xticks([])
    ax.tick_params(axis='y', colors=c)
    background_color='lightgray'
    ax.axvspan(3.5, 6.5, facecolor=background_color, alpha=0.5)
    ax.axvspan(10.5, 13.5, facecolor=background_color, alpha=0.5)
    ax.axvspan(17.5, 20.5, facecolor=background_color, alpha=0.5)
    plt.legend(handles=legend_elements, fontsize=10, loc='upper center', bbox_to_anchor=(0.5, 1.1),
               ncol=3, fancybox=True, shadow=True)

    plt.xticks([4, 11, 18],   ['2MB', '30MB', '100MB'])
    plt.ylabel("RAM %", fontsize=15,color=c)
    plt.xlabel("File Sizes in MB")

    #fig.tight_layout()
    plt.show()
    #plt.savefig('results/cpu-load')


def draw_throughput():
    filename = "dataset/fanout/throughput.csv"
    colnames = ['O 2M', 'C 2M', 'W 2M', 'O 4M', 'C 4M', 'W 4M',
                'O 6M', 'C 6M', 'W 6M', 'O 8M', 'C 8M', 'W 8M',
                'O 10M', 'C 10M', 'W 10M', 'O 20M', 'C 20M', 'W 20M',
                'O 30M', 'C 30M', 'W 30M', 'O 40M', 'C 40M', 'W 40M',
                'O 60M', 'C 60M', 'W 60M', 'O 100M', 'C 100M', 'W 100M']
    df = pd.read_csv(filename, names=colnames, header=None, sep=",", float_precision='high').astype(float)
    df.head()

    ax = df.loc[:, ['O 2M','O 4M', 'O 6M','O 8M','O 10M', 'O 20M','O 40M','O 60M','O 100M']].mean().plot(
        label="Openfaas")
    ax = df.loc[:, ['C 2M','C 4M', 'C 6M','C 8M','C 10M', 'C 20M', 'C 40M','C 60M','C 100M']].mean().plot(ax=ax,
                                                                                                              label="Cwasi")
    ax = df.loc[:, ['W 2M','W 4M', 'W 6M','W 8M','W 10M', 'W 20M', 'W 40M','W 60M','W 100M']].mean().plot(ax=ax,
                                                                                                              label="Wasmedge")
    plt.legend(loc="upper right")
    #ax.set_yscale('log')
    # ax.set_xticklabels(("4","10","20","40"))
    plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8], ['2', '4', '6', '8', '10', '20', '40','60','100'])
    plt.ylabel("Requests per second")

    plt.xlabel("File Sizes in MB")
    # set legend position

    plt.show()
    plt.savefig('results/execution-time/execution-time')
def draw_latency():
    filename = "dataset/fanout/latency.csv"
    colnames = ['O 2M', 'C 2M', 'W 2M', 'O 4M', 'C 4M', 'W 4M',
                'O 6M', 'C 6M', 'W 6M', 'O 8M', 'C 8M', 'W 8M',
                'O 10M', 'C 10M', 'W 10M', 'O 20M', 'C 20M', 'W 20M',
                'O 30M', 'C 30M', 'W 30M', 'O 40M', 'C 40M', 'W 40M',
                'O 60M', 'C 60M', 'W 60M', 'O 100M', 'C 100M', 'W 100M']
    df = pd.read_csv(filename, names=colnames, header=None, sep=",", float_precision='high').astype(float)
    df.head()

    ax = df.loc[:, ['O 2M','O 4M', 'O 6M','O 8M','O 10M', 'O 20M','O 40M','O 60M','O 100M']].sum().plot(
        label="Openfaas")
    ax = df.loc[:, ['C 2M','C 4M', 'C 6M','C 8M','C 10M', 'C 20M', 'C 40M','C 60M','C 100M']].sum().plot(ax=ax,
                                                                                                              label="Cwasi")
    ax = df.loc[:, ['W 2M','W 4M', 'W 6M','W 8M','W 10M', 'W 20M', 'W 40M','W 60M','W 100M']].sum().plot(ax=ax,
                                                                                                              label="Wasmedge")
    plt.legend(loc="upper left")

    #ax.set_yscale('log')
    # ax.set_xticklabels(("4","10","20","40"))
    plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8], ['2', '4', '6', '8', '10', '20', '40','60','100'])
    plt.ylabel("Seconds")

    plt.xlabel("File Sizes in MB")
    # set legend position

    plt.show()
    plt.savefig('results/execution-time/execution-time')