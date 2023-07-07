import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch, Rectangle
import numpy as np
from scipy.interpolate import make_interp_spline, interpolate

import fanout
import fanout_execution
plt.rcParams['axes.grid'] = True
plt.rcParams["font.size"] = 14
plt.rcParams["figure.figsize"] = (5,3.2)
plt.rcParams["hatch.color"] = 'white'
#plt.rcParams["figure.autolayout"] = True



def draw_ram(plot):
    colnames = ['O 2M', 'C 2M', 'W 2M', 'O 15M', 'C 15M', 'W 15M', 'O 30M', 'C 30M', 'W 30M']
    filename = "dataset/May20/ram_nuc.csv"
    df = pd.read_csv(filename, names=colnames, header=None, sep=",", float_precision='high').astype(float)
    # df.head()
    groups = ['Openfaas', 'Cwasi', 'Wasmedge']
    #fig = plot.figure()
    legend_elements = []
    df.loc[:, ['O 2M', 'O 15M', 'O 30M']].boxplot(grid=True, figsize=(15, 10), positions=[1.3, 3.6, 5.6],
                                                  widths=0.2, fontsize=15, whis=[0, 100],
                                                  medianprops=dict(color="white", linewidth=1.5),
                                                  boxprops=dict(facecolor="darkblue"),
                                                  patch_artist=True)
    legend_elements.append(Patch(facecolor='darkblue',label=groups[0]))
    df.loc[:, ['C 2M', 'C 15M', 'C 30M']].boxplot(grid=True, figsize=(15, 10), positions=[2, 4.2, 6.2],
                                                  widths=0.2, fontsize=15, patch_artist=True, whis=[0, 100],
                                                  boxprops=dict(facecolor="darkorange"),
                                                  medianprops=dict(color="black", linewidth=1),
                                                  whiskerprops=dict(color="black"))
    legend_elements.append(Patch(facecolor='darkorange', label=groups[1]))
    ax = df.loc[:, ['W 2M', 'W 15M', 'W 30M']].boxplot(grid=True, figsize=(15, 10), positions=[2.6, 4.8, 6.8],
                                                       widths=0.2, boxprops=dict(facecolor="green"),
                                                       medianprops=dict(color="black", linewidth=1),
                                                       fontsize=15, patch_artist=True, whis=[0, 100])
    legend_elements.append(Patch(facecolor='green', label=groups[2]))
    ax.grid(False)
    #ax.set_xticks([])

    #plot.ylabel("Usage %", fontsize=15)

    #plt.legend(handles=legend_elements, fontsize=10)
    #fig.tight_layout()
    plot.show()
    # plt.savefig('results/cpu-load')



def draw():
    filename = "dataset/May20/cpu_nuc.csv"
    colnames = ['O 2M', 'C 2M', 'W 2M', 'O 30M', 'C 30M', 'W 30M', 'O 100M', 'C 100M', 'W 100M']
    groups = ['Openfaas', 'Cwasi', 'Wasmedge']
    df = pd.read_csv(filename, names=colnames,header=None, sep=",",float_precision='high').astype(float)
    #df.plot()
    fig, axes = plt.subplots(nrows=1, ncols=2)
    legend_elements = []
    hatches = ['o', '++', 'x']
    ax=df.loc[:, ['O 2M', 'O 30M', 'O 100M']].boxplot(ax=axes[0],grid=True, positions=[1, 5, 9],
                                                   widths=0.8, whis=[0, 100],
                                                   medianprops=dict(color="white", linewidth=1.5),
                                                   whiskerprops=dict(color="black"),
                                                   patch_artist=True)

    legend_elements.append(Patch(label=groups[0],hatch='|'))
    df.loc[:, ['C 2M', 'C 30M', 'C 100M']].boxplot(ax=axes[0],grid=True, positions=[2, 6, 10],
                                                   widths=0.8, patch_artist=True, whis=[0, 100],
                                                   boxprops=dict(facecolor="tab:orange"),
                                                   medianprops=dict(color="black", linewidth=1),
                                                   whiskerprops=dict(color="black"))
    legend_elements.append(Patch(facecolor='tab:orange', label=groups[1]))
    ax: object = df.loc[:, ['W 2M', 'W 30M', 'W 100M']].boxplot(ax=axes[0],grid=True,  positions=[3, 7, 11],
                                                                widths=0.8, boxprops=dict(facecolor="tab:green"),
                                                                medianprops=dict(color="black", linewidth=1),
                                                                whiskerprops=dict(color="black"),
                                                                patch_artist=True, whis=[0, 100])
    legend_elements.append(Patch(facecolor='tab:green', label=groups[2]))

    ax.set_xticks([2, 6, 10], ['2MB', '30MB', '100MB'])
    filename = "dataset/May20/ram_nuc.csv"
    df = pd.read_csv(filename, names=colnames, header=None, sep=",", float_precision='high').astype(float)
    # df.head()
    groups = ['Openfaas', 'Cwasi', 'Wasmedge']
    # fig = plot.figure()
    legend_elements = []
    c = "black"
    df.loc[:, ['O 2M', 'O 30M', 'O 100M']].boxplot(ax=axes[1],grid=True,positions=[1, 5, 9],
                                                   widths=0.8,  whis=[0, 100],
                                                   medianprops=dict(color="white", linewidth=1),
                                                   whiskerprops=dict(color=c),
                                                   patch_artist=True)
    legend_elements.append(Patch(label=groups[0],hatch='|'))
    df.loc[:, ['C 2M', 'C 30M', 'C 100M']].boxplot(ax=axes[1],grid=True, positions=[2, 6, 10],
                                                   widths=0.8, patch_artist=True, whis=[0, 100],
                                                   boxprops=dict(facecolor="tab:orange", color=c),
                                                   capprops=dict(color=c),
                                                   medianprops=dict(color=c, linewidth=1),
                                                   whiskerprops=dict(color=c))
    legend_elements.append(Patch(facecolor='tab:orange', label=groups[1]))
    ax = df.loc[:, ['W 2M', 'W 30M', 'W 100M']].boxplot(ax=axes[1],grid=True,  positions=[3, 7, 11],
                                                        widths=0.8, boxprops=dict(facecolor="tab:green", color=c),
                                                        medianprops=dict(color=c, linewidth=1),
                                                        whiskerprops=dict(color=c),
                                                        capprops=dict(color=c),
                                                        patch_artist=True, whis=[0, 100])
    legend_elements.append(Patch(facecolor='tab:green', label=groups[2]))

    #fig.tight_layout()
    plt.legend(handles=legend_elements, fontsize=13, loc='upper center',bbox_to_anchor=(-0.1, 1.14),
               ncol=3, fancybox=True, shadow=True,frameon=False,columnspacing=0.5,handletextpad=0)
    #plt.legend(loc='upper center', fontsize=12, bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True,
    #           handletextpad=0.1, columnspacing=0.8, frameon=False)
    #plt.subplots_adjust(wspace=0.1, hspace=0)
    plt.xticks([2, 6, 10], ['2MB', '30MB', '100MB'])
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()

    axes[0].set_ylabel("%")
    axes[0].set_xlabel("CPU")
    axes[0].xaxis.set_label_coords(.5, -.112)
    #axes[0].yaxis.set_label_coords(-.02, .5)
    axes[1].set_ylabel("Kb")
    axes[1].set_xlabel("RAM")
    axes[1].yaxis.set_label_position("right")
    axes[1].xaxis.set_label_coords(.5, -.112)
    axes[1].get_yaxis().get_offset_text().set_position((1.25, 0))
    fig.subplots_adjust(
        bottom=0.13,
        left=0.13,
        right=0.867,
        hspace=0.1,
        wspace=0.1
    )
    # set legend position

    plt.savefig('figures/seq_resource.eps')
    plt.show()


def draw_resource():
    filename = "dataset/May20/cpu_nuc.csv"
    colnames = ['O 2M', 'C 2M', 'W 2M', 'O 30M', 'C 30M', 'W 30M', 'O 100M', 'C 100M', 'W 100M']
    groups = ['Openfaas', 'Cwasi', 'Wasmedge']
    df = pd.read_csv(filename, names=colnames, header=None, sep=",", float_precision='high').astype(float)
    fig, axs = plt.subplots(1, 2)

    legend_elements = []
    df.loc[:, ['O 2M', 'O 30M','O 100M']].boxplot(grid=True, figsize=(15, 10), positions=[1, 5, 15],
                                                  widths=0.8, fontsize=15, whis=[0, 100],
                                                  medianprops=dict(color="white", linewidth=1.5),
                                                  whiskerprops=dict(color="black"),
                                                  patch_artist=True)
    legend_elements.append(Patch(label=groups[0]))
    df.loc[:, ['C 2M', 'C 30M','C 100M']].boxplot(grid=True, figsize=(15, 10), positions=[2, 6, 16],
                                                  widths=0.8, fontsize=15, patch_artist=True, whis=[0, 100],
                                                  boxprops=dict(facecolor="orange"),
                                                  medianprops=dict(color="black", linewidth=1),
                                                  whiskerprops=dict(color="black"))
    legend_elements.append(Patch(facecolor='orange', label=groups[1]))
    ax: object = df.loc[:, ['W 2M', 'W 30M','W 100M']].boxplot(grid=True, figsize=(15, 10), positions=[3, 7, 17],
                                                       widths=0.8, boxprops=dict(facecolor="yellowgreen"),
                                                       medianprops=dict(color="black", linewidth=1),
                                                       whiskerprops=dict(color="black"),
                                                       fontsize=15, patch_artist=True, whis=[0, 100])
    legend_elements.append(Patch(facecolor='yellowgreen', label=groups[2]))
    #ax.set_xticklabels([1, 2, 3,4,5,6,7,8,9,],
    #           ['2MB','','','', '15MB', '30MB','','','',])
    ax.set_xticks([])
    plt.ylabel("CPU %", fontsize=15)

    ax.set_xlabel("File Sizes in MB",fontsize=15)

    plt.twinx(ax)
    filename = "dataset/May20/ram_nuc.csv"
    df = pd.read_csv(filename, names=colnames, header=None, sep=",", float_precision='high').astype(float)
    # df.head()
    groups = ['Openfaas', 'Cwasi', 'Wasmedge']
    # fig = plot.figure()
    legend_elements = []
    c="brown"
    df.loc[:, ['O 2M', 'O 30M','O 100M']].boxplot(grid=True, figsize=(15, 10), positions=[4, 11, 18],
                                                  widths=0.8, fontsize=15, whis=[0, 100],
                                                  medianprops=dict(color=c, linewidth=1),
                                                  boxprops=dict(color=c),
                                                  whiskerprops=dict(color=c),
                                                  capprops=dict(color=c),
                                                  patch_artist=True)
    legend_elements.append(Patch(label=groups[0]))
    df.loc[:, ['C 2M', 'C 30M','C 100M']].boxplot(grid=True, figsize=(15, 10), positions=[5, 12, 19],
                                                  widths=0.8, fontsize=15, patch_artist=True, whis=[0, 100],
                                                  boxprops=dict(facecolor="orange",color=c),
                                                  capprops=dict(color=c),
                                                  medianprops=dict(color=c, linewidth=1),
                                                  whiskerprops=dict(color=c))
    legend_elements.append(Patch(facecolor='orange', label=groups[1]))
    ax = df.loc[:, ['W 2M', 'W 30M','W 100M']].boxplot(grid=True, figsize=(15, 10), positions=[6, 13, 20],
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
    plt.legend(handles=legend_elements,fontsize=10,loc='upper center', bbox_to_anchor=(0.5, 1.1),
              ncol=3, fancybox=True, shadow=True)

    plt.xticks([4, 11, 18],   ['2MB', '30MB', '100MB'])
    plt.ylabel("RAM %", fontsize=15,color=c)
    plt.xlabel("File Sizes in MB")

    #fig.tight_layout()
    plt.show()
    #plt.savefig('results/cpu-load')


def draw_exec_storage():
    filename = "dataset/May20/x86_nuc_exec.csv"
    colnames = ['O 2M', 'OS 2M', 'C 2M', 'W 2M', 'WS 2M',
                'O 4M', 'OS 4M', 'C 4M', 'W 4M', 'WS 4M',
                'O 6M', 'OS 6M', 'C 6M', 'W 6M', 'WS 6M',
                'O 8M', 'OS 8M', 'C 8M', 'W 8M', 'WS 8M',
                'O 10M', 'OS 10M', 'C 10M', 'W 10M', 'WS 10M',
                'O 20M', 'OS 20M', 'C 20M', 'W 20M', 'WS 20M',
                'O 30M', 'OS 30M', 'C 30M', 'W 30M', 'WS 30M',
                'O 40M', 'OS 40M', 'C 40M', 'W 40M', 'WS 40M',
                'O 60M', 'OS 60M', 'C 60M', 'W 60M', 'WS 60M',
                'O 100M', 'OS 100M', 'C 100M', 'W 100M', 'WS 100M']
    df = pd.read_csv(filename, names=colnames, header=None, sep=",", float_precision='high').astype(float)
    df.head()

    ax = df.loc[:, ['O 2M','O 4M', 'O 6M','O 8M','O 10M', 'O 20M','O 40M','O 60M','O 100M']].mean().plot(color="deepskyblue",
        label="Openfaas")
    ax = df.loc[:, ['OS 2M', 'OS 4M', 'OS 6M', 'OS 8M', 'OS 10M', 'OS 20M', 'OS 40M', 'OS 60M', 'OS 100M']].mean().plot(color="darkblue",
        label="Openfaas + Storage")
    ax = df.loc[:, ['C 2M','C 4M', 'C 6M','C 8M','C 10M', 'C 20M', 'C 40M','C 60M','C 100M']].mean().plot(ax=ax, color="darkorange",
                                                                                                              label="Cwasi")
    ax = df.loc[:, ['W 2M','W 4M', 'W 6M','W 8M','W 10M', 'W 20M', 'W 40M','W 60M','W 100M']].mean().plot(ax=ax, color="lightgreen",
                                                                                                              label="Wasmedge")
    ax = df.loc[:, ['WS 2M', 'WS 4M', 'WS 6M', 'WS 8M', 'WS 10M', 'WS 20M', 'WS 40M', 'WS 60M', 'WS 100M']].mean().plot(ax=ax,color="forestgreen",
                                                                                                              label="Wasmedge + Storage")
    plt.legend(loc='upper center', fontsize=12, bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True,
               handletextpad=0.1, columnspacing=0.8, frameon=False)

    #ax.set_yscale('log')
    # ax.set_xticklabels(("4","10","20","40"))
    plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8], ['2', '4', '6', '8', '10', '20', '40','60','100'])
    plt.ylabel("Seconds")
    plt.xlabel("File Sizes in MB")
    # set legend position

    plt.show()
    plt.savefig('results/execution-time/execution-time')
def draw_latency():
    filename = "dataset/May20/x86_nuc_exec.csv"
    colnames = ['O 2M', 'C 2M', 'W 2M', 'O 4M', 'C 4M', 'W 4M',
                'O 6M', 'C 6M', 'W 6M', 'O 8M', 'C 8M', 'W 8M',
                'O 10M', 'C 10M', 'W 10M','O 15M', 'C 15M', 'W 15M', 'O 20M', 'C 20M', 'W 20M',
                'O 30M', 'C 30M', 'W 30M', 'O 40M', 'C 40M', 'W 40M',
                'O 60M', 'C 60M', 'W 60M', 'O 100M', 'C 100M', 'W 100M']
    fig = plt.gcf()
    df = pd.read_csv(filename, names=colnames, header=None, sep=",", float_precision='high').astype(float)
    df.head()

    ax = df.loc[:, ['O 2M','O 4M', 'O 6M','O 8M','O 10M', 'O 20M','O 40M','O 60M','O 100M']].mean().plot(label="Openfaas",marker='o',markevery=1)
    ax = df.loc[:, ['C 2M','C 4M', 'C 6M','C 8M','C 10M', 'C 20M', 'C 40M','C 60M','C 100M']].mean().plot(ax=ax, label="Cwasi",marker='X',markevery=1)
    ax = df.loc[:, ['W 2M','W 4M', 'W 6M','W 8M','W 10M', 'W 20M', 'W 40M','W 60M','W 100M']].mean().plot(ax=ax,label="Wasmedge",marker='d',markevery=1)
    plt.legend(loc='upper center', fontsize=12, bbox_to_anchor=(0.5, 1.2),ncol=3, fancybox=True,
               handletextpad=0.1,columnspacing=0.8,frameon=False)

    ax.set_yscale('log')
    # ax.set_xticklabels(("4","10","20","40"))
    plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8], ['2', '4', '6', '8', '10', '20', '40','60','100'])
    plt.ylabel("Seconds")
    plt.xlabel("Input size in MB")
    #ax.yaxis.set_label_coords(-.115, .5)
    # set legend position
    fig.tight_layout()
    plt.savefig('figures/seq_latency.eps')
    plt.show()


def draw_exec_inter():
    filename = "dataset/May20/x86_nuc_exec.csv"
    x=['2','4','6', '8','10', '20','40','60','100']
    colnames = ['O 2M', 'C 2M', 'W 2M','O 4M', 'C 4M', 'W 4M',
                'O 6M', 'C 6M', 'W 6M','O 8M', 'C 8M', 'W 8M',
                'O 10M', 'C 10M', 'W 10M','O 15M', 'C 15M', 'W 15M',
                'O 20M', 'C 20M', 'W 20M',
                'O 30M', 'C 30M', 'W 30M','O 40M', 'C 40M', 'W 40M',
                'O 60M', 'C 60M', 'W 60M','O 100M', 'C 100M', 'W 100M']
    df = pd.read_csv(filename, names=colnames, header=None, sep=",", float_precision='high').astype(float)
    df.head()
    # x_new, bspline, y_new
    y_o=df.loc[:, ['O 2M','O 4M', 'O 6M','O 8M','O 10M', 'O 20M', 'O 40M','O 60M','O 100M']].mean().to_numpy()
    y_c=df.loc[:, ['C 2M','C 4M', 'C 6M','C 8M','C 10M', 'C 20M', 'C 40M','C 60M','C 100M']].mean().to_numpy()
    y_w =df.loc[:,['W 2M','W 4M', 'W 6M','W 8M','W 10M', 'W 20M', 'W 40M','W 60M','W 100M']].mean().to_numpy()

    # x_new, bspline, y_new
    x_new = np.linspace(1, 100,50)

    ax = plt.plot(x_new, interpolate.make_interp_spline(x, y_o)(x_new), label="Openfaas")
    ax = plt.plot(x_new, interpolate.make_interp_spline(x, y_c)(x_new), label="Cwasi")
    ax = plt.plot(x_new, interpolate.make_interp_spline(x, y_w)(x_new), label="Wasmedge")

    plt.legend(loc="upper left")
    plt.yscale('log')


    #ax.set_xticklabels(("4","10","20","40"))
    #plt.xticks([0, 1, 2, 3,4,5,6,7,8],x)
    plt.ylabel("Transfer time in seconds")


    plt.xlabel("File Sizes in MB")
    # set legend position

    plt.show()
    plt.savefig('results/execution-time/execution-time')

def draw_exec_inter_storage():
    filename = "dataset/May20/x86_nuc_exec.csv"
    x=['2','4','6', '8','10', '20','40','60','100']
    colnames = ['O 2M', 'OS 2M', 'C 2M', 'W 2M','WS 2M',
                'O 4M', 'OS 4M', 'C 4M', 'W 4M','WS 4M',
                'O 6M', 'OS 6M','C 6M', 'W 6M','WS 6M',
                'O 8M', 'OS 8M','C 8M', 'W 8M','WS 8M',
                'O 10M', 'OS 10M','C 10M', 'W 10M','WS 10M',
                'O 20M','OS 20M', 'C 20M', 'W 20M','WS 20M',
                'O 30M','OS 30M', 'C 30M', 'W 30M','WS 30M', 'O 40M','OS 40M', 'C 40M', 'W 40M','WS 40M',
                'O 60M','OS 60M', 'C 60M', 'W 60M','WS 60M','O 100M','OS 100M', 'C 100M', 'W 100M','WS 100M']
    df = pd.read_csv(filename, names=colnames, header=None, sep=",", float_precision='high').astype(float)
    df.head()
    # x_new, bspline, y_new
    y_o=df.loc[:, ['O 2M','O 4M', 'O 6M','O 8M','O 10M', 'O 20M', 'O 40M','O 60M','O 100M']].mean().to_numpy()
    y_os = df.loc[:, ['OS 2M', 'OS 4M', 'OS 6M', 'OS 8M', 'OS 10M', 'OS 20M', 'OS 40M', 'OS 60M', 'OS 100M']].mean().to_numpy()
    y_c=df.loc[:, ['C 2M','C 4M', 'C 6M','C 8M','C 10M', 'C 20M', 'C 40M','C 60M','C 100M']].mean().to_numpy()
    y_w =df.loc[:,['W 2M','W 4M', 'W 6M','W 8M','W 10M', 'W 20M', 'W 40M','W 60M','W 100M']].mean().to_numpy()
    y_ws = df.loc[:, ['WS 2M', 'WS 4M', 'WS 6M', 'WS 8M', 'WS 10M', 'WS 20M', 'WS 40M', 'WS 60M', 'WS 100M']].mean().to_numpy()

    # x_new, bspline, y_new
    x_new = np.linspace(1, 100,50)

    ax = plt.plot(x_new, interpolate.make_interp_spline(x, y_o)(x_new), label="Openfaas")
    ax = plt.plot(x_new, interpolate.make_interp_spline(x, y_os)(x_new), label="Openfaas + Storage")
    ax = plt.plot(x_new, interpolate.make_interp_spline(x, y_c)(x_new), label="Cwasi")
    ax = plt.plot(x_new, interpolate.make_interp_spline(x, y_w)(x_new), label="Wasmedge")
    ax = plt.plot(x_new, interpolate.make_interp_spline(x, y_ws)(x_new), label="Wasmedge + Storage")

    plt.legend(loc="upper left")
    plt.yscale('log')


    #ax.set_xticklabels(("4","10","20","40"))
    #plt.xticks([0, 1, 2, 3,4,5,6,7,8],x)
    plt.ylabel("Transfer time in seconds")


    plt.xlabel("File Sizes in MB")
    # set legend position

    plt.show()
    plt.savefig('results/execution-time/execution-time')

def draw_nettraffic():
    filename = "dataset/May20/net_traffic_compact.csv"
    colnames = ['O 4M', 'C 4M', 'W 4M', 'O 10M', 'C 10M', 'W 10M', 'O 20M', 'C 20M', 'W 20M', 'O 40M', 'C 40M', 'W 40M']
    df = pd.read_csv(filename, names=colnames, header=None, sep=",", float_precision='high').astype(float)
    df.head()
    # df.plot()
    fig, axes = plt.subplots(nrows=4, ncols=1)
    # add DataFrames to subplots
    ax = df.loc[:, ['O 4M', 'C 4M', 'W 4M']].plot(ax=axes[0], kind='area', stacked=False, xlim=(0, 23))
    ax.legend(loc='center right', bbox_to_anchor=(1.14, 0.5))
    ax = df.loc[:, ['O 10M', 'C 10M', 'W 10M']].plot(ax=axes[1], kind='area', stacked=False, xlim=(0, 38))
    ax.legend(loc='center right', bbox_to_anchor=(1.14, 0.5))
    ax = df.loc[:, ['O 20M', 'C 20M', 'W 20M']].plot(ax=axes[2], kind='area', stacked=False, xlim=(0, 63))
    ax.legend(loc='center right', bbox_to_anchor=(1.14, 0.5))
    ax = df.loc[:, ['O 40M', 'C 40M', 'W 40M']].plot(ax=axes[3], kind='area', stacked=False, xlim=(0, 120))
    ax.legend(loc='center right', bbox_to_anchor=(1.14, 0.5))
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.6, hspace=0.6, left=0.15, bottom=0.12, right=0.9, top=0.96)
    plt.ylabel("Network IO rate Kbps")
    ax.yaxis.set_label_coords(-.12, 2.5)
    plt.xlabel("Seconds")
    # set legend position
    plt.show()
    plt.savefig('results/execution-time/execution-time')

def draw_throughput():
    filename = "dataset/May20/throughput.csv"
    colnames = ['O 2M', 'C 2M', 'W 2M', 'O 4M', 'C 4M', 'W 4M',
                'O 6M', 'C 6M', 'W 6M', 'O 8M', 'C 8M', 'W 8M',
                'O 10M', 'C 10M', 'W 10M', 'O 20M', 'C 20M', 'W 20M',
                'O 30M', 'C 30M', 'W 30M', 'O 40M', 'C 40M', 'W 40M',
                'O 60M', 'C 60M', 'W 60M', 'O 100M', 'C 100M', 'W 100M']
    fig = plt.gcf()
    df = pd.read_csv(filename, names=colnames, header=None, sep=",", float_precision='high').astype(float)
    df.head()

    ax = df.loc[:, ['O 2M','O 4M', 'O 6M','O 8M','O 10M', 'O 20M','O 40M','O 60M','O 100M']].mean().plot(
        label="Openfaas",marker='o',markevery=1)
    ax = df.loc[:, ['C 2M','C 4M', 'C 6M','C 8M','C 10M', 'C 20M', 'C 40M','C 60M','C 100M']].mean().plot(ax=ax,
                                                                                                              label="Cwasi",marker='X',markevery=1)
    ax = df.loc[:, ['W 2M','W 4M', 'W 6M','W 8M','W 10M', 'W 20M', 'W 40M','W 60M','W 100M']].mean().plot(ax=ax,
                                                                                                              label="Wasmedge",marker='d',markevery=1)
    plt.legend(loc='upper center', fontsize=12, bbox_to_anchor=(0.5, 1.2), ncol=3, fancybox=True,
               handletextpad=0.1, columnspacing=0.8, frameon=False)
    ax.set_yscale('log')
    # ax.set_xticklabels(("4","10","20","40"))
    plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8], ['2', '4', '6', '8', '10', '20', '40','60','100'])
    plt.ylabel("Req/sec")

    plt.xlabel("Input size in MB")
    # set legend position
    fig.tight_layout()
    plt.savefig('figures/seq_throughput.eps')
    plt.show()


if __name__ == '__main__':
    #fanout_execution.draw_latency_fanin()
    #fanout_execution.draw_latency_fanout()
    fanout_execution.draw_resource()
    #fanout_execution.draw_throughput_fanout()
    #fanout_execution.draw_throughput_fanin()
    #draw()
    #draw_exec_storage()
    #draw_throughput()
    #fanout.draw_resource()
    #fanout.draw_throughput()
    #fanout.draw_latency()
    #draw_latency()
    #draw_exec_inter()
    #draw_resource()
    #draw_ram()
    # draw_nettraffic()
