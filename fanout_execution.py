import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Patch




def draw_resource():
    filename = "dataset/fanout2/cpu.csv"
    colnames = ['O 2M', 'C 2M', 'W 2M', 'O 30M', 'C 30M', 'W 30M', 'O 100M', 'C 100M', 'W 100M']
    groups = ['Openfaas', 'Cwasi', 'Wasmedge']
    df = pd.read_csv(filename, names=colnames, header=None, sep=",", float_precision='high').astype(float)
    # df.plot()
    fig, axes = plt.subplots(nrows=1, ncols=2)
    legend_elements = []
    df.loc[:, ['O 2M', 'O 30M', 'O 100M']].boxplot(ax=axes[0], grid=True, positions=[1, 5, 9],
                                                   widths=0.8, whis=[0, 100],
                                                   medianprops=dict(color="white", linewidth=1.5),
                                                   whiskerprops=dict(color="black"),
                                                   patch_artist=True)
    legend_elements.append(Patch(label=groups[0]))
    df.loc[:, ['C 2M', 'C 30M', 'C 100M']].boxplot(ax=axes[0], grid=True, positions=[2, 6, 10],
                                                   widths=0.8, patch_artist=True, whis=[0, 100],
                                                   boxprops=dict(facecolor="tab:orange"),
                                                   medianprops=dict(color="black", linewidth=1),
                                                   whiskerprops=dict(color="black"))
    legend_elements.append(Patch(facecolor='orange', label=groups[1]))
    ax: object = df.loc[:, ['W 2M', 'W 30M', 'W 100M']].boxplot(ax=axes[0], grid=True,
                                                                positions=[3, 7, 11],
                                                                widths=0.8, boxprops=dict(facecolor="tab:green"),
                                                                medianprops=dict(color="black", linewidth=1),
                                                                whiskerprops=dict(color="black"),
                                                                patch_artist=True, whis=[0, 100])
    legend_elements.append(Patch(facecolor='tab:green', label=groups[2]))

    ax.set_xticks([2, 6, 10], ['10', '100', '500'])
    filename = "dataset/fanout2/ram.csv"
    df = pd.read_csv(filename, names=colnames, header=None, sep=",", float_precision='high').astype(float)
    # df.head()
    groups = ['Openfaas', 'Cwasi', 'Wasmedge']
    # fig = plot.figure()
    legend_elements = []
    c = "black"
    df.loc[:, ['O 2M', 'O 30M', 'O 100M']].boxplot(ax=axes[1], grid=True,  positions=[1, 5, 9],
                                                   widths=0.8, whis=[0, 100],
                                                   medianprops=dict(color="white", linewidth=1.5),
                                                   whiskerprops=dict(color="black"),
                                                   patch_artist=True)
    legend_elements.append(Patch(label=groups[0],hatch='|'))
    df.loc[:, ['C 2M', 'C 30M', 'C 100M']].boxplot(ax=axes[1], grid=True, positions=[2, 6, 10],
                                                   widths=0.8,  patch_artist=True, whis=[0, 100],
                                                   boxprops=dict(facecolor="tab:orange", color=c),
                                                   capprops=dict(color=c),
                                                   medianprops=dict(color=c, linewidth=1),
                                                   whiskerprops=dict(color=c))
    legend_elements.append(Patch(facecolor='tab:orange', label=groups[1]))
    ax = df.loc[:, ['W 2M', 'W 30M', 'W 100M']].boxplot(ax=axes[1], grid=True,  positions=[3, 7, 11],
                                                        widths=0.8, boxprops=dict(facecolor="tab:green", color=c),
                                                        medianprops=dict(color=c, linewidth=1),
                                                        whiskerprops=dict(color=c),
                                                        capprops=dict(color=c),
                                                         patch_artist=True, whis=[0, 100])
    legend_elements.append(Patch(facecolor='tab:green', label=groups[2]))

    # fig.tight_layout()
    plt.legend(handles=legend_elements, fontsize=12,loc='upper center', bbox_to_anchor=(-0.1, 1.15),
               ncol=3, fancybox=True, shadow=True, frameon=False, columnspacing=0.5,handletextpad=0.1)

    plt.xticks([2, 6, 10], ['10', '100', '500'])
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()

    axes[0].set_ylabel("%")
    axes[0].set_xlabel("CPU")
    axes[0].xaxis.set_label_coords(.5, -.2)
    axes[0].yaxis.set_label_coords(-.2, .5)
    axes[1].set_ylabel("Kb")
    axes[1].set_xlabel("RAM")
    axes[1].yaxis.set_label_position("right")
    axes[1].xaxis.set_label_coords(.5, -.2)
    axes[1].get_yaxis().get_offset_text().set_position((1.22, 0))
    #axes[1].yaxis.set_label_coords(.5, -.1)
    fig.subplots_adjust(
        top=0.9,
        bottom=0.2,
        left=0.12,
        right=0.835,
        hspace=0.1,
        wspace=0.1
    )

    plt.title("Executions", y=-0.22,x=-0.1, fontsize=12)

    # set legend position
    plt.savefig('figures/fanout_resource.eps')
    plt.show()


def draw_throughput_fanout():
    filename = "dataset/fanout2/out-throughput.csv"
    colnames = ['Openfaas', 'Cwasi', 'Wasmedge']
    df = pd.read_csv(filename, names=colnames, header=None, sep=",", float_precision='high').astype(float)
    df.head()
    df_cwasi = df.loc[:, ['Cwasi']].sample(n=50).sort_index()
    x = np.linspace(1, 50, 50)
    df_cwasi_reindex = pd.DataFrame(data=df_cwasi.to_numpy(),
                                    index=x,
                                    columns=['Cwasi'])

    ax = df.loc[:, ['Openfaas']].plot(label="Openfaas",marker='o',markevery=5)
    ax = df_cwasi_reindex.plot(ax=ax, label="Cwasi",marker='X',markevery=5)
    ax = df.loc[:, ['Wasmedge']].plot(ax=ax,marker='d',markevery=5)

    plt.legend(loc='upper center', fontsize=12, bbox_to_anchor=(0.5, 1.2), ncol=3, fancybox=True,
               handletextpad=0.1, columnspacing=0.8, frameon=False)
    ax.set_yscale('log')
    ax.set_xticklabels(("4", "10", "50", "100", "200", "400", "500"))
    plt.ylabel("Req/sec")

    plt.xlabel("Executions")
    plt.tight_layout()
    plt.savefig('figures/fanout_throughput.eps')
    # set legend position
    plt.show()



def draw_throughput_fanin():
    filename = "dataset/fanout2/in-throughput.csv"
    colnames = ['Openfaas', 'Cwasi', 'Wasmedge']
    df = pd.read_csv(filename, names=colnames, header=None, sep=",", float_precision='high').astype(float)
    df.head()
    df_cwasi = df.loc[:, ['Cwasi']].sample(n=50).sort_index()
    x = np.linspace(1, 50, 50)
    df_cwasi_reindex = pd.DataFrame(data=df_cwasi.to_numpy(),
                                    index=x,
                                    columns=['Cwasi'])

    ax = df.loc[:, ['Openfaas']].plot(label="Openfaas",marker='o',markevery=5)
    ax = df_cwasi_reindex.plot(ax=ax, label="Cwasi",marker='X',markevery=5)
    ax = df.loc[:, ['Wasmedge']].plot(ax=ax,marker='d',markevery=5)

    plt.legend(loc='upper center', fontsize=12, bbox_to_anchor=(0.5, 1.18), ncol=3, fancybox=True,
               handletextpad=0.1, columnspacing=0.8, frameon=False)
    ax.set_yscale('log')
    ax.set_xticklabels(("4", "10", "50", "100", "200", "400", "500"))
    plt.ylabel("Req/sec")
    plt.tight_layout()
    plt.xlabel("Executions")
    # set legend position
    plt.savefig('figures/fanin_throughput.eps')
    plt.show()



def draw_latency_fanout():
    filename = "dataset/fanout2/fanout_latency.csv"
    colnames = ['Openfaas', 'Cwasi', 'Wasmedge']
    fig = plt.gcf()
    df = pd.read_csv(filename, names=colnames, header=None, sep=",", float_precision='high').astype(float)
    df.head()
    x = np.linspace(1, 50, 50)
    df_of = df.loc[:, ['Openfaas']].sample(n=50).sort_index()
    df_of_reindex = pd.DataFrame(data=df_of.to_numpy(),
                                 index=x,
                                 columns=['Openfaas'])
    df_cwasi = df.loc[:, ['Cwasi']].sample(n=50).sort_index()
    df_cwasi_reindex = pd.DataFrame(data=df_cwasi.to_numpy(),
                                    index=x,
                                    columns=['Cwasi'])
    df_we = df.loc[:, ['Wasmedge']].sample(n=50).sort_index()
    df_we_reindex = pd.DataFrame(data=df_we.to_numpy(),
                                 index=x,
                                 columns=['Wasmedge'])

    ax = df_of_reindex.plot(label="Openfaas",marker='o',markevery=5)
    ax = df_cwasi_reindex.plot(ax=ax, label="Cwasi",marker='X',markevery=5)
    ax = df_we_reindex.plot(ax=ax, label="Wasmedge",marker='d',markevery=5)
    plt.legend(loc='upper center', fontsize=12, bbox_to_anchor=(0.5, 1.2), ncol=3, fancybox=True,
               handletextpad=0.1, columnspacing=0.8, frameon=False)

    ax.set_yscale('log')
    ax.set_xticklabels(("4", "10", "50", "100", "200", "400", "500"))

    plt.ylabel("Seconds")

    plt.xlabel("Executions")
    plt.tight_layout()
    # set legend position
    plt.savefig('figures/fanout_latency.eps')
    plt.show()



def draw_latency_fanin():
    filename = "dataset/fanout2/fanin_latency.csv"
    colnames = ['Openfaas', 'Cwasi', 'Wasmedge']
    df = pd.read_csv(filename, names=colnames, header=None, sep=",", float_precision='high').astype(float)
    df.head()
    x = np.linspace(1, 50, 50)
    df_of = df.loc[:, ['Openfaas']].sample(n=50).sort_index()
    df_of_reindex = pd.DataFrame(data=df_of.to_numpy(),
                                 index=x,
                                 columns=['Openfaas'])
    df_cwasi = df.loc[:, ['Cwasi']].sample(n=50,replace=True).sort_index()
    df_cwasi_reindex = pd.DataFrame(data=df_cwasi.to_numpy(),
                                    index=x,
                                    columns=['Cwasi'])
    df_we = df.loc[:, ['Wasmedge']].sample(n=50).sort_index()
    df_we_reindex = pd.DataFrame(data=df_we.to_numpy(),
                                 index=x,
                                 columns=['Wasmedge'])

    ax = df_of_reindex.plot(label="Openfaas",marker='o',markevery=5)
    ax = df_cwasi_reindex.plot(ax=ax, label="Cwasi",marker='X',markevery=5)
    ax = df_we_reindex.plot(ax=ax, label="Wasmedge",marker='d',markevery=5)
    plt.legend(loc='upper center', fontsize=12, bbox_to_anchor=(0.5, 1.18), ncol=3, fancybox=True,
               handletextpad=0.1, columnspacing=0.8, frameon=False)

    ax.set_yscale('log')
    ax.set_xticklabels(("4", "10", "50", "100", "200", "400", "500"))
    plt.ylabel("Seconds")
    plt.tight_layout()

    plt.xlabel("Executions")
    # set legend position
    plt.savefig('figures/fanin_latency.eps')
    plt.show()

