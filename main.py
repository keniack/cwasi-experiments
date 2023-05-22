import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams['axes.grid'] = True

def draw_cpu():
    filename = "dataset/May20/cpu_nuc.csv"
    colnames = ['O 4M', 'C 4M','W 4M','O 10M', 'C 10M','W 10M','C 20M','O 20M','W 20M','O 40M','W 40M']
    df = pd.read_csv(filename, names=colnames,header=None, sep=",",float_precision='high').astype(float)
    df.head()
    #df.plot()
    fig, axes = plt.subplots(nrows=4, ncols=1)
    # add DataFrames to subplots
    df.loc[:,['O 4M', 'C 4M','W 4M']].resample(rule='H').mean().plot(ax=axes[0])
    df.loc[:,['O 10M', 'C 10M','W 10M']].plot(ax=axes[1])
    df.loc[:,['O 20M','C 20M','W 20M']].plot(ax=axes[2])
    df.loc[:,['O 40M','W 40M']].plot(ax=axes[3])
    fig.tight_layout()

    # set legend position
    plt.show()
    plt.savefig('results/execution-time/execution-time')
def draw_exec():
    filename = "dataset/May20/x86_nuc_exec.csv"
    colnames = ['O 4M', 'C 4M','W 4M','O 10M', 'C 10M','W 10M','O 20M','W 20M','O 40M','C 40M','W 40M']
    df = pd.read_csv(filename, names=colnames,header=None, sep=",",float_precision='high').astype(float)
    df.head()
    #df.plot()
    fig, axes = plt.subplots(nrows=4, ncols=1)
    # add DataFrames to subplots
    df.loc[:,['O 4M', 'C 4M','W 4M']].plot(ax=axes[0])
    df.loc[:,['O 10M', 'C 10M','W 10M']].plot(ax=axes[1])
    df.loc[:,['O 20M','W 20M']].plot(ax=axes[2])
    df.loc[:,['O 40M','C 40M','W 40M']].plot(ax=axes[3])
    fig.tight_layout()

    # set legend position
    plt.show()
    plt.savefig('results/execution-time/execution-time')



if __name__ == '__main__':
    #draw_exec()
    draw_cpu()
