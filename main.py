import matplotlib.pyplot as plt
import pandas as pd


def draw():
    filename = "dataset/first-results.csv"
    colnames = ['Cwasi', 'WasmEdge']
    df = pd.read_csv(filename, names=colnames,header=None, sep=",",float_precision='high').astype(float)
    df.head()
    ax=df.boxplot(grid=True,figsize=(5,7),fontsize=15,whis=[0,100])
    plt.yticks(fontsize=15)
    ax.set_ylabel("Seconds")
    plt.show()
    plt.savefig('results/execution-time/execution-time')



if __name__ == '__main__':
    draw()
