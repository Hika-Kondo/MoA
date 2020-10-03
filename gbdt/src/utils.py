import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def after_process(pred, ans, name):
    '''
    distribution of predicted dataframe
    args:
        pred: dataframe predicted
        ans:  dataframe ans
    '''
    concat = pd.DataFrame({"pred": pred, "ans" :ans})
    one = concat[concat["ans"] == 1]
    zero = concat[concat["ans"] == 0]

    sns.set()
    sns.set_style('whitegrid')
    sns.set_palette('gray')

    fig = plt.figure()
    ax = fig.add_subplot(2,2,1)
    ax.hist(one, bins=50)
    ax.set_xlabel("ones")

    ax = fig.add_subplot(2,2,2)
    ax.hist(zero, bins=50)
    ax.set_xlabel("zeros")

    fig = plt.figure()
    ax = fig.add_subplot(2,2,3)
    ax.hist(one, bins=50)
    ax.set_xlabel("ones")
    ax.set_yscale("log")

    ax = fig.add_subplot(2,2,4)
    ax.hist(zero, bins=50)
    ax.set_xlabel("zeros")
    ax.set_yscale("log")

    plt.savefig(name)
