""" A visualizer module for Thinc """

try:
    import seaborn
    import matplotlib.pyplot as plt
except ImportError:
    pass


def visualize_attention(x, y, weights, layer="Encoder", self_attn=True):
    """
        Visualize self/outer attention
        Args:
            x: sentence
            y: sentence
            weights: (nH, nL, nL)
    """

    def heatmap(x, y, data, ax):
        seaborn.heatmap(
            data,
            square=True,
            xticklabels=y,
            yticklabels=x,
            vmin=0.0,
            vmax=1.0,
            cbar_kws=dict(use_gridspec=False, location="top"),
            ax=ax,
        )

    num = min(weights.shape[0], 4)
    fig, axs = plt.subplots(1, num)
    attn_type = "self attention" if self_attn else "outer attention"
    fig.suptitle("{} {} for all the heads".format(layer, attn_type))
    if len(weights.shape) == 3:
        for i in range(num):
            heatmap(x, y, weights[i], axs[i])
    else:
        raise ValueError("Wrong input weights dimensions")
    plt.show()
