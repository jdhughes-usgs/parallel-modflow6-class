import matplotlib.pyplot as plt

figwidth = 180  # 90 # mm
figwidth = figwidth / 10 / 2.54  # inches
figheight = figwidth
figsize = (figwidth, figheight)


def plot_virtual_memory(
    ax: plt.axis, processors: list, total: list, non_virtual: list
) -> None:
    """

    Parameters
    ----------
    ax: axis
        matplotlib axis object
    processors: list
        sorted list of processor numbers
    total: list
        total memory usage
    non_virtual: list
        non-virtual memory usage

    Returns
    -------
    None

    """
    ax.plot(
        processors,
        non_virtual,
        lw=0.5,
        color="black",
        marker=".",
        ms=4,
        mfc="none",
        mec="black",
        label="Non-virtual",
    )
    ax.fill_between(
        processors,
        non_virtual,
        y2=total,
        lw=0,
        fc="0.90",
        label="Virtual",
    )
    ax.legend(frameon=False)
    return
