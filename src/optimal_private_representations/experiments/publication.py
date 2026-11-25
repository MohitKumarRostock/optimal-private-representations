"""Render the recorded figure design with publication-sized labels.

Presentation changes only: the numerical summaries are recalculated from
working copies of recorded data. The frozen references are never written.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text

from . import summaries, plot_population_runtime, plot_decomposition
from ..io import prepare_output


def publication_save(fig, folder, name):
    """Retain recorded plot geometry and the final presentation treatment."""
    folder = prepare_output(folder)
    if name == "representation_geometry":
        fig.set_size_inches(10, 4.6)
        left, right = fig.axes
        left.set_title(r"A  Exact coarsening at $\varepsilon=\log 2$")
        right.set_title("B  Privacy-dependent modulus")
        left.set_xticks([1e-26, 1e-20, 1e-14, 1e-8, 1e-2])
        right.set_xticks([1e-15, 1e-11, 1e-7, 1e-3, 1e1])
        for text in list(right.texts):
            text.remove()
        handles, labels = right.get_legend_handles_labels()
        right.get_legend().remove()
        fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    fig.canvas.draw()
    for text in fig.findobj(match=Text):
        text.set_fontsize(max(12.5, 1.4 * text.get_fontsize()))
    for ax in fig.axes:
        for line in ax.lines:
            line.set_markersize(max(4, line.get_markersize()))
            line.set_linewidth(max(1.5, line.get_linewidth()))
    fig.savefig(folder / (name + ".pdf"), bbox_inches="tight")
    fig.savefig(folder / (name + ".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def render(data_dir, figure_dir):
    data, figures = prepare_output(data_dir), prepare_output(figure_dir)
    modules = [summaries, plot_population_runtime, plot_decomposition]
    previous = [module.save for module in modules]
    try:
        for module in modules:
            module.save = publication_save
        summaries.main(["--data-dir", str(data), "--figure-dir", str(figures)])
        plot_population_runtime.main(data, figures)
        plot_decomposition.main(data, figures)
    finally:
        for module, save in zip(modules, previous):
            module.save = save


def render_learning(data_dir, figure_dir):
    """Summarize a complete new simulation and render its three learning figures."""
    data, figures = prepare_output(data_dir), prepare_output(figure_dir)
    previous = summaries.save
    try:
        summaries.save = publication_save
        summaries.main(["--data-dir", str(data), "--figure-dir", str(figures)])
    finally:
        summaries.save = previous

