import sys
import pathlib

_ROOT = (pathlib.Path(__file__).parent / "../../").as_posix()
sys.path.append(_ROOT)

import argparse
import json
import evaluation
import tqdm

import os
import pickle
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.lines as lines

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
})


def read_dataset(data_dict, dataset_filepath, filter_field=None):

    with open(dataset_filepath, "r") as fh:
        dataset = json.load(fh)

    if filter_field is None:
        filter_field = dataset["info"]["filter_field"]

    for idx, datum in enumerate(dataset["data"]):

        filter_value = datum[filter_field]
        error = datum["verification"]["has_error"]
        is_equivalent = datum["verification"]["is_equivalent"]
        if is_equivalent is None:
            is_equivalent = False

        if error is not None \
            and ("timeout" in error.lower() \
                or "seconds" in error.lower()):
            continue

        compl_list, equiv_list, verify_list = data_dict.setdefault(
            filter_value, ([], [], []))

        compl_list.append(int(error is None))
        equiv_list.append(int(is_equivalent))

        try:
            verify_list.append(int(datum["llm_verification"]["is_equivalent"]))
            pass
        except Exception:
            verify_list.append(None)
            pass

def get_dataset_filepath(base_dir, run_no, dataset_type, model_name):

    return "%s/run%d/%s/dataset_nlfs_%s_verified.json" % (
        base_dir, run_no, dataset_type, model_name)

def get_model_data_dict(data_dict, model_name):

    return data_dict.setdefault(model_name, {})

def get_data_dict_dir(base_dir):

    return "%s/parsed_data.pkl" % (base_dir)

def get_data_dict(base_dir):

    try:
        fh = open(get_data_dict_dir(base_dir), "rb")
        return pickle.load(fh)
    except Exception:

        return None

def save_data_dict(base_dir, data_dict):

    fh = open(get_data_dict_dir(base_dir), "wb")
    pickle.dump(data_dict, fh)

def clean_data_dict(base_dir):

    try:
        os.remove(get_data_dict_dir(base_dir))
    except Exception:

        pass

def parse_data(args):

    data_dict = {}

    progress_bar = tqdm.tqdm(
        total=len(args.models) * len(args.datasets) * len(args.runs),
        unit=" experiments",
        leave=False)

    for dataset_type in args.datasets:

        data_dict[dataset_type] = {}
        for model_name in args.models:

            data_dict[dataset_type][model_name] = {}
            for run_no in args.runs:

                dataset_filepath = get_dataset_filepath(
                    args.base_dir,
                    run_no,
                    dataset_type,
                    model_name)
                read_dataset(data_dict[dataset_type][model_name],
                             dataset_filepath)
                progress_bar.update(1)
    progress_bar.close()

    return data_dict

def parse_equiv_data(bdir):
    data_dict = {}
    for dataset_type in ['fol','fol_human','ksat','plogic','regex']:
        data_dict[dataset_type] = {}
        for model_name in ['gpt-3.5-turbo','llama-3-8b','mistral','phi-3']:
            data_dict[dataset_type][model_name] = {}
            for run_no in range(10):
                print("%s/equiv_results/%s_equiv_results/run%i/%s/dataset_nlfs_gpt-4o_verified.json"%(bdir,model_name,run_no,dataset_type))
                read_dataset(data_dict[dataset_type][model_name],"%s/equiv_results/%s_equiv_results/run%i/%s/dataset_nlfs_gpt-4o_verified.json"%(bdir,model_name,run_no,dataset_type))
    return data_dict

def plot_precision_recall(ax, twinx, data, plot_props, scatter_y_offset=0):


    x = []
    y_precision = []
    y_specificity = []
    for x_value in sorted(data.keys()):

        tp = 0
        tn = 0
        fp = 0
        fn = 0

        compl_list, verifier_equiv, llm_equiv = data[x_value]
        assert len(llm_equiv) > 0

        for compl, gt, v in zip(compl_list, verifier_equiv, llm_equiv):

            if not compl or v is None:
                continue

            if gt:
                if v:
                    tp += 1
                else:
                    fn += 1
            else:
                if not v:
                    tn += 1
                else:
                    fp += 1

        if (tp + fp) == 0:
            continue
        else:
            precision = tp / (tp + fp)

        if (tn + fp) == 0:
            continue
        else:
            specificity = tn / (tn + fp)

        x.append(x_value)
        y_precision.append(precision)
        y_specificity.append(specificity)

    ax.plot(x, y_precision, color=plot_props["color"])

    scatter_point = len(x) // 2
    ax.plot([x[scatter_point]], [y_precision[scatter_point]
                                 + scatter_y_offset],
               marker="o",
               markersize=8,
               color=plot_props["color"])
    twinx.plot(x, y_specificity, alpha=0.45, linestyle="dashed")

def plot_precision(ax, data, plot_props, scatter_y_offset=0):


    x = []
    y_precision = []
    for x_value in sorted(data.keys()):

        tp = 0
        tn = 0
        fp = 0
        fn = 0

        compl_list, verifier_equiv, llm_equiv = data[x_value]
        assert len(llm_equiv) > 0

        for compl, gt, v in zip(compl_list, verifier_equiv, llm_equiv):

            if not compl or v is None:
                continue

            if gt:
                if v:
                    tp += 1
                else:
                    fn += 1
            else:
                if not v:
                    tn += 1
                else:
                    fp += 1

        if (tp + fp) == 0:
            continue
        else:
            precision = tp / (tp + fp)

        x.append(x_value)
        y_precision.append(precision)

    ax.plot(x, y_precision, color=plot_props["color"],linestyle=plot_props["linestyle"])

def plot_specificity(ax, data, plot_props, scatter_y_offset=0):


    x = []
    y_specificity = []
    for x_value in sorted(data.keys()):

        tp = 0
        tn = 0
        fp = 0
        fn = 0

        compl_list, verifier_equiv, llm_equiv = data[x_value]
        assert len(llm_equiv) > 0

        for compl, gt, v in zip(compl_list, verifier_equiv, llm_equiv):

            if not compl or v is None:
                continue

            if gt:
                if v:
                    tp += 1
                else:
                    fn += 1
            else:
                if not v:
                    tn += 1
                else:
                    fp += 1

        if (tn + fp) == 0:
            continue
        else:
            specificity = tn / (tn + fp)

        x.append(x_value)
        y_specificity.append(specificity)

    ax.plot(x, y_specificity, color=plot_props["color"],linestyle=plot_props["linestyle"])

def plot_f1(ax, data, plot_props, scatter_y_offset=0):


    x = []
    y_f1 = []
    for x_value in sorted(data.keys()):

        tp = 0
        tn = 0
        fp = 0
        fn = 0

        compl_list, verifier_equiv, llm_equiv = data[x_value]
        assert len(llm_equiv) > 0

        for compl, gt, v in zip(compl_list, verifier_equiv, llm_equiv):

            if not compl or v is None:
                continue

            if gt:
                if v:
                    tp += 1
                else:
                    fn += 1
            else:
                if not v:
                    tn += 1
                else:
                    fp += 1

        f1 = tp + (0.5 * (fp + fn))
        if f1 == 0:
            continue
        f1 = tp / f1

        x.append(x_value)
        y_f1.append(f1)

    ax.plot(x, y_f1, color=plot_props["color"],linestyle=plot_props["linestyle"])


def plot(ax, data, idx, plot_props):

    x = []
    y = []
    y_err = []
    for x_value in sorted(data.keys()):

        x.append(x_value)
        y_values = data[x_value][idx]

        mean = np.mean(y_values)
        std = np.std(y_values)

        y.append(mean)
        y_err.append(std)

    x = np.asarray(x)
    y = np.asarray(y)

    ax.plot(x, y, color=plot_props["color"],
            linestyle=plot_props["linestyle"])

    # ax.fill_between(x, y - y_err, y + y_err, color=plot_props["color"],
    #                 alpha=0.05)


PLOT_PROPS = {

    "gpt-4o": {

        "color": "tab:blue",
        "linestyle": "-",
        "label": "GPT-4",
    },

    "gpt-3.5-turbo": {

        "color": "tab:green",
        "linestyle": "dashed",
        "label": "ChatGPT",
    },

    "claude": {

        "color": "tab:red",
        "linestyle": "dashdot",
        "label": "Sonnet",
    },

    "mistral": {

        "color": "tab:orange",
        "linestyle": (0, (3, 10, 1, 10)),
        "label": "Mistral",
    },

    "phi-3": {

        "color": "tab:pink",
        "linestyle": (0, (5, 10)),
        "label": "Phi",
    },

    "llama-3-8b": {

        "color": "tab:grey",
        "linestyle": "dotted",
        "label": "LLama3",
    },

    "ksat": {
        "title": "$3$--SAT$(12)$",
        "xticks": [0, 10, 20, 30, 40, 50, 60],
        "scatter_y_offset": 0,
    },

    "plogic": {
        "title": "Propositional Logic$(12)$",
        "xticks": [0, 10, 20, 30, 40],
        "scatter_y_offset": 0,
    },

    "fol": {
        "title": "First-order Logic$(8, 12)$\n(Synthetic)",
        "xticks": [0, 10, 20, 30, 35],
        "scatter_y_offset": 0.03,
    },

    "fol_human": {
        "title": "First-order Logic$(8, 12)$\n(English)",
        "xticks": [0, 10, 20, 30, 35],
        "scatter_y_offset": 0,
    },

    "regex": {
        "title": "Regular Expression$(2)$",
        "xticks": [0, 10, 20, 30, 40],
        "scatter_y_offset": 0,
    },

    "figsize": (18, 9),

    "title_fontsize": 20,
    "legend_fontsize": 20,
    "xticklabel_fontsize": 18,
    "yticklabel_fontsize": 18,
    "legend_handlelength": 3,
    "legend_bbox_anchor": (0.9, 1.03),

    "precision_bbox_anchor": (0.57, 0.25),
    "recall_bbox_anchor": (0.60, 0.39),
    "precision_legend_fontsize": 16,

    "xlabel_fontsize": 20,
    "ylabel_fontsize": 20,
    "yticks": [0, 0.25, 0.5, 0.75, 1.0]
}


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=str, default=None,
                        required=True)
    parser.add_argument("--runs", type=int, nargs="+",
                        default=range(10))
    parser.add_argument("--datasets", type=str, nargs="+",
                        default= ["plogic", "fol", "fol_human", "regex"])
    parser.add_argument("--models", type=str, nargs="+",
                        default=evaluation.SUPPORTED_MODELS)
    parser.add_argument("--clean", default=False, action="store_true")


    args = parser.parse_args()


    if args.clean:
        clean_data_dict(args.base_dir)

    data_dict = get_data_dict(args.base_dir)
    if data_dict is None:

        data_dict = parse_data(args)
        save_data_dict(args.base_dir, data_dict)

    fig = plt.Figure(figsize=PLOT_PROPS["figsize"])
    gs = GridSpec(nrows=4, ncols=len(args.datasets), figure=fig,
                  hspace=0.07, wspace=0.05)

    equiv_models = parse_equiv_data(args.base_dir)
    for i, dataset_type in enumerate(args.datasets):

        ax0 = fig.add_subplot(gs[0, i])
        ax1 = fig.add_subplot(gs[1, i])
        ax2 = fig.add_subplot(gs[2, i])
        ax3 = fig.add_subplot(gs[3, i])

        # plot_precision_recall(ax2,
        #       twin_ax2,
        #       data_dict[dataset_type]["gpt-4o"],
        #       PLOT_PROPS["gpt-4o"],
        #       scatter_y_offset=PLOT_PROPS[dataset_type]["scatter_y_offset"])
        
        # for model_name in ['gpt-3.5-turbo','llama-3-8b','mistral','phi-3']:
        #     plot_precision_recall(ax2,twin_ax2,equiv_models[dataset_type][model_name],PLOT_PROPS[model_name],scatter_y_offset=PLOT_PROPS[dataset_type]["scatter_y_offset"])

        plot_precision(ax2,
              data_dict[dataset_type]["gpt-4o"],
              PLOT_PROPS["gpt-4o"],
              scatter_y_offset=PLOT_PROPS[dataset_type]["scatter_y_offset"])
        
        plot_f1(ax3,
              data_dict[dataset_type]["gpt-4o"],
              PLOT_PROPS["gpt-4o"],
              scatter_y_offset=PLOT_PROPS[dataset_type]["scatter_y_offset"])
        
        for model_name in ['gpt-3.5-turbo','llama-3-8b','mistral','phi-3']:
            plot_precision(ax2,equiv_models[dataset_type][model_name],PLOT_PROPS[model_name],scatter_y_offset=PLOT_PROPS[dataset_type]["scatter_y_offset"])
            plot_f1(ax3,equiv_models[dataset_type][model_name],PLOT_PROPS[model_name],scatter_y_offset=PLOT_PROPS[dataset_type]["scatter_y_offset"])

        ax0.set_title(PLOT_PROPS[dataset_type]["title"],
                      fontsize=PLOT_PROPS["title_fontsize"])
        for model_name in args.models:

            plot(ax0, data_dict[dataset_type][model_name], 0,
                 PLOT_PROPS[model_name])
            ax0.set_ylim([-0.1, 1.1])
            ax0.set_xticklabels([])

            plot(ax1, data_dict[dataset_type][model_name], 1,
                 PLOT_PROPS[model_name])
            ax1.set_xticklabels([])
            ax2.set_xticklabels([])
            ax1.set_ylim([-0.1, 1.1])
            ax2.set_ylim([-0.1, 1.1])
            ax3.set_ylim([-0.1, 1.1])
            #twin_ax2.set_ylim([-0.1, 1.1])

            if i != 0:
                ax0.set_yticklabels([])
                ax1.set_yticklabels([])
                ax2.set_yticklabels([])
                ax3.set_yticklabels([])
                # if i != len(args.datasets) - 1:
                #     twin_ax2.set_yticklabels([])
                # else:
                #     twin_ax2.set_ylabel("\S A3: Specificity",
                #                         fontsize=PLOT_PROPS["ylabel_fontsize"])
            else:

                # ax2.legend(labels=["_", "Precision"], frameon=False,
                #            ncols=1,
                #            fontsize=PLOT_PROPS["precision_legend_fontsize"],
                #            bbox_to_anchor=PLOT_PROPS["precision_bbox_anchor"])
                # twin_ax2.legend(labels=["Specificity"], frameon=False,
                #            ncols=1,
                #            fontsize=PLOT_PROPS["precision_legend_fontsize"],
                #            bbox_to_anchor=PLOT_PROPS["recall_bbox_anchor"])
                # twin_ax2.set_yticklabels([])
                ax0.set_ylabel("\S A1: Syntactic\nCompliance",
                               fontsize=PLOT_PROPS["ylabel_fontsize"])
                ax1.set_ylabel("\S A2: Accuracy",
                               fontsize=PLOT_PROPS["ylabel_fontsize"])
                ax2.set_ylabel("\S A3: Precision",
                               fontsize=PLOT_PROPS["ylabel_fontsize"])
                ax3.set_ylabel("\S A3: $F_1$ Score",
                               fontsize=PLOT_PROPS["ylabel_fontsize"])

            # for ax in [ax0, ax1, ax2, twin_ax2]:
            for ax in [ax0, ax1, ax2, ax3]:
                ax.set_yticks(PLOT_PROPS["yticks"])
                ax.set_xticks(PLOT_PROPS[dataset_type]["xticks"])
                ax.tick_params(
                    axis="y", labelsize=PLOT_PROPS["yticklabel_fontsize"])
                ax.tick_params(
                    axis="x", labelsize=PLOT_PROPS["xticklabel_fontsize"])

    fig.add_artist(lines.Line2D([0.13, 0.70], [0.05, 0.05],
                                color="black"))
    fig.text(0.16, 0, "\\# of Operators: $\land, \lor, \\neg$ ($\\neg$ is counted as an operator iff not succeeded by a terminal)",
             fontsize=PLOT_PROPS["xlabel_fontsize"])


    fig.add_artist(lines.Line2D([0.715, 0.9], [0.05, 0.05],
                                color="black"))
    fig.text(0.74, 0, "CFG Parse Tree Depth",
             fontsize=PLOT_PROPS["xlabel_fontsize"])

    labels = [PLOT_PROPS[model]["label"] for model in args.models]
    fig.legend(labels=labels, frameon=False,
               ncols=6,
               bbox_to_anchor=PLOT_PROPS["legend_bbox_anchor"],
               fontsize=PLOT_PROPS["legend_fontsize"],
               handlelength=PLOT_PROPS["legend_handlelength"])

    fig.savefig("%s/plot.png" % (args.base_dir), bbox_inches="tight")
    fig.savefig("%s/results.pdf" % (args.base_dir), bbox_inches="tight")
    pass
