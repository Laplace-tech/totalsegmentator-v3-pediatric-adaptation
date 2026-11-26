"""Rebuild the report's CT illustrations and M0 learning curves from released evidence."""
import argparse
import json
import sqlite3
from collections import Counter
from contextlib import closing
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.ndimage import gaussian_filter
from skimage.measure import marching_cubes

from tsv3.data import DEFAULT_RAW, ROOT
from tsv3.evaluation import MODELS, dice_from_counts, evaluation_inputs, prepare_test_case
from tsv3.training import build_model, infer_volume

ADRENAL = "adrenal_gland_left"
COLORS = {"liver": "#e39865", "spleen": "#bd82dc", "kidney_left": "#4bb4ff",
          "kidney_right": "#56ce9d", "pancreas": "#ffe16a",
          ADRENAL: "#ff7a91", "adrenal_gland_right": "#64ebed"}
NAMES = ("Liver", "Spleen", "Left kidney", "Right kidney", "Pancreas", "Left adrenal", "Right adrenal")
REFERENCE, PREDICTION = "#52eee3", "#ffab53"
DARK = "#101923"


def median_case(db, rows, cohort, seed):
    comparator = "M2" if cohort == "pediatric" else "M0"
    ranked = []
    for row in rows:
        if (row["cohort"], row["label"], row["primary_eligible"]) != (cohort, ADRENAL, "1"):
            continue
        records = db.execute("SELECT model,result FROM scores WHERE seed=? AND case_id=?", (seed, row["case_id"]))
        scores = {model: json.loads(result)["organs"] for model, result in records}
        delta = scores["M1"][ADRENAL]["Dice"] - scores[comparator][ADRENAL]["Dice"]
        ranked.append((delta, row["case_id"], scores))
    ranked.sort()
    return ranked[len(ranked) // 2][1:]


def show_ct(ax, plane):
    ax.imshow(plane.T, origin="lower", cmap="gray", vmin=-150, vmax=250, interpolation="nearest")
    ax.set_axis_off()


def save(fig, output, name):
    fig.savefig(output / name, dpi=190, bbox_inches="tight", pad_inches=0.12, facecolor=fig.get_facecolor())
    plt.close(fig)


def overview(ct, prediction, reference, labels, case, seed, output):
    # Center the coronal view on the reference left kidney, independently of the prediction.
    y = int(np.argmax(reference["kidney_left"].sum(axis=(0, 2))))
    fig = plt.figure(figsize=(12, 5.1), facecolor=DARK)
    grid = fig.add_gridspec(1, 3, left=.02, right=.99, bottom=.14, top=.85, wspace=.025)
    for col, name in enumerate(("Coronal CT", "M1 segmentation", "3D prediction · posterior view")):
        fig.text((col + .5) / 3, .866, name, color="white", ha="center", fontsize=13, weight="bold")
    foreground = np.logical_or.reduce(list(reference.values()))
    coords = np.array(np.nonzero(foreground))
    low, high = coords.min(axis=1), coords.max(axis=1)
    for col in (0, 1):
        ax = fig.add_subplot(grid[col])
        show_ct(ax, ct[:, y, :])
        if col:
            rgba = np.zeros((*prediction[:, y, :].T.shape, 4))
            for i, label in enumerate(labels, 1):
                rgba[prediction[:, y, :].T == i] = matplotlib.colors.to_rgba(COLORS[label], .63)
            ax.imshow(rgba, origin="lower", interpolation="nearest")
        ax.set(xlim=(high[0]+18, low[0]-18), ylim=(max(0, low[2]-12), min(ct.shape[2]-1, high[2]+12)))
        ax.text(.03, .95, "R", transform=ax.transAxes, color="white", weight="bold", va="top")
    ax = fig.add_subplot(grid[2], projection="3d", facecolor=DARK)
    ax.set_axis_off()
    triangles, shades = [], []
    light = np.array([-.45, -.55, .70])
    for i, label in enumerate(labels, 1):
        mask = prediction == i
        xyz = np.array(np.nonzero(mask))
        lo, hi = xyz.min(axis=1), xyz.max(axis=1) + 1
        region = mask[tuple(slice(a, b) for a, b in zip(lo, hi))]
        padded = np.pad(region.astype(np.float32), 2)
        vertices, faces, _, _ = marching_cubes(padded, .5)
        # Smooth the display normals only. The triangle positions stay on the binary mask surface.
        gradient = np.gradient(gaussian_filter(padded, 1.1))
        index = tuple(np.rint(vertices).astype(int).T)
        normals = -np.stack([g[index] for g in gradient], axis=1)
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8)
        intensity = .67 + .33 * (normals[faces].mean(axis=1) @ light)
        triangles.append((vertices + lo - 2)[faces])
        shades.append(intensity[:, None] * np.array(matplotlib.colors.to_rgb(COLORS[label])))
    # Sort faces from all organs together for correct occlusion between adjoining surfaces.
    ax.add_collection3d(Poly3DCollection(np.concatenate(triangles), facecolors=np.concatenate(shades),
                                         linewidths=0, antialiased=False))
    ax.set(xlim=(low[0]-2, high[0]+2), ylim=(low[1]-2, high[1]+2), zlim=(low[2]-2, high[2]+2))
    ax.set_box_aspect(high - low + 5, zoom=1.10)
    ax.view_init(elev=18, azim=-72)
    fig.text(.02, .962, "Seven abdominal organs on pediatric CT", color="white", fontsize=20, weight="bold")
    fig.text(.98, .961, f"{case}  /  M1  /  seed {seed}", color="#c9d5e2", ha="right", fontsize=10)
    fig.legend(handles=[Patch(color=COLORS[k], label=v) for k, v in zip(labels, NAMES)],
               loc="lower center", ncol=7, frameon=False, labelcolor="white", fontsize=10,
               columnspacing=1.2, handlelength=1.1, bbox_to_anchor=(.5, .028))
    save(fig, output, "pediatric_ct_segmentation.png")


def adrenal_row(axes, ct, reference, masks, dice, case, cohort):
    z = int(np.argmax(reference.sum(axis=(0, 1))))
    plane = reference[:, :, z]
    x, y = np.array(np.nonzero(plane)).mean(axis=1)
    half = 20
    for col, ax in enumerate(axes):
        show_ct(ax, ct[:, :, z])
        if col == 0:
            ax.add_patch(Rectangle((x-half, y-half), 2*half, 2*half, fill=False, ec="white", lw=1.1))
            ax.text(.03, .96, "R", color="white", transform=ax.transAxes, va="top", weight="bold")
            ax.set_title(f"{cohort.title()}  ·  {case}", fontsize=12, color="white", loc="left")
        else:
            model = MODELS[col - 1]
            for contour, color, style in [(plane, REFERENCE, "solid"), (masks[model], PREDICTION, "dashed")]:
                if contour.any():
                    ax.contour(contour.T.astype(float), levels=[.5], colors=[color], linewidths=1.65,
                               linestyles=style, origin=None)
            ax.set(xlim=(x-half, x+half), ylim=(y-half, y+half))
            ax.set_title(f"{model}    Dice {dice[model]:.3f}", fontsize=12, color="white")
        ax.invert_xaxis()


def ct_figures(args):
    plan, _, norm, contract, rows, checkpoints, identity = evaluation_inputs(args)
    seed = plan["study"]["training"]["replication_plan"]["seeds"][0]
    labels = plan["labels"]
    torch.set_num_threads(checkpoints[seed, "M0"][1]["cpu_threads"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    fig, axes = plt.subplots(2, 4, figsize=(11, 6.3), facecolor=DARK)
    fig.subplots_adjust(left=.02, right=.985, top=.84, bottom=.08, hspace=.26, wspace=.06)
    fig.text(.02, .95, "Left adrenal segmentation", color="white", fontsize=21, weight="bold")
    fig.legend(handles=[Line2D([0], [0], color=REFERENCE, label="Public reference"),
                        Line2D([0], [0], color=PREDICTION, ls="--", label="Model prediction")],
               loc="upper right", bbox_to_anchor=(.99, .992), ncol=2, frameon=False, labelcolor="white")
    score_file = ROOT / "outputs/test_abdominal_v1/scores.sqlite"
    with closing(sqlite3.connect(f"{score_file.as_uri()}?mode=ro", uri=True)) as db:
        saved_identity = json.loads(db.execute("SELECT identity FROM run").fetchone()[0])
        if any(saved_identity[k] != v for k, v in identity.items()):
            raise ValueError("Score database differs from the frozen evaluation inputs")
        selections = []
        for cohort in ("pediatric", "adult"):
            case, scores = median_case(db, rows, cohort, seed)
            selections.append((cohort, case, scores))
        counts = Counter(r["case_id"] for r in rows if r["cohort"] == "pediatric" and r["primary_eligible"] == "1")
        complete = []
        for case, count in counts.items():
            if count == len(labels):
                row = db.execute("SELECT result FROM scores WHERE seed=? AND case_id=? AND model='M1'",
                                 (seed, case)).fetchone()
                scores = json.loads(row[0])["organs"]
                complete.append((np.mean([scores[k]["Dice"] for k in labels]), case, scores))
        complete.sort()
        _, case, scores = complete[len(complete) // 2]
        selections.append(("overview", case, {"M1": scores}))
        for i, (cohort, case, scores) in enumerate(selections):
            data, inverse, shape, refs = prepare_test_case(args.raw, case, labels, contract["raw_sources"], norm)
            source = nib.load(args.raw / case / "ct.nii.gz")
            ct = np.asanyarray(nib.as_closest_canonical(source).dataobj)
            to_ras = nib.orientations.ornt_transform(nib.orientations.io_orientation(source.affine),
                                                     nib.orientations.axcodes2ornt(("R", "A", "S")))
            reference = {k: nib.orientations.apply_orientation(v, to_ras) for k, v in refs.items()}
            z = int(np.argmax(reference[ADRENAL].sum(axis=(0, 1))))
            masks, adrenal_dice = {}, {}
            for branch in scores:
                path, config, _ = checkpoints[seed, branch]
                model = build_model(config, len(labels) + 1)
                model.load_state_dict(torch.load(path, weights_only=False, map_location="cpu")["model"])
                pred, _ = infer_volume(model, data, labels, config)
                native = nib.orientations.apply_orientation(pred, inverse)
                if native.shape != shape:
                    raise ValueError(f"Unexpected native grid for {case}")
                differences = []
                for idx, label in enumerate(labels, 1):
                    mask = native == idx
                    score = scores[branch][label]
                    dice = float(dice_from_counts(np.count_nonzero(mask & refs[label]), refs[label].sum(), mask.sum()))
                    differences.append(abs(dice - score["Dice"]))
                    if refs[label].sum() != score["reference_voxels"] or not np.isclose(
                            dice, score["Dice"], rtol=0, atol=.0005):
                        raise ValueError(f"Reference volume or reproduced Dice differs: {case}/{branch}/{label}")
                    if label == ADRENAL:
                        adrenal_dice[branch] = dice
                print(f"{case}/{branch} max absolute Dice deviation: {max(differences):.8f}", flush=True)
                masks[branch] = pred[:, :, z] == labels.index(ADRENAL) + 1
                if cohort == "overview":
                    overview(ct, pred, reference, labels, case, seed, args.output)
            if cohort != "overview":
                adrenal_row(axes[i], ct, reference[ADRENAL], masks, adrenal_dice, case, cohort)
    fig.text(.5, .025, f"Same axial slice and crop within each row   /   whole-volume Dice   /   seed {seed}",
             ha="center", color="#c9d5e2", fontsize=10)
    save(fig, args.output, "left_adrenal_test_examples.png")


def learning_curves(args):
    study = json.loads(args.study.read_text())
    seeds = study["training"]["replication_plan"]["seeds"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9), layout="constrained", sharey=True)
    for seed in seeds:
        folder = ROOT / f"outputs/training_{study['study_version']}_seed{seed}/M0"
        records = {int(p.stem.split("_")[-1]): json.loads(p.read_text())["adult_validation"]
                   for p in sorted(folder.glob("validation_*.json"))}
        selected = json.loads((folder / "status.json").read_text())["trainer_state"]["best_update"]
        x = np.array(list(records)) / 1000
        line, = axes[0].plot(x, [r["macro_Dice"] for r in records.values()], label=str(seed), lw=1.6)
        axes[0].scatter(selected/1000, records[selected]["macro_Dice"], color=line.get_color(),
                        s=42, marker="D", edgecolor="white", linewidth=.5, zorder=4)
        if seed == 20261101:
            for idx, name, style in [(-2, "Left adrenal", "-"), (-1, "Right adrenal", "--")]:
                axes[1].plot(x, [r["organ_mean_Dice"][idx] for r in records.values()], style, label=name)
            axes[1].axvline(selected/1000, color="#6a6a6a", lw=1, ls=":", label="Selected M0")
    for ax, title in zip(axes, ["A   Seven-organ macro Dice", "B   Adrenal Dice · seed 20261101"]):
        ax.set_title(title, loc="left", weight="bold")
        ax.set(xlabel="Successful updates (thousands)", ylim=(-.04, 1.02))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.2)
        ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(.5, -.17), ncol=3)
    axes[0].set_ylabel("Adult validation Dice")
    save(fig, args.output, "m0_validation_curves.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/figures")
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--study", type=Path, default=ROOT / "configs/study.json")
    parser.add_argument("--review", type=Path, default=ROOT / "manifests/review_v300")
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True, parents=True)
    ct_figures(args)
    learning_curves(args)
