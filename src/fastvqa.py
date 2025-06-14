import argparse
import json
import pathlib

import decord
import numpy as np
import torch
import tqdm
import yaml

from fastvqa.datasets import FragmentSampleFrames, SampleFrames, get_spatial_fragments
from fastvqa.models import DiViDeAddEvaluator

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sigmoid_rescale(score, model="FasterVQA"):
    mean, std = (-0.110198185, 0.04178565)
    x = (score - mean) / std
    print(f"Inferring with model [{model}]:")
    score = 1 / (1 + np.exp(-x))
    return score


def evaluate_video(
    video_path: pathlib.Path,
    opt: dict,
    evaluator: DiViDeAddEvaluator,
) -> float:
    video_reader = decord.VideoReader(str(video_path))
    vsamples = {}
    t_data_opt = opt["data"]["val-kv1k"]["args"]
    s_data_opt = opt["data"]["val-kv1k"]["args"]["sample_types"]
    for sample_type, sample_args in s_data_opt.items():
        ## Sample Temporally
        if t_data_opt.get("t_frag", 1) > 1:
            sampler = FragmentSampleFrames(
                fsize_t=sample_args["clip_len"] // sample_args.get("t_frag", 1),
                fragments_t=sample_args.get("t_frag", 1),
                num_clips=sample_args.get("num_clips", 1),
            )
        else:
            sampler = SampleFrames(
                clip_len=sample_args["clip_len"], num_clips=sample_args["num_clips"]
            )

        num_clips = sample_args.get("num_clips", 1)
        frames = sampler(len(video_reader))
        frame_dict = {idx: video_reader[idx] for idx in np.unique(frames)}
        imgs = [frame_dict[idx] for idx in frames]
        video = torch.stack(imgs, 0)
        video = video.permute(3, 0, 1, 2)

        ## Sample Spatially
        sampled_video = get_spatial_fragments(video, **sample_args)
        mean, std = (
            torch.FloatTensor([123.675, 116.28, 103.53]),
            torch.FloatTensor([58.395, 57.12, 57.375]),
        )
        sampled_video = ((sampled_video.permute(1, 2, 3, 0) - mean) / std).permute(
            3, 0, 1, 2
        )

        sampled_video = sampled_video.reshape(
            sampled_video.shape[0], num_clips, -1, *sampled_video.shape[2:]
        ).transpose(0, 1)
        vsamples[sample_type] = sampled_video.to(DEVICE)
    result = evaluator(vsamples)
    score = sigmoid_rescale(result.mean().item(), model=args.model)
    return score


opts = {
    "FasterVQA": "./options/fast/f3dvqa-b.yml",
    "FasterVQA-MS": "./options/fast/fastervqa-ms.yml",
    "FasterVQA-MT": "./options/fast/fastervqa-mt.yml",
    "FAST-VQA": "./options/fast/fast-b.yml",
    "FAST-VQA-M": "./options/fast/fast-m.yml",
}

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "-m",
        "--model",
        type=pathlib.Path,
        required=True,
    )
    argparser.add_argument(
        "-o",
        "--opt",
        type=pathlib.Path,
        required=True,
    )
    argparser.add_argument(
        "-r",
        "--refdir",
        type=pathlib.Path,
        required=True,
    )
    argparser.add_argument(
        "-t",
        "--testdir",
        type=pathlib.Path,
        required=True,
    )

    args = argparser.parse_args()

    with open(args.opt, "r") as f:
        opt = yaml.safe_load(f)

    ### Model Definition
    evaluator = DiViDeAddEvaluator(**opt["model"]["args"]).to(DEVICE)
    evaluator.load_state_dict(torch.load(args.model, map_location=DEVICE)["state_dict"])

    ref_vids = sorted(args.refdir.glob("*.mp4"))
    print(f"Found {len(list(ref_vids))} reference videos.")
    scores = {
        "original": {},
    }
    for ref_vid in tqdm.tqdm(ref_vids, desc="Processing reference videos"):
        ref_score = evaluate_video(ref_vid, opt, evaluator)
        scores["original"][ref_vid.stem] = ref_score

    test_vids = sorted(args.testdir.glob("**/reconstructed.mp4"))
    print(f"Found {len(list(test_vids))} test videos.")
    for test_vid in tqdm.tqdm(test_vids, desc="Processing test videos"):
        rec_score = evaluate_video(test_vid, opt, evaluator)
        video = test_vid.parent.stem
        model = test_vid.parent.parent.stem
        if model not in scores:
            scores[model] = {}
        scores[model][video] = rec_score

    with open(args.testdir / "fastvqascores.json", "w") as f:
        json.dump(scores, f, indent=4)
