import torch
import argparse
import json
import numpy as np
import imageio
import os

from preprocess import Preprocessor
from utils import visualize, utils
from renderer.nerf_renderer import NRFRenderer
from tqdm import tqdm

def parsarguments():
    parser = argparse.ArgumentParser(description="nerf")
    parser.add_argument(
        "--config", type=str, default="configs/default.json", help="Config file"
    )
    parser.add_argument(
        "--data_path", type=str, default="data/tiny_nerf_data.npz"
    )
    parser.add_argument(
        "--out_dir", type=str, default="out/tiny_nerf_data"
    )
    parser.add_argument(
        "--model_path", type=str, default=None
    )
    parser.add_argument(
        "--metric", type=str, default=["psnr", "ssim"], nargs="*"
    )
    args = parser.parse_args()
    print(args.metric)
    return args


if __name__ == '__main__':
    device = utils.get_device()
    print("Using device: ", device)

    args = parsarguments()
    
    if args.config is not None:
        config = json.load(open(args.config, "r"))
        for key in config:
            args.__dict__[key] = config[key]
 
    os.makedirs(os.path.join(args.out_dir, 'ckpt'), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, 'test_results'), exist_ok=True)

    # ---------------------------------------------------------------------

    # Load Data
    preprocessor = Preprocessor()
    preprocessor.load_new_data(args.data_path)
    split_params = (True, 100)
    train_images, train_poses, test_images, test_poses = preprocessor.split_data(split_params, randomize=False)
    # visualize.plot_images(test_images[1].numpy())

    # Set metric
    metric_fns = {}
    metric_dict = {}
    if "psnr" in args.metric:
        metric_dict["psnr"] = 0
        metric_fns["psnr"] = utils.psnr
    if "ssim" in args.metric:
        metric_dict["ssim"] = 0
        metric_fns["ssim"] = utils.ssim
    #else:
        #raise ValueError(f"Unsupported metric '{args.metric}'. Expected 'psnr' or 'ssim'.")

    # pred = test_images[1]
    # target = train_images[1]
    # print(utils.apply_metric(pred, target, metric))

    NerfRenderer = NRFRenderer().to(device)
    optimizer = torch.optim.Adam(NerfRenderer.parameters(), lr=1e-3)
    start_epoch = 0

    # Load saved model
    if hasattr(args, 'model_path'):
        checkpoint = torch.load(args.model_path, map_location=device)
        NerfRenderer.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
    else:
        print("Model path required for testing")
        assert(False)

    # test
    NerfRenderer.eval()
    total_metric = 0
    for idx, (img, pose) in enumerate(zip(test_images, test_poses)):
        rays = NerfRenderer.get_rays(preprocessor.H, preprocessor.W, preprocessor.focal, pose.to(device))
        rgb, depth = NerfRenderer(rays)
        visualize.save_result_comparison(rgb.detach().cpu().numpy(), img.numpy(), os.path.join(args.out_dir, 'test_results', f"{idx}.jpg"))

        rgb = torch.permute(rgb.unsqueeze(0), (0, 3, 1, 2))
        img = torch.permute(img.unsqueeze(0), (0, 3, 1, 2))
        for metric_name, metric_fn in metric_fns.items():
            metric_score = metric_fn(rgb,img)
            metric_dict[metric_name] += metric_score

    score_string = ""
    for name, score in metric_dict.items():
        score_string += " " + name + ": " + "{:.2f}".format(score.item() / (idx+1)) 

    print(f'Final Validation: {score_string}')








