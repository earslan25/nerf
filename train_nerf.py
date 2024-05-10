import torch
import argparse
import json
import numpy as np
import imageio
import os

from preprocess import Preprocessor
from preprocess_folder import PreprocessorFolder
from utils import visualize, utils
from renderer.nerf_renderer import NRFRenderer
from tqdm import tqdm

def parsarguments():
    parser = argparse.ArgumentParser(description="nerf")
    parser.add_argument(
        "--config", type=str, default="configs/default.json", help="Config file"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=15
    )
    parser.add_argument(
        "--data_path", type=str, default="data/tiny_nerf_data.npz"
    )
    parser.add_argument(
        "--out_dir", type=str, default="out/tiny_nerf_data"
    )
    parser.add_argument(
        "--model_path", type=str, default=None, help="trained model"
    )
    parser.add_argument(
        "--metric", type=str, default=["psnr", "ssim"], nargs="*"
    )
    args = parser.parse_args()

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
    if os.path.isfile(args.data_path):
        preprocessor = Preprocessor()
        preprocessor.load_new_data(args.data_path)
        split_params = (True, 100)
        train_images, train_poses, test_images, test_poses = preprocessor.split_data(split_params, randomize=False)
    else:
        preprocessor = PreprocessorFolder()
        train_images, train_poses = preprocessor.load_train_data(args.data_path)

    # Set metric
    metric_fns = {}
    metric_dict = {}
    if "psnr" in args.metric:
        metric_dict["psnr"] = 0
        metric_fns["psnr"] = utils.psnr
    if "ssim" in args.metric:
        metric_dict["ssim"] = 0
        metric_fns["ssim"] = utils.ssim
    else:
        raise ValueError(f"Unsupported metric '{args.metric}'. Expected 'psnr' or 'ssim'.")

    NerfRenderer = NRFRenderer().to(device)
    optimizer = torch.optim.Adam(NerfRenderer.parameters(), lr=1e-3)
    start_epoch = 0

    # Load saved model
    if args.model_path:
        checkpoint = torch.load(args.model_path, map_location=device)
        NerfRenderer.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Continue training from epoch {start_epoch}")

    NerfRenderer.train()
    for epoch in tqdm(range(start_epoch, args.num_epochs)):
        for img, pose in zip(train_images, train_poses):
            optimizer.zero_grad()

            rays = NerfRenderer.get_rays(preprocessor.H, preprocessor.W, preprocessor.focal, pose.to(device))
            rgb, depth = NerfRenderer(rays)

            loss = utils.mse_loss(rgb, img.to(device))

            loss.backward()

            optimizer.step()

        print(f'Epoch {epoch}: Loss: {loss.item()}')

    # Save model for training mode
    utils.save_model(args.num_epochs - 1, NerfRenderer, optimizer, os.path.join(args.out_dir, 'ckpt', "model.ckpt"))

    # test
    if os.path.isdir(args.data_path):
        test_images, test_poses = preprocessor.load_test_data(args.data_path)

    NerfRenderer.eval()
    total_metric = 0
    for idx, (img, pose) in enumerate(zip(test_images, test_poses)):
        rays = NerfRenderer.get_rays(preprocessor.H, preprocessor.W, preprocessor.focal, pose.to(device))
        with torch.no_grad():
            rgb, depth = NerfRenderer(rays)
        visualize.save_result_comparison(rgb.detach().cpu().numpy(), img.numpy(), os.path.join(args.out_dir, 'test_results', f"{idx}.jpg"))

        rgb = torch.permute(rgb.unsqueeze(0), (0, 3, 1, 2))
        img = torch.permute(img.unsqueeze(0), (0, 3, 1, 2))
        for metric_name, metric_fn in metric_fns.items():
            metric_score = metric_fn(rgb, img.to(device))
            metric_dict[metric_name] += metric_score

    score_string = ""
    for name, score in metric_dict.items():
        score_string += " " + name + ": " + "{:.2f}".format(score.item() / (idx+1)) 

    print(f'Final Validation: {score_string}')








