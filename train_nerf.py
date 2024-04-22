import torch

from preprocess import Preprocessor
from utils import visualize, utils
from renderer.nerf_renderer import NRFRenderer
from tqdm import tqdm

if __name__ == '__main__':
    # TODO add argparse
    device = utils.get_device()
    print("Using device: ", device)

    path = 'data/tiny_nerf_data.npz'
    preprocessor = Preprocessor()
    preprocessor.load_new_data(path)
    split_params = (True, 100)
    train_images, train_poses, test_images, test_poses = preprocessor.split_data(split_params, randomize=False)
    # visualize.plot_images(test_images[1].numpy())

    pred = test_images[1]
    target = train_images[1]

    metric = utils.psnr
    # print(utils.apply_metric(pred, target, metric))

    num_epochs = 250

    NerfRenderer = NRFRenderer().to(device)
    NerfRenderer.train()

    optimizer = torch.optim.Adam(NerfRenderer.parameters(), lr=1e-3)

    for epoch in tqdm(range(num_epochs)):
        optimizer.zero_grad()

        rays = NerfRenderer.get_rays(preprocessor.H, preprocessor.W, preprocessor.focal, train_poses[0].to(device))

        rgb, depth = NerfRenderer(rays)

        loss = utils.mse_loss(rgb, train_images[0].to(device))

        loss.backward()

        # for param in NerfRenderer.parameters():
        #     print(param.grad)

        optimizer.step()

        if epoch % 10 == 0:
            print('')
            print(f'Epoch {epoch + 1}: Loss: {loss.item()}')
            print(f'PSNR: {utils.apply_metric(rgb, train_images[0].to(device), utils.psnr)}')

            visualize.plot_images(rgb.detach().cpu().numpy())
            # visualize.plot_images(train_images[0].numpy())








