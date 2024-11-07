import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from models import DnCNN
from dataset import prepare_data, Dataset
from utils import *
import torch
import numpy as np
from torchvision.transforms import transforms
import numpy as np
import random
from skimage.util import random_noise
import torch
import random
from skimage.util import random_noise


def add_noise(img_tensor):
    # Ensure img_tensor is float32 and on the correct device
    img_tensor = img_tensor.to(torch.float32)
    device = img_tensor.device

    noise_t = random.randint(0, 6)
    if noise_t == 0:
        # Gaussian noise
        gaussian_noise = (
            torch.tensor(
                random_noise(
                    img_tensor.cpu().numpy(),
                    mode="gaussian",
                    mean=0,
                    var=0.05,
                    clip=True,
                )
            ).to(device, dtype=torch.float32)
            - img_tensor
        )
        noise_tensor = gaussian_noise

    elif noise_t == 1:
        # Gaussian + Salt noise
        img_gaussian = torch.tensor(
            random_noise(
                img_tensor.cpu().numpy(), mode="gaussian", mean=0, var=0.05, clip=True
            )
        ).to(device, dtype=torch.float32)
        img_gaussian_salt = torch.tensor(
            random_noise(
                img_gaussian.cpu().numpy(), mode="salt", amount=0.05, clip=True
            )
        ).to(device, dtype=torch.float32)
        noise_tensor = img_gaussian_salt - img_tensor

    elif noise_t == 2:
        # Speckle noise
        speckle_noise = torch.normal(
            mean=0, std=0.05**0.5, size=img_tensor.size(), device=device
        )
        noise_tensor = img_tensor * speckle_noise

    elif noise_t == 3:
        # Poisson noise
        poisson_noise = (
            torch.tensor(
                random_noise(img_tensor.cpu().numpy(), mode="poisson", clip=True)
            ).to(device, dtype=torch.float32)
            - img_tensor
        )
        noise_tensor = poisson_noise

    elif noise_t == 4:
        # Salt noise
        salt_noise = (
            torch.tensor(
                random_noise(img_tensor.cpu().numpy(), mode="salt", amount=0.05)
            ).to(device, dtype=torch.float32)
            - img_tensor
        )
        noise_tensor = salt_noise

    elif noise_t == 5:
        # Speckle noise (Gaussian multiplicative)
        speckle_noise = (
            torch.tensor(
                random_noise(
                    img_tensor.cpu().numpy(),
                    mode="speckle",
                    mean=0,
                    var=0.05,
                    clip=True,
                )
            ).to(device, dtype=torch.float32)
            - img_tensor
        )
        noise_tensor = speckle_noise

    elif noise_t == 6:
        # Speckle + Salt noise
        speckle_noise = torch.normal(
            mean=0, std=0.01**0.5, size=img_tensor.size(), device=device
        )
        speckle_component = img_tensor * speckle_noise
        img_with_speckle = img_tensor + speckle_component
        noisy_img_np = random_noise(
            img_with_speckle.cpu().numpy(), mode="salt", amount=0.05
        )
        noise_tensor = (
            torch.tensor(noisy_img_np).to(device, dtype=torch.float32) - img_tensor
        )

    return noise_tensor



# Set visible devices to use all 8 GPUs
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,5,6"  # Use all GPUs

parser = argparse.ArgumentParser(description="DnCNN")
parser.add_argument(
    "--preprocess", type=bool, default=False, help="run prepare_data or not"
)
parser.add_argument("--batchSize", type=int, default=512, help="Training batch size")
parser.add_argument(
    "--num_of_layers", type=int, default=17, help="Number of total layers"
)
parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
parser.add_argument(
    "--milestone",
    type=int,
    default=1,
    help="When to decay learning rate; should be less than epochs",
)
parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
parser.add_argument("--outf", type=str, default="logs", help="path of log files")
parser.add_argument(
    "--mode",
    type=str,
    default="S",
    help="with known noise level (S) or blind training (B)",
)
parser.add_argument(
    "--noiseL", type=float, default=25, help="noise level; ignored when mode=B"
)
parser.add_argument(
    "--val_noiseL", type=float, default=25, help="noise level used on validation set"
)
opt = parser.parse_args()


def main():
    # Load dataset
    print("Loading dataset ...\n")
    dataset_train = Dataset(train=True)
    dataset_val = Dataset(train=False)
    loader_train = DataLoader(
        dataset=dataset_train, num_workers=12, batch_size=opt.batchSize, shuffle=True
    )
    print("# of training samples: %d\n" % int(len(dataset_train)))

    # Build model
    net = DnCNN(channels=1, num_of_layers=opt.num_of_layers)

    net.apply(weights_init_kaiming)
    criterion = nn.MSELoss(reduction="mean")

    # Move to GPU
    device_ids = list(
        range(torch.cuda.device_count())
    )  # Automatically get all available GPUs
    model = nn.DataParallel(net, device_ids=device_ids).cuda()  # Use all available GPUs
    criterion.cuda()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    # TensorBoard writer
    writer = SummaryWriter(opt.outf)
    step = 0
    noiseL_B = [0, 55]  # Ignored when opt.mode=='S'

    for epoch in range(opt.epochs):
        if epoch < opt.milestone:
            current_lr = opt.lr
        else:
            current_lr = opt.lr / 10.0

        # Set learning rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        print("learning rate %f" % current_lr)

        # Train
        for i, data in enumerate(loader_train, 0):
            # Training step
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            img_train = data

            if opt.mode == "S":
                noise = torch.FloatTensor(img_train.size()).normal_(
                    mean=0, std=opt.noiseL / 255.0
                )
            if opt.mode == "B":
                noise = torch.zeros(img_train.size())
                stdN = np.random.uniform(noiseL_B[0], noiseL_B[1], size=noise.size()[0])
                for n in range(noise.size()[0]):
                    sizeN = noise[0, :, :, :].size()
                    noise[n, :, :, :] = torch.FloatTensor(sizeN).normal_(
                        mean=0, std=stdN[n] / 255.0
                    )
            if opt.mode == "K":
                noise = add_noise(img_train)

            imgn_train = img_train + noise
            img_train, imgn_train = Variable(img_train.cuda()), Variable(
                imgn_train.cuda()
            )

            noise = Variable(noise.cuda())
            out_train = model(imgn_train)
            loss = criterion(out_train, noise) / (imgn_train.size()[0] * 2)
            loss.backward()
            optimizer.step()

            # Results
            model.eval()
            out_train = torch.clamp(imgn_train - model(imgn_train), 0.0, 1.0)
            psnr_train = batch_PSNR(out_train, img_train, 1.0)
            print(
                "[epoch %d][%d/%d] loss: %.4f PSNR_train: %.4f"
                % (epoch + 1, i + 1, len(loader_train), loss.item(), psnr_train)
            )

            if step % 10 == 0:
                # Log the scalar values
                writer.add_scalar("loss", loss.item(), step)
                writer.add_scalar("PSNR on training data", psnr_train, step)
            step += 1

        # End of each epoch
        # Save model
        torch.save(model.state_dict(), os.path.join(opt.outf, "net_30.pth"))

        model.eval()
        # Validate
        psnr_val = 0
        for k in range(len(dataset_val)):
            img_val = torch.unsqueeze(dataset_val[k], 0)
            noise = torch.FloatTensor(img_val.size()).normal_(
                mean=0, std=opt.val_noiseL / 255.0
            )
            imgn_val = img_val + noise
            img_val, imgn_val = Variable(img_val.cuda(), volatile=True), Variable(
                imgn_val.cuda(), volatile=True
            )
            out_val = torch.clamp(imgn_val - model(imgn_val), 0.0, 1.0)
            psnr_val += batch_PSNR(out_val, img_val, 1.0)

        psnr_val /= len(dataset_val)
        print("\n[epoch %d] PSNR_val: %.4f" % (epoch + 1, psnr_val))
        writer.add_scalar("PSNR on validation data", psnr_val, epoch)

        # Log the images
        out_train = torch.clamp(imgn_train - model(imgn_train), 0.0, 1.0)
        Img = utils.make_grid(img_train.data, nrow=8, normalize=True, scale_each=True)
        Imgn = utils.make_grid(imgn_train.data, nrow=8, normalize=True, scale_each=True)
        Irecon = utils.make_grid(
            out_train.data, nrow=8, normalize=True, scale_each=True
        )
        writer.add_image("clean image", Img, epoch)
        writer.add_image("noisy image", Imgn, epoch)
        writer.add_image("reconstructed image", Irecon, epoch)


if __name__ == "__main__":
    if opt.preprocess:
        if opt.mode == "S":
            prepare_data(data_path="data", patch_size=40, stride=10, aug_times=1)
        if opt.mode == "B":
            prepare_data(data_path="data", patch_size=50, stride=10, aug_times=2)
        if opt.mode == "K":
            prepare_data(data_path="data", patch_size=50, stride=10, aug_times=2)
    main()
