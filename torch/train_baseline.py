import numpy as np
from scipy.io import loadmat

from dataset.dataloader import MMwaveDataset
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from models.baseline import Baseline
from losses.losses import CharbonnierLoss
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms
import torch
import time
import os
import random
import platform
from argparse import ArgumentParser

def train(args):
    # load dataset and some pro-process steps
    data = loadmat(args.data_root)

    inputsTrain = np.array(data['inputsTrain'])
    labelsTrain = np.array(data['labelsTrain'])
    labelsTrain = labelsTrain / 255.0
    # change the data type from numpy array to torch.tensor
    inputsTrain = torch.from_numpy(inputsTrain).float()
    labelsTrain = torch.from_numpy(labelsTrain).float()

    inputsPred = np.array(data['inputsPred'])
    labelsPred = np.array(data['labelsPred'])
    labelsPred = labelsPred / 255.0
    inputsPred = torch.from_numpy(inputsPred).float()
    labelsPred = torch.from_numpy(labelsPred).float()

    TrainDataset = TensorDataset(inputsTrain, labelsTrain)
    ValDataset = TensorDataset(inputsPred, labelsPred)
    print(len(TrainDataset))
    train_dataloader = DataLoader(TrainDataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    # batch size is the input data for training at once
    val_dataloader = DataLoader(ValDataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    # empty the GPU memory
    torch.cuda.empty_cache()
    # load the baseline network from baseline.py
    model = Baseline(in_ch=1, out_ch=1)
    model = model.to(device)  # load the model to GPU for fast training

    # *** set the optimizer *** #
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    # *** set the loss function *** #
    loss_F = CharbonnierLoss()

    # *** establish folders for saving experiment results ***
    model_dir = "./%s/Baseline_total_epoch_%d_batch_size_%d_lr_%.4f" % (args.model_dir, args.end_epoch,
                                                                        args.batch_size, args.learning_rate)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    log_file_name = "./%s/Baseline_total_epoch_%d_batch_size_%d_lr_%.4f" % (args.log_dir, args.end_epoch,
                                                                            args.batch_size, args.learning_rate)

    # run directory for tensorboard information
    writer = SummaryWriter(log_file_name)

    iter = 0
    best_mae = 0
    eval_now = 1000
    # Training loop
    for epoch in range(args.start_epoch + 1, args.end_epoch + 1):
        # =====================Train============================
        model.train()
        train_epoch_loss = []
        for idx, (model_input, ground_truth) in enumerate(train_dataloader, 0):
            model_input = model_input.reshape(-1, 1, 402).to(device)   # reshape the input tensor size
            ground_truth = ground_truth.reshape(-1, 1, 256, 256).to(device)   # reshape, we can move the Flatten layer
            model_output = model(model_input)

            # cal. the loss function
            loss = loss_F(ground_truth, model_output)
            train_epoch_loss.append(loss.item())
            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            

            # write the training information into the tensorboard and visualize them
            writer.add_scalar("Train loss (iteration)", loss.item(), iter)
            # each 10 iteration, we visualize the comparison between output and label during the training in tensorboard
            if iter % 2000 == 0:
                output_input_gt = torch.cat((model_output, ground_truth), dim=0)
                grid = torchvision.utils.make_grid(output_input_gt,
                                                   scale_each=True,
                                                   nrow=args.batch_size,
                                                   normalize=True).cpu().detach().numpy()
                writer.add_image("Output_vs_gt", grid, iter)
                print("Iter %07d   Epoch %03d   epoch_loss %0.5f" %(iter, epoch, loss.item()))

            iter +=1

            # for save the best model base on MAE metric
            if iter % eval_now == 0 and iter > 0 and (epoch in [100, 300] or epoch > 400):
                model.eval()
                mae_val = []
                for model_input, ground_truth in val_dataloader:
                    model_input = model_input.reshape(-1, 1, 402).to(device)   # reshape the input tensor size
                    ground_truth = ground_truth.reshape(-1, 1, 256, 256).to(device)   # reshape, we can move the Flatten layer

                    with torch.no_grad():  # this make the inference process don't calculate the gradient of each layer
                        model_output = model(model_input)

                    mae = loss_F(ground_truth, model_output)
                    mae_val.append(mae.item())
                mae_val = np.stack(mae_val).mean()

                if mae_val > best_mae:
                    best_mae = mae_val
                    best_epoch = epoch
                    best_iter = iter
                    torch.save({"model_state_dict": model.state_dict(),
                                "optim_state_dict": optimizer.state_dict(),
                                "epoch": epoch
                                }, os.path.join(model_dir, 'model_best.pth'))
                    print("[epoch %d iter %d MAE: %.4f --- best_epoch %d best_iter %d Best_MAE %.4f]" % (
                        epoch, iter, mae_val, best_epoch, best_iter, best_mae))
            train_loss = np.stack(train_epoch_loss).mean()
            writer.add_scalars('loss in epoch ', {'train': train_loss}, epoch)

        # =====================valid============================
        model.eval()
        valid_epoch_loss = []
        start = time.time()
        for model_input, ground_truth in val_dataloader:
            model_input = model_input.reshape(-1, 1, 402).to(device)   # reshape the input tensor size
            ground_truth = ground_truth.reshape(-1, 1, 256, 256).to(device) 
            with torch.no_grad():  # this make the inference process don't calculate the gradient of each layer
                model_output = model(model_input)
            loss = loss_F(ground_truth, model_output)
            valid_epoch_loss.append(loss.item())
        print("-- Running Time:", time.time() - start)
        valid_loss = np.stack(valid_epoch_loss).mean()
        writer.add_scalars('loss in epoch ', {'valid': valid_loss}, epoch)
        model.train()

    torch.save({"model_state_dict": model.state_dict(),
                "optim_state_dict": optimizer.state_dict(),
                "epoch": epoch
                }, os.path.join(model_dir, 'model_epoch_%d_iter_%s.pth' % (epoch, iter)))

    results = {"epoch": epoch,
               "iter": iter,
               "loss": np.array(loss)}

    print(results)


if __name__ == '__main__':
    parser = ArgumentParser(description='baseline')

    parser.add_argument('--start_epoch', type=int, default=0, help='epoch number of start training')
    parser.add_argument('--end_epoch', type=int, default=100, help='epoch number of end training')
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size of training data')
    parser.add_argument('--learning_rate', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--gpu_list', type=str, default='0', help='gpu index')
    parser.add_argument('--num_workers', type=int, default=0, help='multi-process data loading with '
                                                                   'the specified number of loader worker processes')

    parser.add_argument('--model_dir', type=str, default='./checkpoint', help='trained or pre-trained model directory')
    parser.add_argument('--data_root', type=str, default='data/activations_USAF1951202301111510.mat',
                        help='training data directory')
    parser.add_argument('--log_dir', type=str, default='./log', help='log directory')

    args = parser.parse_args()
    print(args)
    train(args=args)




