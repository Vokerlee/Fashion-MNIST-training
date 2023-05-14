import numpy as np
import imageio
import matplotlib
import os
import sys
import shutil

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

from matplotlib import pyplot as plt
from matplotlib.gridspec  import GridSpec
from matplotlib.ticker    import MaxNLocator

from collections import defaultdict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import FashionMNIST

matplotlib.rcParams['axes.formatter.limits'] = (-5, 4)

MAX_LOCATOR_NUMBER = 10
FIGURE_XSIZE = 10
FIGURE_YSIZE = 8

BACKGROUND_RGB = '#F5F5F5'
MAJOR_GRID_RGB = '#919191'

LEGEND_FRAME_ALPHA = 0.95

def set_axis_properties(axes):
    axes.xaxis.set_major_locator(MaxNLocator(MAX_LOCATOR_NUMBER))
    axes.minorticks_on()
    axes.grid(which='major', linewidth=2, color=MAJOR_GRID_RGB)
    axes.grid(which='minor', linestyle=':')

class ConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out, stride):
        super().__init__()
        self.conv = nn.Conv2d(ch_in, ch_out,
                              kernel_size=(3, 3), stride=stride)
        self.bn = nn.BatchNorm2d(ch_out)
        self.relu = nn.ReLU()

    def forward(self, input):
        conv_out = self.conv(input)
        batch_norm_out = self.bn(conv_out)
        return self.relu(batch_norm_out)

class NeuralNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        layer_config = ((64, 2), (64, 1), (128, 2), (128, 1))

        ch_in = 1
        block_list = []
        for ch_out, stride in layer_config:
            block = ConvBlock(ch_in, ch_out, stride)
            block_list.append(block)
            ch_in = ch_out

        self.backbone = nn.Sequential(*block_list)

        bottleneck_channel = 2
        self.bottleneck = nn.Linear(layer_config[-1][0], bottleneck_channel)
        self.head       = nn.Linear(bottleneck_channel, num_classes)

        self.softmax = nn.Softmax()

    def forward(self, input):
        featuremap = self.backbone(input)
        squashed = F.adaptive_avg_pool2d(featuremap, output_size=(1, 1))
        squeezed = squashed.view(squashed.shape[0], -1)

        self.bottleneck_out = self.bottleneck(squeezed)
        pred = self.head(self.bottleneck_out)

        self.softmax_vals = self.softmax(pred)
        return pred

    @classmethod
    def loss(cls, pred, gt):
        return F.cross_entropy(pred, gt)

class Trainer:
    def __init__(self):

        self.train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(size=(28, 28), scale=(0.7, 1.1)),
            transforms.ToTensor(),
        ])
        self.val_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        self.train_dataset = FashionMNIST("./data", train=True,
                                          transform=self.train_transform,
                                          download=True)
        self.val_dataset = FashionMNIST("./data", train=False,
                                        transform=self.val_transform,
                                        download=True)

        self.index_to_class_dict = defaultdict(list)
        self.index_to_class = {index: image_class for image_class, index in self.val_dataset.class_to_idx.items()}

        n_samples_per_class = 10 # total 10 classes
        for sample in self.val_dataset:
            class_index = sample[1]
            if len(self.index_to_class_dict[class_index]) < n_samples_per_class:
                self.index_to_class_dict[class_index].append(sample)

        self.dataset_classes = [self.index_to_class[key] for key in sorted(list(self.index_to_class_dict.keys()))]
        self.dataset_classes = [image_class.split('/')[0] for image_class in self.dataset_classes]

        batch_size = 1024
        self.train_loader = data.DataLoader(self.train_dataset,
                                            batch_size=batch_size,
                                            shuffle=True, num_workers=4)
        self.val_loader = data.DataLoader(self.val_dataset, batch_size=batch_size,
                                          shuffle=False, num_workers=4)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.net = NeuralNet()
        self.net.to(self.device)

        self.logger = SummaryWriter()
        self.i_batch = 0
        self.i_epoch = 0

        self.gifs_list = []

    def train(self, n_epochs):

        optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3)

        for i_epoch in range(n_epochs):
            self.net.train()

            for feature_batch, gt_batch in self.train_loader:
                feature_batch = feature_batch.to(self.device)
                gt_batch = gt_batch.to(self.device)

                pred_batch = self.net(feature_batch)

                loss = NeuralNet.loss(pred_batch, gt_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                self.logger.add_scalar("train/loss", loss.item(), self.i_batch)

                if self.i_batch % 100 == 0:
                    print(f"batch={self.i_batch} loss={loss.item():.6f}")

                # Samples visualization
                if self.i_batch % 10 == 0:
                    self.net.eval()
                    self.visualize()
                    self.net.train()

                self.i_batch += 1

            self.i_epoch = i_epoch
            self.validate()

    def validate(self, show_conf_matrix=False, show_hard_samples=False):
        self.net.eval()

        loss_all = []
        pred_all = []
        gt_all = []
        softmax_all = []

        for feature_batch, gt_batch in self.val_loader:
            feature_batch = feature_batch.to(self.device)
            gt_batch = gt_batch.to(self.device)

            with torch.no_grad():
                pred_batch = self.net(feature_batch)
                loss = NeuralNet.loss(pred_batch, gt_batch)
                softmax_batch = self.net.softmax_vals

            loss_all.append(loss.item())
            pred_all.append(pred_batch.cpu().numpy())
            gt_all.append(gt_batch.cpu().numpy())
            softmax_all.append(softmax_batch.cpu().numpy())

        loss_mean = np.mean(np.array(loss_all))
        pred_all = np.argmax(np.concatenate(pred_all, axis=0), axis=1)
        gt_all = np.concatenate(np.array(gt_all, dtype=object))
        softmax_all = np.concatenate(np.array(softmax_all, dtype=object))

        if show_conf_matrix == True:
            conf_matrix = confusion_matrix(gt_all, pred_all)
            disp = ConfusionMatrixDisplay(conf_matrix, display_labels=self.dataset_classes)
            figure, axes = plt.subplots(figsize=(FIGURE_XSIZE, FIGURE_XSIZE))
            disp.plot(ax=axes)
            plt.savefig("confusion_matrix.png")

        # Show top-5 hardest samples for each class
        if show_hard_samples == True:
            if not os.path.exists(f"hard_neg"):
                os.mkdir(f"hard_neg")

            mis_classifications_indixes = [index for index in range(0, len(gt_all)) if pred_all[index] != gt_all[index]]
            class_misclassifications = defaultdict(list)
            for index in mis_classifications_indixes:
                class_label = self.val_dataset[index][1]
                class_misclassifications[class_label].append([index, softmax_all[index]])

            for class_label in range(0, len(self.index_to_class_dict)):
                current_class_misses = sorted(class_misclassifications[class_label], key=lambda x: max(x[1]), reverse=True)
                for i in range(0, min(5, len(current_class_misses))):
                    miss_label = np.argmax(current_class_misses[i][1])

                    current_sample = self.val_dataset[current_class_misses[i][0]]
                    if not os.path.exists(f"hard_neg/{self.dataset_classes[class_label]}"):
                        os.mkdir(f"hard_neg/{self.dataset_classes[class_label]}")

                    plt.imsave(f"hard_neg/{self.dataset_classes[class_label]}/hard_neg_{i + 1}({self.dataset_classes[miss_label]}).png", current_sample[0][0])
                    file_softmax = open(f"hard_neg/{self.dataset_classes[class_label]}/hard_neg_{i + 1}({self.dataset_classes[miss_label]}).txt", "w+")
                    file_softmax.write(f"softmax = {current_class_misses[i][1][miss_label]}")
                    file_softmax.close()


        accuracy = np.sum(np.equal(pred_all, gt_all)) / len(pred_all)

        self.logger.add_scalar("val/loss", loss_mean, self.i_batch)
        self.logger.add_scalar("val/accuracy", accuracy, self.i_batch)

        print(f"Val_loss={loss_mean} val_accuracy={accuracy:.6f}")

    def visualize(self):
        colors = ['green', 'cyan', 'magenta', 'blue', 'purple', \
                  'pink', 'orange', 'yellow', 'red', 'lime']

        figure = plt.figure(figsize=(FIGURE_XSIZE, FIGURE_YSIZE), facecolor=BACKGROUND_RGB)
        gs = GridSpec(ncols=1, nrows=1, figure=figure)
        axes = figure.add_subplot(gs[0, 0])
        set_axis_properties(axes)

        axes.set_xlim(-50, 50)
        axes.set_ylim(-50, 50)
        axes.set_title(f"Epoch {self.i_epoch}, batch {self.i_batch}")

        x = []
        y = []

        for index, sample in self.index_to_class_dict.items():
            flatten_samples = [sample[i][0][None, :, :, :] for i in range(len(sample))]
            feature_batch = torch.cat(flatten_samples, 0).to(self.device)

            with torch.no_grad():
                pred = self.net(feature_batch).to(self.device)

            x = self.net.bottleneck_out[:, 0].cpu().detach().numpy()
            y = self.net.bottleneck_out[:, 1].cpu().detach().numpy()

            axes.scatter(x, y, c=colors[index])

        axes.legend(self.dataset_classes,
                    framealpha=LEGEND_FRAME_ALPHA,
                    bbox_to_anchor=(1.15, 1.00))

        axes.legend(self.dataset_classes,
                    framealpha=LEGEND_FRAME_ALPHA,
                bbox_to_anchor=(1.15, 1.00))

        figure.tight_layout()

        filename = f"epoch_{self.i_epoch}_batch_{self.i_batch}.png"
        if not os.path.exists("epoch_visualization"):
            os.makedirs("epoch_visualization")

        figure.savefig("epoch_visualization/" + filename)
        plt.close(figure)

        self.gifs_list.append("epoch_visualization/" + filename)

    def load(self):
        self.net = torch.load("model.pth")

    def save(self):
        torch.save(self.net, "model.pth")

def main():

    trainer = Trainer()
    if len(sys.argv) > 1 and sys.argv[1] == "load":
        trainer.load()
    else:
        trainer.train(n_epochs=100)
        trainer.save()

        images = []
        for filename in trainer.gifs_list:
            image = imageio.v2.imread(filename)
            images.append(image)

        imageio.mimsave('information_bottleneck.gif', images, duration = 0.2)
        shutil.rmtree("epoch_visualization")

    trainer.validate(show_conf_matrix=True, show_hard_samples=True)

if __name__ == "__main__":
    main()
