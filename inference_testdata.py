from argparse import ArgumentParser

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ignite.engine import Events, Engine
from ignite.metrics import RunningAverage, Loss, ConfusionMatrix, IoU
from denoising_code.datasets import DENSE, Normalize, Compose, RandomHorizontalFlip
from denoising_code.datasets.transforms import ToTensor
from denoising_code.model import WeatherNet
from denoising_code.model.Squeezeseg import Squeezeseg
from denoising_code.model.PPliteseg import PPLiteSeg
from denoising_code.model.SAFDN import SAFDN
from ignite.contrib.handlers import ProgressBar
from ignite.utils import convert_tensor
import numpy as np
import h5py


def get_data_loaders(data_dir, batch_size=None, num_workers=None):
    normalize = Normalize(mean=DENSE.mean(), std=DENSE.std())
    transforms = Compose([
        ToTensor()
    ])

    test_loader = DataLoader(DENSE(root=data_dir, split='test', transform=transforms),
                             batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return test_loader


def run(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    num_classes = DENSE.num_classes()
    # model = WeatherNet(num_classes)
    # model = Squeezeseg(num_classes)
    model = PPLiteSeg(num_classes)

    # device_count = torch.cuda.device_count()
    # if device_count > 1:
    #     print("Using %d GPU(s)" % device_count)
    #     model = nn.DataParallel(model)
    #     args.batch_size = device_count * args.batch_size
    #     args.val_batch_size = device_count * args.val_batch_size

    model = model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    test_loader = get_data_loaders(data_dir=args.dataset_dir, batch_size=args.batch_size, num_workers=args.num_workers)
    print(test_loader.__len__())

    # model.load_state_dict(torch.load('checkpoints_PPliteseg_tezhen/model_epoch20_mIoU=72.3.pth'))
    # model.load_state_dict(torch.load('checkpoints_squeezeseg/model_epoch20_mIoU=70.2.pth'))
    # model.load_state_dict(torch.load('checkpoints_PPliteseg_tezhen/model_epoch16_mIoU=73.1.pth'))
    model.load_state_dict(torch.load('checkpoints_PPliteseg_both/model_epoch46_mIoU=85.3.pth'))


    def _prepare_batch(batch, non_blocking=True):
        distance, reflectivity, target = batch

        return (torch.cat([convert_tensor(distance, device=device, non_blocking=non_blocking),
                convert_tensor(reflectivity, device=device, non_blocking=non_blocking)], 1),
                convert_tensor(target, device=device, non_blocking=non_blocking))
    # def _prepare_batch(batch, non_blocking=True):
    #     distance, reflectivity, target = batch
    #
    #     return (convert_tensor(distance, device=device, non_blocking=non_blocking),
    #             convert_tensor(reflectivity, device=device, non_blocking=non_blocking),
    #             convert_tensor(target, device=device, non_blocking=non_blocking))

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            distance_and_reflectivity, target = _prepare_batch(batch)
            pred = model(distance_and_reflectivity)

            return pred, target

    # def _inference(engine, batch):
    #     model.eval()
    #     with torch.no_grad():
    #         dis, ref, target = _prepare_batch(batch)
    #         pred = model(dis, ref)
    #
    #         return pred, target

    evaluator = Engine(_inference)
    cm = ConfusionMatrix(num_classes)
    cm.attach(evaluator, 'cm')
    IoU(cm, ignore_index=0).attach(evaluator, 'IoU')
    Loss(criterion).attach(evaluator, 'loss')

    # RunningAverage(output_transform=lambda x: x).attach(evaluator, 'loss')
    pbar2 = ProgressBar(persist=True, desc='Eval Epoch')
    pbar2.attach(evaluator)

    print("Start validation")
    evaluator.run(test_loader, max_epochs=1)
    metrics = evaluator.state.metrics
    loss = metrics['loss']
    cm = metrics['cm']
    iou = metrics['IoU'] * 100.0
    mean_iou = iou.mean()

    iou_text = ', '.join(['{}: {:.1f}'.format(DENSE.classes[i + 1].name, v) for i, v in enumerate(iou.tolist())])
    pbar2.log_message("Validation results - Epoch: [{}/{}]: Loss: {:.2e}\n IoU: {}\n mIoU: {:.1f} \n cm: {}"
                      .format(evaluator.state.epoch, evaluator.state.max_epochs, loss, iou_text, mean_iou, cm))


if __name__ == '__main__':
    parser = ArgumentParser('WeatherNet with PyTorch')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='input batch size for training')
    # parser.add_argument('--val-batch-size', type=int, default=8,
    #                     help='input batch size for validation')
    parser.add_argument('--num-workers', type=int, default=6,
                        help='number of workers')
    parser.add_argument('--output-dir', default='checkpoints',
                        help='directory to save model checkpoints')
    parser.add_argument("--dataset-dir", type=str, default="cnn_denoising/",
                        help="location of the dataset")

    run(parser.parse_args())
