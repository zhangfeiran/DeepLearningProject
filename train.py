import os, sys

if sys.platform == 'linux':
    import pip
    pip.main(['install', 'pandas'])
    pip.main(['install', 'iterative-stratification'])
    pip.main(['install', 'imgaug'])
    pip.main(['install', 'tensorboardX'])
    pip.main(['install', 'pretrainedmodels'])
    os.chdir('./cos_person/proteinatlas/')
    sys.path.append('./')
    os.environ['TORCH_MODEL_ZOO'] = './'

import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math
from tensorboardX import SummaryWriter
# from transforms import policy_transform
from models import Resnet34, binary_focal_loss, adam, f1_score
from dataset import HPADataset
from BatchCollator import BatchCollator
import utils_checkpoint


def inference(model, images):
    logits = model(images)
    if isinstance(logits, tuple):
        logits, aux_logits = logits
    else:
        aux_logits = None
    probabilities = F.sigmoid(logits)
    return logits, aux_logits, probabilities


def evaluate_single_epoch(model, dataloader, criterion, epoch, writer, postfix_dict):
    model.eval()

    with torch.no_grad():
        batch_size = 32
        total_size = len(dataloader.dataset)
        total_step = math.ceil(total_size / batch_size)

        probability_list = []
        label_list = []
        loss_list = []
        for i, data in enumerate(dataloader):
            images = data['image'].cuda()
            labels = data['label'].cuda()
            logits, aux_logits, probabilities = inference(model, images)

            loss = criterion(logits, labels.float())
            if aux_logits is not None:
                aux_loss = criterion(aux_logits, labels.float())
                loss = loss + 0.4 * aux_loss
            loss_list.append(loss.item())

            probability_list.extend(probabilities.cpu().numpy())
            label_list.extend(labels.cpu().numpy())

            f_epoch = epoch + i / total_step

        log_dict = {}
        labels = np.array(label_list)
        probabilities = np.array(probability_list)

        predictions = (probabilities > 0.5).astype(int)
        accuracy = np.sum((predictions == labels).astype(float)) / float(predictions.size)

        log_dict['acc'] = accuracy
        log_dict['f1'] = f1_score(labels, predictions)
        log_dict['loss'] = sum(loss_list) / len(loss_list)

        if writer is not None:
            for l in range(28):
                f1 = f1_score(labels[:, l], predictions[:, l], 'binary')
                writer.add_scalar('val/f1_{:02d}'.format(l), f1, epoch)

        for key, value in log_dict.items():
            if writer is not None:
                writer.add_scalar('val/{}'.format(key), value, epoch)
            postfix_dict['val/{}'.format(key)] = value

        return f1


def train_single_epoch(model, dataloader, criterion, optimizer, epoch, writer, postfix_dict):
    model.train()

    batch_size = 32
    total_size = len(dataloader.dataset)
    total_step = math.ceil(total_size / batch_size)

    log_dict = {}
    for i, data in enumerate(dataloader):
        images = data['image'].cuda()
        labels = data['label'].cuda()
        logits, aux_logits, probabilities = inference(model, images)
        loss = criterion(logits, labels.float())
        if aux_logits is not None:
            aux_loss = criterion(aux_logits, labels.float())
            loss = loss + 0.4 * aux_loss
        log_dict['loss'] = loss.item()

        predictions = (probabilities > 0.5).long()
        accuracy = (predictions == labels).sum().float() / float(predictions.numel())
        log_dict['acc'] = accuracy.item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        f_epoch = epoch + i / total_step

        log_dict['lr'] = optimizer.param_groups[0]['lr']
        for key, value in log_dict.items():
            postfix_dict['train/{}'.format(key)] = value

        if i % 100 == 0:
            log_step = int(f_epoch * 10000)
            if writer is not None:
                for key, value in log_dict.items():
                    writer.add_scalar('train/{}'.format(key), value, log_step)
                    print('train/{}'.format(key), value, log_step)


def train(train_dir, model, dataloaders, criterion, optimizer, writer, start_epoch):
    num_epochs = 20

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model = model.cuda()

    postfix_dict = {'train/lr': 0.0, 'train/acc': 0.0, 'train/loss': 0.0, 'val/f1': 0.0, 'val/acc': 0.0, 'val/loss': 0.0}

    f1_list = []
    best_f1 = 0.0
    best_f1_mavg = 0.0
    for epoch in range(start_epoch, num_epochs):
        print('epoch:', epoch)
        # train phase
        train_single_epoch(model, dataloaders['train'], criterion, optimizer, epoch, writer, postfix_dict)

        # val phase
        f1 = evaluate_single_epoch(model, dataloaders['val'], criterion, epoch, writer, postfix_dict)

        utils_checkpoint.save_checkpoint(train_dir, model, optimizer, epoch, 0)

        f1_list.append(f1)
        f1_list = f1_list[-10:]
        f1_mavg = sum(f1_list) / len(f1_list)

        if f1 > best_f1:
            best_f1 = f1
        if f1_mavg > best_f1_mavg:
            best_f1_mavg = f1_mavg
    return {'f1': best_f1, 'f1_mavg': best_f1_mavg}


import warnings
warnings.filterwarnings("ignore")

for i in range(4):
    train_dir = './results233/resnet34.' + str(i)
    os.makedirs(os.path.join(train_dir, 'checkpoint'), exist_ok=True)
    model = Resnet34().cuda()
    criterion = binary_focal_loss()
    optimizer = adam(model.parameters())
    checkpoint = utils_checkpoint.get_initial_checkpoint(train_dir)
    if checkpoint is not None:
        last_epoch, step = utils_checkpoint.load_checkpoint(model, optimizer, checkpoint)
    else:
        last_epoch, step = -1, -1


    dataloaders = {}
    path = './'
    train_dataset = HPADataset(path, i, 'train')
    val_dataset = HPADataset(path, i, 'val')
    train_sampler = torch.utils.data.WeightedRandomSampler(train_dataset.weights, len(train_dataset.weights))
    batch_sampler = torch.utils.data.sampler.BatchSampler(train_sampler, 32, drop_last=True)
    collator = BatchCollator()

    dataloaders['train'] = DataLoader(train_dataset, batch_sampler=batch_sampler)
    dataloaders['val'] = DataLoader(val_dataset, shuffle=True, batch_size=32)

    writer = SummaryWriter(train_dir)
    print('start training')
    train(train_dir, model, dataloaders, criterion, optimizer, writer, last_epoch + 1)
