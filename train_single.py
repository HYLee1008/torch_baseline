import os
import json
import torch
import argparse
import numpy as np
from importlib import import_module
import sys
import time
import torchvision
import torchvision.transforms as transforms


from utils import *


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def validation(epoch, model, data_loader, device):
    print('Start validation #{}'.format(epoch))
    model.eval()
    with torch.no_grad():
        images_count = 0
        correct_count = 0
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            images_count += len(labels)

            outs = model(inputs)
            preds = torch.argmax(outs, dim=-1)

            acc_item = (labels == preds).sum().item()
            correct_count += acc_item


        val_acc = correct_count / images_count

    return val_acc


def train(num_epochs, model, train_data_loader, val_data_loader, criterion, optimizer, saved_dir, val_every, device, model_name):
    # start training
    print('Start training..')
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max')
    best_acc = 0

    for epoch in range(num_epochs):
        model.train()
        loss_value = 0
        matches = 0
        start_time = time.time()
        for step, (images, labels) in enumerate(train_data_loader):
            # gpu 연산을 위해 device 할당
            images, labels = images.to(device), labels.to(device)
                  
            # inference
            outs = model(images)
            preds = torch.argmax(outs, dim=-1)
            
            # loss 계산 (cross entropy loss)
            loss = criterion(outs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()

            loss_value += loss.item()
            matches += (preds == labels).sum().item()
            
            # step 주기에 따른 loss 출력
            if (step + 1) % 100 == 0:
                train_loss = loss_value / 100
                train_acc = matches / args.batch_size / 100
                print(
                    f"Epoch[{epoch}/{args.epochs}]({step + 1}/{len(train_data_loader)}) || "
                    f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%}"
                )

                loss_value = 0
                matches = 0
        
        # validation 주기에 따른 loss 출력 및 best model 저장
        if (epoch + 1) % val_every == 0:
            val_acc = validation(epoch + 1, model, val_data_loader, device)
            scheduler.step(val_acc)
            for param_group in optimizer.param_groups:
                print('epoch: {}, lr: {}'.format(epoch + 1, param_group['lr']))

            if val_acc > best_acc:
                print('Best accuracy performance at epoch: {}'.format(epoch + 1))
                print('Save model in', saved_dir)
                best_acc = val_acc

                # save best model
                output_path = os.path.join(saved_dir, f"{model_name}_best_acc.pt")
                torch.save(model.state_dict(), output_path)
        
    # save last model
    output_path = os.path.join(saved_dir, f"{model_name}_last.pt")
    torch.save(model.state_dict(), output_path)


def main(args):
    # fix seed
    seed_everything(seed=args.seed)

    # defind save directory
    save_dir = increment_path(os.path.join(args.model_dir, args.model))
    os.makedirs(save_dir, exist_ok=True)

    # save args on .json file
    with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)

    # open sys logging
    consoleLog = Logger(save_dir=save_dir)
    sys.stdout = consoleLog

    # set augmentation methods
    # transform_module = getattr(import_module("augmentation"), args.augmentation)
    # train_transform = transform_module()
    # val_transform = transform_module()

    train_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    val_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Load dataset
    # train_dataset = PseudoTrainset(data_dir=train_path, transform=train_transform)
    # val_dataset = PseudoTrainset(data_dir=val_path, transform=val_transform)

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=train_transform)

    val_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=val_transform)


    # collate_fn needs for batch
    # def collate_fn(batch):
    #     return tuple(zip(*batch))

    # DataLoader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                            batch_size=args.batch_size,
                                            shuffle=True,
                                            num_workers=args.num_workers,
                                            # collate_fn=collate_fn
                                            )

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, 
                                            batch_size=args.batch_size,
                                            shuffle=False,
                                            num_workers=args.num_workers,
                                            # collate_fn=collate_fn
                                            )

    # define model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model_module = getattr(import_module('model'), args.model)
    model = model_module(num_classes=10).to(device)

    # define optimizer  & criterion
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion_module = getattr(import_module("loss"), args.loss)
    criterion = criterion_module()

    # train model
    val_every = 1
    train(args.epochs, model, train_loader, val_loader, criterion, optimizer, save_dir, val_every, device, args.model)

    # close sys logging
    del consoleLog


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=1024, help='random seed (default: 1024)')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs for train (deafult: 20)')
    parser.add_argument('--batch_size', type=int, default=8, help='input batch size for training (deafult: 8)')
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers for dataloader (default: 2)')

    parser.add_argument('--learning_rate', type=float, default=1e-5, help='learning rate for training (default: 1e-6)')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='weight decay (default: 1e-6)')

    parser.add_argument('--model_dir', type=str, default='./results', help='directory where model would be saved (default: ./results)')

    parser.add_argument('--model', type=str, default='Resnet50', help='backbone bert model for training (default: FCN8s)')
    parser.add_argument('--augmentation', type=str, default='BasicAugmentation', help='augmentation method for training')
    parser.add_argument('--loss', type=str, default='CrossEntropyLoss', help='loss function to train the model')

    args = parser.parse_args()

    main(args)