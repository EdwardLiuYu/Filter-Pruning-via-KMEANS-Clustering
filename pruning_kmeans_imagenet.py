# https://github.com/pytorch/vision/blob/master/torchvision/models/__init__.py
import argparse
import os, sys
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models
from utils import convert_secs2time, time_string, time_file_str, timing
# from models import print_log
import models
import random
import numpy as np
from scipy.spatial import distance
from collections import OrderedDict
from sklearn.cluster import KMeans
import math


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--save_dir', type=str, default='./', help='Folder to save checkpoints and log.')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W',
                    help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=200, type=int, metavar='N', help='print frequency (default: 100)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--use_pretrain', dest='use_pretrain', action='store_true', help='use pre-trained model or not')

# compress rate
parser.add_argument('--pruning_rate', type=float, default=0.1, help='the reducing ratio of pruning based on Distance')

parser.add_argument('--layer_begin', type=int, default=3, help='compress layer of model')
parser.add_argument('--layer_end', type=int, default=3, help='compress layer of model')
parser.add_argument('--layer_inter', type=int, default=1, help='compress layer of model')
parser.add_argument('--skip_downsample', type=int, default=1, help='compress layer of model')
parser.add_argument('--use_sparse', dest='use_sparse', action='store_true', help='use sparse model as initial or not')
parser.add_argument('--sparse',
                    default='/data/yahe/imagenet/resnet50-rate-0.7/checkpoint.resnet50.2018-01-07-9744.pth.tar',
                    type=str, metavar='PATH', help='path of sparse model')
parser.add_argument('--lr_adjust', type=int, default=30, help='number of epochs that change learning rate')
parser.add_argument('--VGG_pruned_style', choices=["CP_5x", "Thinet_conv"],
                    help='number of epochs that change learning rate')
parser.add_argument('--n_clusters', type=int, default=4, help='number of clusters for kmeans')
parser.add_argument('--cos', dest='cos', action='store_true', help='use cos update lr')
args = parser.parse_args()
args.use_cuda = torch.cuda.is_available()

args.prefix = time_file_str()


def main():
    best_prec1 = 0

    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)
    log = open(os.path.join(args.save_dir, '{}.{}.log'.format(args.arch, args.prefix)), 'w')

    # version information
    print_log("PyThon  version : {}".format(sys.version.replace('\n', ' ')), log)
    print_log("PyTorch version : {}".format(torch.__version__), log)
    print_log("cuDNN   version : {}".format(torch.backends.cudnn.version()), log)
    print_log("Vision  version : {}".format(torchvision.__version__), log)
    # create model
    print_log("=> creating model '{}'".format(args.arch), log)
    model = models.__dict__[args.arch](pretrained=args.use_pretrain)
    if args.use_sparse:
        model = import_sparse(model)
    print_log("=> Model : {}".format(model), log)
    print_log("=> parameter : {}".format(args), log)
    print_log("Pruning Rate: {}".format(args.pruning_rate), log)
    print_log("Layer Begin: {}".format(args.layer_begin), log)
    print_log("Layer End: {}".format(args.layer_end), log)
    print_log("Layer Inter: {}".format(args.layer_inter), log)
    print_log("Skip downsample : {}".format(args.skip_downsample), log)
    print_log("Workers         : {}".format(args.workers), log)
    print_log("Learning-Rate   : {}".format(args.lr), log)
    print_log("Use Pre-Trained : {}".format(args.use_pretrain), log)
    print_log("lr adjust : {}".format(args.lr_adjust), log)
    print_log("Cosine lr: {}".format(args.cos), log)

    if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
    else:
        model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print_log("=> loading checkpoint '{}'".format(args.resume), log)
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print_log("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']), log)
        else:
            print_log("=> no checkpoint found at '{}'".format(args.resume), log)

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=None)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, log)
        return

    filename = os.path.join(args.save_dir, 'checkpoint.{:}.{:}.pth.tar'.format(args.arch, args.prefix))
    bestname = os.path.join(args.save_dir, 'best.{:}.{:}.pth.tar'.format(args.arch, args.prefix))

    m = Mask(model)

    m.init_length()
    print("-" * 10 + "one epoch begin" + "-" * 10)
    print("reducing ratio of pruning : %f" % args.pruning_rate)
    print("total remaining ratio is %f" % (1 - args.pruning_rate))


   
    
    m.model = model
    m.init_mask(args.pruning_rate, args.n_clusters)
    m.do_similar_mask()
    model = m.model
    m.if_zero()
    if args.use_cuda:
        model = model.cuda()
    val_acc_1 = validate(val_loader, model, criterion, log)
    print_log(">>>>> accu after pruning is: {:}".format(val_acc_1),log)

    
    start_time = time.time()
    epoch_time = AverageMeter()
    for epoch in range(args.start_epoch, args.epochs):
        if args.cos:
            print_log('Using cos lr',log)
            cos_learning_rate(optimizer, epoch)
        else:
            adjust_learning_rate(optimizer, epoch)

        need_hour, need_mins, need_secs = convert_secs2time(epoch_time.val * (args.epochs - epoch))
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)
        print_log(
            ' [{:s}] :: {:3d}/{:3d} ----- [{:s}] {:s}'.format(args.arch, epoch, args.epochs, time_string(), need_time),
            log)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, log, m)

        val_acc_2 = validate(val_loader, model, criterion, log)

        # remember best prec@1 and save checkpoint
        is_best = val_acc_2 > best_prec1
        best_prec1 = max(val_acc_2, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best, filename, bestname)
        # measure elapsed time
        epoch_time.update(time.time() - start_time)
        start_time = time.time()
        
    print_log('Best Accuracy:{}'.format(best_prec1),log)
    log.close()


def import_sparse(model):
    checkpoint = torch.load(args.sparse)
    new_state_dict = OrderedDict()
    
    for k, v in checkpoint['state_dict'].items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    print("sparse_model_loaded")
    return model


def train(train_loader, model, criterion, optimizer, epoch, log, m):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        input_var = input.cuda()
        target_var = target

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        # Mask grad for iteration
        m.do_grad_mask()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print_log('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5), log)


def validate(val_loader, model, criterion, log):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input_var = input.cuda()
            target_var = target

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print_log('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5), log)

        print_log(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1, top5=top5,
                                                                                           error1=100 - top1.avg), log)

    return top1.avg


def save_checkpoint(state, is_best, filename, bestname):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, bestname)


def print_log(print_string, log):
    print("{:}".format(print_string))
    log.write('{:}\n'.format(print_string))
    log.flush()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // args.lr_adjust))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
def cos_learning_rate(optimizer, epoch):
    if epoch <= 100-10 and epoch > 30:
        min_lr = args.lr * 0.001
        tmp_lr = min_lr + 0.5*(args.lr-min_lr)*(1+math.cos(math.pi*(epoch-30)*1./\
                        (100-10-30)))
    elif epoch > 100-10:
        tmp_lr = args.lr * 0.001
    else:
        tmp_lr = args.lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = tmp_lr
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class Mask:
    def __init__(self, model):
        self.model_size = {}
        self.model_length = {}
        self.distance_rate = {}
        self.model = model
        self.mask_index = []
        self.filter_small_index = {}
        self.filter_large_index = {}
        self.similar_matrix = {}


    @timing
    def get_filter_kmeans(self, weight_torch, distance_rate, length, num_clusters):
        codebook = np.ones(length)      # length = Ni+1*Ni*k*k
        label_sum = []              # num_filters for each cluster
        if len(weight_torch.size()) == 4:
            similar_pruned_num = []
            weight_vec = weight_torch.view(weight_torch.size()[0], -1)
            # num_clusters = int(weight_vec.size()[0]/num_clusters)

            
            # KMeans to cluster filters
            kmeans = KMeans(n_clusters=num_clusters, max_iter=300).fit(weight_vec.cpu().numpy())
            centroids = torch.from_numpy(kmeans.cluster_centers_)   # return torch.size(n_cluster, N_i*k*k)
            labels = kmeans.labels_       # return size=N_i+1 
            
            # get a list for how many filters in each cluster， define prune num in each cluster
            for i in range(num_clusters):
                label_sum.append(sum(labels==i))
                similar_pruned_num.append(int(label_sum[i]* distance_rate))

            # for distance similar: get the filter index with largest similarity == small distance to centroids
            for i in range(num_clusters):
                weight_sub = weight_vec.cpu()[labels==i]             # size=N_i+1*cluster_num
                weight_index = np.where(labels==i)[0]          # get index of weight_sub in weight_vec
                norm2_np = torch.norm(centroids[i]-weight_sub,2,1).numpy()         # using broadcast
                filter_large_index = norm2_np.argsort()[similar_pruned_num[i]:]
                filter_small_index = norm2_np.argsort()[:similar_pruned_num[i]]
                if i != 0 :
                    kmeans_index_for_filter = np.append(kmeans_index_for_filter, weight_index[filter_small_index])
                else:
                    kmeans_index_for_filter = weight_index[filter_small_index]
            
            # assert pruning ratio
            if sum(similar_pruned_num) < sum(label_sum)*distance_rate:
                norm_prune_num = int((sum(label_sum)*distance_rate - sum(similar_pruned_num)))
                left_index = [x for x in np.array(range(weight_vec.size()[0])) if x not in  kmeans_index_for_filter]
                weight_sub_1 = weight_vec.cpu()[left_index]
                norm2_np = torch.norm(weight_sub_1,2,1).numpy()
                norm_small_index = norm2_np.argsort()[:norm_prune_num]
                norm_index_for_filter = np.array(left_index)[norm_small_index]
            kernel_length = weight_torch.size()[1] * weight_torch.size()[2] * weight_torch.size()[3]
            codebook = torch.FloatTensor(codebook.reshape(sum(label_sum),kernel_length))
            codebook[kmeans_index_for_filter,:] = 0
            codebook[norm_index_for_filter,:] = 0
            codebook = codebook.view(length).numpy() 
            print("kmeans index done") 
        else:
            pass
        return codebook ,weight_torch


    def convert2tensor(self, x):
        x = torch.FloatTensor(x)
        return x

    def init_length(self):
        for index, item in enumerate(self.model.parameters()):
            self.model_size[index] = item.size()

        for index1 in self.model_size:
            for index2 in range(0, len(self.model_size[index1])):
                if index2 == 0:
                    self.model_length[index1] = self.model_size[index1][0]
                else:
                    self.model_length[index1] *= self.model_size[index1][index2]

    def init_rate(self, rate_dist_per_layer):
        if 'vgg' in args.arch:
            cfg_official = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
            cfg_CP_5x = [24, 22, 41, 51, 108, 89, 111, 184, 276, 228, 512, 512, 512]
            # cfg = [32, 64, 128, 128, 256, 256, 256, 256, 256, 256, 256, 256, 256]
            cfg_Thinet_conv = [32, 32, 64, 64, 128, 128, 128, 256, 256, 256, 512, 512, 512]
            if args.VGG_pruned_style == "CP_5x":
                cfg_now = cfg_CP_5x
            elif args.VGG_pruned_style == "Thinet_conv":
                cfg_now = cfg_Thinet_conv

            cfg_index = 0
            previous_cfg = True
            for index, item in enumerate(self.model.named_parameters()):
                if len(item[1].size()) == 4:
                    if not previous_cfg:
                        self.distance_rate[index] = rate_dist_per_layer
                        self.mask_index.append(index)
                        print(item[0], "self.mask_index", self.mask_index)
                    else:
                        self.distance_rate[index] = 1 - cfg_now[cfg_index] / item[1].size()[0]
                        self.mask_index.append(index)
                        print(item[0], "self.mask_index", self.mask_index, cfg_index, cfg_now[cfg_index])
                        cfg_index += 1
        elif "resnet" in args.arch:
            for index, item in enumerate(self.model.parameters()):
                self.distance_rate[index] = 1
            for key in range(args.layer_begin, args.layer_end + 1, args.layer_inter):
                self.distance_rate[key] = rate_dist_per_layer
            # different setting for  different architecture
            if args.arch == 'resnet18':
                # last index include last fc layer
                last_index = 60
                skip_list = [21, 36, 51]
            elif args.arch == 'resnet34':
                last_index = 108
                skip_list = [27, 54, 93]
            elif args.arch == 'resnet50':
                last_index = 159
                skip_list = [12, 42, 81, 138]
            elif args.arch == 'resnet101':
                last_index = 312
                skip_list = [12, 42, 81, 291]
            elif args.arch == 'resnet152':
                last_index = 465
                skip_list = [12, 42, 117, 444]
            self.mask_index = [x for x in range(0, last_index, 3)]
            # skip downsample layer
            if args.skip_downsample == 1:
                for x in skip_list:
                    self.mask_index.remove(x)
                    print(self.mask_index)
            else:
                pass

    def init_mask(self, rate_dist_per_layer, num_clusters):
        self.init_rate(rate_dist_per_layer)
        for index, item in enumerate(self.model.parameters()):
            if index in self.mask_index:
                self.similar_matrix[index],weight_torch = self.get_filter_kmeans(item.data, self.distance_rate[index], self.model_length[index],num_clusters)
                # item.data = weight_torch
                self.similar_matrix[index] = self.convert2tensor(self.similar_matrix[index])
                if args.use_cuda:
                    self.similar_matrix[index] = self.similar_matrix[index].cuda()
        print("mask Ready")

    def do_similar_mask(self):
        for index, item in enumerate(self.model.parameters()):
            if index in self.mask_index:
                a = item.data.view(self.model_length[index])
                b = a * self.similar_matrix[index]
                item.data = b.view(self.model_size[index])
        print("mask similar Done")

    def do_grad_mask(self):
        for index, item in enumerate(self.model.parameters()):
            if index in self.mask_index:
                a = item.grad.data.view(self.model_length[index])
                # reverse the mask of model
                b = a * self.similar_matrix[index]
                item.grad.data = b.view(self.model_size[index])
        # print("grad zero Done")

    def if_zero(self):
        for index, item in enumerate(self.model.parameters()):
            if index in self.mask_index:
                # if index in [x for x in range(args.layer_begin, args.layer_end + 1, args.layer_inter)]:
                a = item.data.view(self.model_length[index])
                b = a.cpu().numpy()

                print("layer: %d, number of nonzero weight is %d, zero is %d" % (
                    index, np.count_nonzero(b), len(b) - np.count_nonzero(b)))


if __name__ == '__main__':
    main()
