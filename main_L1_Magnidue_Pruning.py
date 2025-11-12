'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar

import numpy as np
import random

import time
from torch.nn.utils import prune
import os

# 누적 시간(초)
train_time_sec = 0.0
test_time_sec  = 0.0

def _sync():
    # CUDA일 때 정확한 시간 계측을 위한 동기화
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def _fmt(sec: float) -> str:
    # 보기 좋은 시간 포맷 (HH:MM:SS)
    m, s = divmod(sec, 60)
    h, m = divmod(int(m), 60)
    return f"{h:02d}:{m:02d}:{int(s):02d}"

def set_seed(seed: int = 15):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi-GPU 대응
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed fixed to: {seed}")

def seed_worker(worker_id):
    # 각 워커별 고유 seed (Base는 mySeed)
    worker_seed = mySeed + worker_id
    import torch, numpy as np, random
    torch.manual_seed(worker_seed)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# ====== Magnitude Pruning(전역 L1) 유틸 ======
def _collect_prunable_params(model):
    # Conv/Linear의 weight만 대상으로 전역 프루닝
    params = []
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            params.append((m, 'weight'))
    return params

def _global_sparsity(model) -> float:
    zeros = 0
    total = 0
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            w = m.weight
            zeros += (w == 0).sum().item()
            total += w.numel()
    return zeros, total, zeros / max(1, total)

# 'MY_SEED' 환경 변수에서 seed 값을 읽어옴. 없으면 15를 기본값으로 사용.
mySeed = int(os.getenv("MY_SEED", "15")) 
set_seed(mySeed)

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

generator = torch.Generator().manual_seed(mySeed)


trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2,
    worker_init_fn=seed_worker,
    generator=generator
    )

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2,
    worker_init_fn=seed_worker,
    generator=generator
    )

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = VGG('VGG19')
net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
# net = SimpleDLA()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    #cudnn.benchmark = True
    cudnn.benchmark = False

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        #print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc

_sync(); total_t0 = time.perf_counter()

print('Vic-251108-23h-ResNet18')
myEpoch = 100
PRUNE_AT_EPOCH = myEpoch -1

PRUNE_AMOUNT = float(os.getenv("PRUNE_AMOUNT", "0.50"))  # default 0.50
#PRUNE_AMOUNT     = 0.5     # 전역 희소도 목표(누적) 50%
REMOVE_AT_FINISH = False     # 학습 끝나고 mask를 weight에 굽기(export 용)

print('Training Epoch: ', myEpoch, 'Prune %: ', PRUNE_AMOUNT)
for epoch in range(start_epoch, start_epoch+myEpoch):
    train(epoch)

    # ====== PRUNE: 전역 Random 프루닝으로 변경 ======
    if epoch == PRUNE_AT_EPOCH and PRUNE_AMOUNT == 0 :
        print(f"[PRUNE] Prune has not been applied ")
    elif epoch == PRUNE_AT_EPOCH : 
        model_for_prune = net.module if isinstance(net, nn.DataParallel) else net
        params = _collect_prunable_params(model_for_prune)
        prune.global_unstructured(
            params,
            pruning_method=prune.L1Unstructured,
            amount=PRUNE_AMOUNT,    # 누적 전역 희소도 목표
        )
        pruned_count, total_count, gs = _global_sparsity(model_for_prune)
        print(f"[PRUNE] Applied global Random pruning to {PRUNE_AMOUNT*100:.1f}% target.")
        print(f"        → Pruned (zeroed) weights: {pruned_count:,} / {total_count:,}")
        print(f"        → Current global sparsity: {gs*100:.2f}%")        
    # =============================================

    test(epoch)
    scheduler.step()

_sync(); total_dt = time.perf_counter() - total_t0

epochs_done = (start_epoch+myEpoch) - start_epoch
avg_train = train_time_sec / max(1, epochs_done)
avg_test  = test_time_sec  / max(1, epochs_done)

print("="*60)
print(f"[SUMMARY] epochs: {epochs_done}")
print(f"[SUMMARY] total     : {_fmt(total_dt)}")
#print(f"[SUMMARY] train cum : {_fmt(train_time_sec)}  (avg/epoch: {avg_train:.2f}s)")
#print(f"[SUMMARY] test  cum : {_fmt(test_time_sec)}   (avg/epoch: {avg_test:.2f}s)")
print("="*60)