import torch
from torch.autograd import Variable
import torch.nn as nn
from tqdm import tqdm
from utility.log import IteratorTimer
# import torchvision
import numpy as np
import time
import pickle
import cv2
import random
import math
from math import sin,cos,log,pow


def to_onehot(num_class, label, alpha):
    return torch.zeros((label.shape[0], num_class)).fill_(alpha).scatter_(1, label.unsqueeze(1), 1 - alpha)


def mixup(input, target, gamma):
    # target is onehot format!
    perm = torch.randperm(input.size(0))
    perm_input = input[perm]
    perm_target = target[perm]
    return input.mul_(gamma).add_(1 - gamma, perm_input), target.mul_(gamma).add_(1 - gamma, perm_target)


def clip_grad_norm_(parameters, max_grad):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p[1].grad is not None, parameters))
    max_grad = float(max_grad)

    for name, p in parameters:
        grad = p.grad.data.abs()
        if grad.isnan().any():
            ind = grad.isnan()
            p.grad.data[ind] = 0
            grad = p.grad.data.abs()
        if grad.isinf().any():
            ind = grad.isinf()
            p.grad.data[ind] = 0
            grad = p.grad.data.abs()
        if grad.max() > max_grad:
            ind = grad>max_grad
            p.grad.data[ind] = p.grad.data[ind]/grad[ind]*max_grad  # sign x val

def random_augmentation(x):
    # Rotate
    # x
    anglex = math.radians(random.uniform(-18, 18))
    Rx = torch.tensor([[1, 0, 0],
                       [0, cos(anglex), sin(anglex)],
                       [0, -sin(anglex), cos(anglex)]])
    Rx = Rx.transpose(0,1)
    # y
    angley = math.radians(random.uniform(-18, 18))
    Ry = torch.tensor([[cos(angley), 0, -sin(angley)],
                       [0, 1, 0],
                       [sin(angley), 0, cos(angley)]])
    Ry = Ry.transpose(0,1)
    # z
    anglez = math.radians(random.uniform(-18, 18))
    Rz = torch.tensor([[cos(anglez), sin(anglez), 0],
                       [-sin(anglez), cos(anglez), 0],
                       [0, 0, 1]])
    Rz = Rz.transpose(0,1)
    R_r = torch.matmul(Rz, torch.matmul(Ry, Rx))
    
    # Shear
    sh = random.uniform(-0.3, 0.3)
    R_s = torch.tensor([[1, sh, sh],
                      [sh, 1, sh],
                      [sh, sh, 1]])
    
    #print(x.size()) #N, C, T, V, M
    x = torch.matmul(torch.matmul(x.permute(0, 4, 2, 3, 1), R_r.cuda()), R_s.cuda()) # N, C, T, V, M -> N, M, T, V, C
    x = x.permute(0, 4, 2, 3, 1) # N, M, T, V, C -> N, C, T, V, M
    return x

def train_classifier(label_data_loader, unlabel_data_loader, model, loss_function_reg, loss_function_con, optimizer, global_step, args):
    loss_total = 0
    step = 0
    # process = tqdm(IteratorTimer(data_loader), desc='Train: ')
    labeled_iter = iter(label_data_loader)
    unlabeled_iter = iter(unlabel_data_loader)
    for i in tqdm(range(500)):
        try:
            label_inputsj, label_inputsm, labels = labeled_iter.next()
        except:
            labeled_iter = iter(label_data_loader)
            label_inputsj, label_inputsm, labels = labeled_iter.next()
        
        try:
            unlabel_inputsj, unlabel_inputsm, _ = unlabeled_iter.next()
        except:
            unlabeled_iter = iter(unlabel_data_loader)
            unlabel_inputsj, unlabel_inputsm, _ = unlabeled_iter.next()
        
    # for index, ((label_inputsj, label_inputsm, labels), (unlabel_inputsj, unlabel_inputsm, _)) in enumerate(label_data_loader, unlabel_data_loader):
        label_inputsj, label_inputsm, labels = label_inputsj.cuda(non_blocking=True), label_inputsm.cuda(non_blocking=True), labels.cuda(non_blocking=True)
        unlabel_inputsj, unlabel_inputsm = unlabel_inputsj.cuda(non_blocking=True), unlabel_inputsm.cuda(non_blocking=True)
        # N, C, T, V, M = inputs.shape
        
        #out1s, out1t, out2s, out2t, out3s, out3t, y1, y2, y3 = model(inputs, inputs1)
        #_, _, _, _, _, _, y1, y2, y3 = model(inputs, inputs1)
        #y1, y2, y3 = model(inputs, inputs1)
        out1, out2, out1p, out2p, out1b, out2b, y1, y2 = model(label_inputsj, label_inputsm, unlabel_inputsj, unlabel_inputsm)
        out1 = model.module.mlp(out1)
        out2 = model.module.mlp(out2)
        out1p = model.module.mlp(out1p)
        out2p = model.module.mlp(out2p) # N, C
        out1b = model.module.mlp(out1b)
        out2b = model.module.mlp(out2b)
        y1 = model.module.fc(y1)
        y2 = model.module.fc(y2)
        
        # outputs = out1s+out1t+out2s+out2t+out3s+out3t
        #outputs = y1+y2+y1p+y2p+y1b+y2b
        outputs = y1+y2
        loss_reg = loss_function_reg(outputs, labels)
        
        N = out1.shape[0]
        loss_con = loss_function_con(torch.cat((out1,out2), dim=0), N) + loss_function_con(torch.cat((out1p,out2p), dim=0), N) \
                  +loss_function_con(torch.cat((out1b,out2b), dim=0), N)
                  
        loss = loss_reg + 0.2*loss_con
        
        optimizer.zero_grad()
        loss.backward()
        if args.grad_clip:
            clip_grad_norm_(model.named_parameters(), args.grad_clip)
        optimizer.step()
        global_step += 1
        if len(outputs.data.shape) == 3:  # T N cls
            _, predict_label = torch.max(outputs.data[:, :, :-1].mean(0), 1)
        else:
            _, predict_label = torch.max(outputs.data, 1)
        #loss = loss_function(outputs, targets)
        ls = loss.data.item()
        acc = torch.mean((predict_label == labels.data).float()).item()
        loss_total += ls
        step += 1
        lr = optimizer.param_groups[0]['lr']
        # process.set_description('Train: acc: {:4f}, loss: {:4f}, batch time: {:4f}, lr: {:4f}'.format(acc, ls,
        #                                                                                   process.iterable.last_duration,
        #                                                                                   lr))
    
    # process.close()
    loss = loss_total / step
    return global_step, loss


def val_classifier(data_loader, model, loss_function, global_step, args):
    right_num_total = 0
    total_num = 0
    loss_total = 0
    step = 0
    process = tqdm(IteratorTimer(data_loader), desc='Val: ')
    # s = time.time()
    # t=0
    score_frag = []
    all_pre_true = []
    wrong_path_pre_ture = []
    for index, (inputs, inputs1, labels, path) in enumerate(process):

        with torch.no_grad():
            inputs, inputs1, labels = inputs.cuda(non_blocking=True), inputs1.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            # N, C, T, V, M = inputs.shape
            #out1s, out1t, out2s, out2t, out3s, out3t, y1, y2, y3 = model(inputs, inputs1)
            #_, _, _, _, _, _, y1, y2, y3 = model(inputs, inputs1)
            #y1, y2, y3 = model(inputs, inputs1)
            _, _, _, _, _, _, y1, y2 = model(inputs, inputs1, inputs, inputs1)
            y1 = model.module.fc(y1)
            y2 = model.module.fc(y2)
            
            outputs = y1+y2
            loss = loss_function(outputs, labels)
            
            if len(outputs.data.shape) == 3:  # T N cls
                _, predict_label = torch.max(outputs.data[:, :, :-1].mean(0), 1)
                score_frag.append(outputs.data.cpu().numpy().transpose(1,0,2))
            else:
                _, predict_label = torch.max(outputs.data, 1)
                score_frag.append(outputs.data.cpu().numpy())

        predict = list(predict_label.cpu().numpy())
        true = list(labels.data.cpu().numpy())
        for i, x in enumerate(predict):
            all_pre_true.append(str(x) + ',' + str(true[i]) + '\n')
            if x != true[i]:
                wrong_path_pre_ture.append(str(path[i]) + ',' + str(x) + ',' + str(true[i]) + '\n')

        right_num = torch.sum(predict_label == labels.data).item()
        # right_num = torch.sum(predict_label == labels.data)
        batch_num = labels.data.size(0)
        acc = right_num / batch_num
        ls = loss.data.item()
        # ls = loss.data[0]

        right_num_total += right_num
        total_num += batch_num
        loss_total += ls
        step += 1

        process.set_description(
            'Val-batch: acc: {:4f}, loss: {:4f}, time: {:4f}'.format(acc, ls, process.iterable.last_duration))
        
    score = np.concatenate(score_frag)
    score_dict = dict(zip(data_loader.dataset.sample_name, score))

    process.close()
    loss = loss_total / step
    accuracy = right_num_total / total_num
    print('Accuracy: ', accuracy)

    return loss, accuracy, score_dict, all_pre_true, wrong_path_pre_ture

