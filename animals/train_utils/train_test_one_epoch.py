'''
一个epoch 的 train
一个epoch 的 test
'''
import time

import torch
from train_utils.distributed_utils import SmoothedValue, MetricLogger


# train
def train_epoch(train_loader, model, device, optimizer, loss_function, epoch, lr_scheduler, print_freq=10):
    model.train()
    # 日志: 显示运行过程中的结果
    metric_logger = MetricLogger()
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value: 6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    train_loss = 0
    for i, (train_img, train_label) in enumerate(metric_logger.log_every(train_loader, print_freq=print_freq, header=header)):
        train_img, train_label = train_img.to(device), train_label.to(device)
        output = model(train_img)
        loss = loss_function(output, train_label.long())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lr_scheduler.step()

        train_loss_batch = loss.item()
        train_loss = (train_loss * i + train_loss_batch) / (i+1)

        metric_logger.update(loss=loss)
        now_lr = optimizer.param_groups[0]["lr"]
        '''
        长度为6的字典，包括[‘amsgrad', ‘params', ‘lr', ‘betas', ‘weight_decay', ‘eps']这6个参数；
        '''
        metric_logger.update(lr=now_lr)

    return train_loss, now_lr


# test
def test_epoch(test_loader, model, device, loss_function):
    model.eval()
    metric_logger = MetricLogger()
    header = 'Test: '

    test_loss = 0.0
    acc = 0.0

    with torch.no_grad():
        for i, [test_img, test_label] in enumerate(metric_logger.log_every(test_loader, print_freq=1, header=header)):
            test_img, test_label = test_img.to(device), test_label.to(device)

            model_time = time.time()
            output = model(test_img)
            model_time = time.time() - model_time

            evaluator_time = time.time()
            loss = loss_function(output, test_label.long())
            test_loss_batch = loss.item()
            test_loss = (test_loss * i + test_loss_batch) / (i + 1)

            pred_label = torch.max(output, dim=1)[1]
            acc_batch = torch.eq(pred_label, test_label).sum().item() / test_label.shape[0]
            acc = (acc * i + acc_batch) / (i+1)
            evaluator_time = time.time() - evaluator_time

            metric_logger.update(test_loss=test_loss)
            metric_logger.update(acc=acc)
            metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    return test_loss, acc


'''
由于训练刚开始时，训练数据计算出的梯度 grad 可能与期望方向相反，所以此时采用较小的学习率 learning rate，
随着迭代次数增加，学习率 lr 线性增大，增长率为 1/warmup_steps；
迭代次数等于 warmup_steps 时，学习率为初始设定的学习率；

'''
def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            # warmup后lr倍率因子从1 -> 0
            # 参考deeplab_v2: Learning rate policy
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)
