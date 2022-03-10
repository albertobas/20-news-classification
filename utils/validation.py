import matplotlib.pyplot as plt
import numpy as np
import random
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

SEED = 29
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

plt.rcParams.update({'axes.grid': True, 'axes.edgecolor': '#ffffff', 'axes.facecolor': '#fafafa',
                     'axes.labelcolor': '#3f3f3f', 'figure.facecolor': '#ffffff', 'font.size': 15,
                     'grid.color': '#dddddd', 'legend.edgecolor': '#ffffff', 'legend.facecolor': '#ffffff',
                     'legend.fontsize': 13, 'xtick.color': '#3f3f3f', 'xtick.labelsize': 14,
                     'ytick.color': '#3f3f3f', 'ytick.labelsize': 14})


def train(model, frame_train, frame_val, params_train, params_eval, plot=True):
    if torch.cuda.is_available():
        model = model.cuda()
    criterion = getattr(torch.nn.functional, params_train['criterion']['name'])
    optimizer = getattr(torch.optim, params_train['optimizer']['name'])(
        model.parameters(), **params_train['optimizer']['config'])
    if 'scheduler' in params_train.keys():
        scheduler = getattr(torch.optim.lr_scheduler, params_train['scheduler']['name'])(
            optimizer, **params_train['scheduler']['config'])
        if ((params_train['scheduler']['step']['metric'] != 'val_loss') & (params_train['scheduler']['step']['metric'] != 'val_acc')):
            raise ValueError(
                "The scheduler metrics allowed are 'val_loss' and 'val_acc'.")
    history_losses, history_accuracies, total_running_loss = [], [], []
    try:
        for epoch in range(1, params_train['epochs']+1):
            running_loss, losses, accuracies = _train(
                model, criterion, optimizer, frame_train, frame_val, params_train['data_loader'], params_eval, epoch, scheduler)
            total_running_loss.append(running_loss)
            history_losses += losses,
            history_accuracies += accuracies,

            if 'scheduler' in params_train.keys():
                scheduler_metric = losses[1] if (
                    params_train['scheduler']['step']['metric'] == 'val_loss') else accuracies[1]
                scheduler.step(scheduler_metric)
        if plot:
            _plot(total_running_loss, history_losses, history_accuracies)
    except KeyboardInterrupt:
        if (epoch > 0) & plot:
            _plot(total_running_loss, history_losses, history_accuracies)
        else:
            pass


def _train(model, criterion, optimizer, frame_train, frame_val, params_train, params_eval, epoch, scheduler=None):
    model.train()
    running_loss = []
    loop = tqdm(DataLoader(frame_train, **params_train), ascii=True, unit='step', ncols=110, position=0, leave=True,
                bar_format='{desc}{percentage:3.0f}%|{bar:10}| {n_fmt}/{total_fmt} ''{elapsed}''{postfix}')
    last_step = int(frame_train.data.size()[0]/params_train['batch_size'])
    for i, (data, labels) in enumerate(loop):
        if torch.cuda.is_available():
            data, labels = data.cuda(), labels.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        running_loss.append(loss.item())
        loop.set_description('Epoch {}'.format(epoch))
        if i == last_step:
            losses, accuracies = _evaluate(
                model, criterion, [frame_train, frame_val], params_eval)
            loss = str(round(losses[0], 4))
            acc = str(round(accuracies[0], 2))
            val_loss = str(round(losses[1], 4))
            val_acc = str(round(accuracies[1], 2))
            lr = str(round(optimizer.param_groups[0]['lr'], 7))
            if scheduler is None:
                loop.set_postfix(loss=loss, acc=acc,
                                 val_loss=val_loss, val_acc=val_acc)
            else:
                loop.set_postfix(loss=loss, acc=acc,
                                 val_loss=val_loss, val_acc=val_acc, _lr=lr)
    return running_loss, losses, accuracies


def _evaluate(model, criterion, frames, params):
    model.eval()
    losses, accuracies = (), ()
    for frame in frames:
        outputs = torch.Tensor(0)
        y = torch.LongTensor(0)
        for data, labels in DataLoader(frame, **params):
            if torch.cuda.is_available():
                data, labels = data.cuda(), labels.cuda()
            output = model(data)
            outputs = torch.cat((outputs, output.data))
            y = torch.cat((y, labels))
        _, y_pred = torch.max(outputs.data, 1)
        loss = criterion(outputs, y)
        losses += loss.item(),
        accuracies += (100 * sum(torch.eq(y_pred, y).tolist()) /
                       frame.data.size(0)),
    return losses, accuracies


def _plot(running_losses, losses, accuracies):
    epochs = len(losses)
    epochs_index = list(range(1, epochs+1))
    epochs_tick_labels = [str(x) if x % 2 > 0 else '' for x in epochs_index] if len(
        epochs_index) > 15 else epochs_index
    running_losses = [
        loss for sublist in running_losses for loss in sublist]
    ylim_loss_upper = np.ceil(
        running_losses[0]) if running_losses[0] >= 3.5 else 3.5
    _, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=False, figsize=(20, 5))
    ax1.set(xlabel="Steps", ylabel="Running Loss", ylim=((0, ylim_loss_upper)))
    ax2.set(xlabel="Epochs", ylabel="Loss", xticks=epochs_index,
            xticklabels=epochs_tick_labels, xlim=(1, epochs), ylim=(0, ylim_loss_upper))
    ax3.set(xlabel="Epochs", ylabel="Accuracy", xticks=epochs_index,
            xticklabels=epochs_tick_labels, xlim=(1, epochs), ylim=(0, 100))
    ax1.plot(running_losses, color='#003d7a')
    # ['#214979', '#8eacc4']
    for i, c in enumerate(['#003060', '#47a2ff']):
        loss = [sublist[i] for sublist in losses]
        acc = [sublist[i] for sublist in accuracies]
        ax2.plot(epochs_index, loss, color=c)
        ax3.plot(epochs_index, acc, color=c)
    ax2.legend(('Training', 'Validation'), loc='upper right')
    ax3.legend(('Training', 'Validation'), loc='lower right')
    plt.show()
