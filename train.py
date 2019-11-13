import torch
import time
import copy
from tqdm import tqdm

def train_network(model, n_epochs, optimizer, criterion, scheduler,
                  device, loaders):
    """
    Trains a model, prints intermediate results, returns a trained model.
    ---------
    Arguments:
    loaders: dictionary, has keys 'train', 'val', 'test';
            items are pytorch Dataloaders.
    """
    start_time = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    loss_history = {'train': [],
                    'val': []}
    acc_history = {'train': [],
                   'val': []}

    for epoch in range(n_epochs):
        epoch_start_time = time.time()
        print('-' * 20)
        print('Epoch {} / {}'.format(epoch, n_epochs))

        for phase in ['train', 'val']:

            if phase == 'train':
                scheduler.step()
                model.train()
            elif phase == 'val':
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(loaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase=='train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                preds = torch.argmax(outputs, dim=1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(loaders[phase].dataset)
            epoch_acc = running_corrects.float() / len(loaders[phase].dataset)

            print('{} loss: {:.4f}'.format(phase, epoch_loss))
            print('{} accuracy: {:.4f}'.format(phase, epoch_acc))
            acc_history[phase].append(epoch_acc)
            loss_history[phase].append(epoch_loss)

        if phase == 'val' and epoch_acc > best_acc:
            best_model_wts = copy.deepcopy(model.state_dict)
            best_acc = epoch_acc

        epoch_time = time.time() - epoch_start_time
        print("Epoch time: {:.0f}m, {:.0f}s".format(epoch_time // 60, epoch_time % 60))

    elapsed_time = time.time() - start_time
    print('Model trained in {:.0f}m {:.0f}s'.format(elapsed_time // 60, elapsed_time % 60))
    print('Best accuracy: {:.4f}'.format(best_acc))
    model.load_state_dict(best_model_wts)
    return model, acc_history, loss_history