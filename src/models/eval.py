import torch
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

def get_accuracy(model, dataloader, topk=(1,5), device=None, mt=False):
    """
    Computes the class-wise and average (micro) accuracy@k
    for the specified values of k
    """
    dataset = dataloader.dataset
    total = len(dataset)
    maxk = max(topk)
    correct_count = {k: 0 for k in topk}

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_classes = len(dataloader.dataset.class_map)
    class_correct_count = {k: np.array([0 for n in range(n_classes)]) for k in topk}

    all_labels = np.array([dataset.class_map[x] for x in dataset.label_arr])
    class_totals = np.bincount(all_labels)
    # Handle missing classes
    class_totals[class_totals==0] = -1

    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images = images.to(device)
            if mt:
                labels = labels[0]
            labels = labels.to(device).long()
            outputs = model(images)
            if mt:
                outputs = outputs[0]
            _, pred = torch.topk(outputs, maxk, 1)
            pred = pred.t()
            correct = pred.eq(labels.view(1, -1).expand_as(pred))
            for k in topk:
                # See if the label is present any of the top k predictions
                for cp, cl in zip(pred.t(), labels):
                    if cl in cp[:k]:
                        class_correct_count[k][cl] += 1
                # Sum up correct predictions for current batch
                correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                correct_count[k] += correct_k

    avg_acc = {k: v.item()/total for k, v in correct_count.items()}
    class_acc = {k: v/class_totals for k, v in class_correct_count.items()}

    return avg_acc, class_acc

def generate_acc_df(avg_acc, class_acc, inv_map):
    """
    Returns a pandas DataFrame containing average and class-wise scores
    """
    df = pd.DataFrame(class_acc,
                      index=[inv_map[i] for i in range(len(inv_map))])

    avg_df = pd.DataFrame(avg_acc, index=['Average (Micro)'])
    df = pd.concat([avg_df, df])

    df = df.rename(columns={1: 'Top-1', 5: 'Top-5'})
    df *= 100
    df = df.round(2)

    return df
