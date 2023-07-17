import torch


def categorical_accuracy(y_pred_logit, y_true):
    y_prob = torch.softmax(y_pred_logit, dim=1)
    y_pred = torch.argmax(y_prob, dim=1)
    correct = (y_pred == y_true).float()
    accuracy = correct.sum() / len(correct)

    return accuracy


def binary_accuracy(y_pred_logit, y_true):
    y_prob = torch.sigmoid(y_pred_logit)
    y_pred = torch.round(y_prob)
    correct = (y_pred == y_true).float()
    accuracy = correct.sum() / len(correct)

    return accuracy
