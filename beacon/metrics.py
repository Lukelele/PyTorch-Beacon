import torch


def categorical_accuracy(y_pred_logit, y_true):
    """
    Calculates the categorical accuracy of the predicted logits.

    Args:
    - y_pred_logit: predicted logits
    - y_true: true labels

    Returns:
    - accuracy: categorical accuracy of the predicted logits
    """
    y_prob = torch.softmax(y_pred_logit, dim=1)
    y_pred = torch.argmax(y_prob, dim=1)
    correct = (y_pred == y_true).float()
    accuracy = correct.sum() / len(correct)

    return accuracy


def binary_accuracy(y_pred_logit, y_true):
    """
    Calculates the binary accuracy of the predicted logits.

    Args:
    - y_pred_logit: predicted logits
    - y_true: true labels

    Returns:
    - accuracy: binary accuracy of the predicted logits
    """
    y_prob = torch.sigmoid(y_pred_logit)
    y_pred = torch.round(y_prob)
    correct = (y_pred == y_true).float()
    accuracy = correct.sum() / len(correct)

    return accuracy
