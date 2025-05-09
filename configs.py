import torch


class Configs:
    LR = 0.002
    # LR = 0.0015
    WEIGHT_DECAY = 0.001
    BATCH_SIZE = 128
    EARLY_STOP = 30
    EPOCHS = 9999
    DEVICE = torch.device("cpu")
