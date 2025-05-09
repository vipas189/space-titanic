import model.model_train as model_train
import model.model_val as model_val
import model.model_test as model_test
from configs import Configs
from torch.utils.data import TensorDataset
from sklearn.model_selection import StratifiedKFold
import torch.optim as optim
import torch.nn as nn
from model.pytorchmodel import PyTorchModel
from torch.utils.data import DataLoader, Subset
from model.early_stop import early_stop


def model_start(X_train, X_test, y_train, test_df):
    data = TensorDataset(X_train, y_train)
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    loss_fn = nn.BCELoss()

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "best_val_scores": [],
        "epoch_train_loss": 0,
        "epoch_train_acc": 0,
        "running_train_loss": 0,
        "correct_train": 0,
        "total_train": 0,
        "epoch_val_loss": 0,
        "epoch_val_acc": 0,
        "running_val_loss": 0,
        "correct_val": 0,
        "total_val": 0,
        "best_val_accuracy": 0,
        "best_val_loss": float("inf"),
        "not_improved_val_acc": 0,
        "best_model_wts": None,
    }

    print("Starting Training...")
    while True:
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train, y_train)):

            print(f"_____Fold: {fold}_____")

            train_data = Subset(data, train_idx)
            val_data = Subset(data, val_idx)
            train_loader = DataLoader(
                train_data, batch_size=Configs.BATCH_SIZE, shuffle=True
            )
            val_loader = DataLoader(
                val_data, batch_size=Configs.BATCH_SIZE, shuffle=False
            )
            model = PyTorchModel(X_train.shape[1]).to(Configs.DEVICE).double()
            optimizer = optim.AdamW(
                model.parameters(), lr=Configs.LR, weight_decay=Configs.WEIGHT_DECAY
            )

            for epoch in range(Configs.EPOCHS):
                model_train.train(model, loss_fn, optimizer, train_loader, history)
                model_val.val(model, loss_fn, val_loader, history)
                print(
                    f"Epoch [{epoch+1}/{Configs.EPOCHS}] | "
                    f"Train Loss: {history.get("epoch_train_loss"):.4f} | Train Acc: {history.get("epoch_train_acc"):.4f} | "
                    f"Val Loss: {history.get("epoch_val_loss"):.4f} | Val Acc: {history.get("epoch_val_acc"):.4f} - Best: {history.get("best_val_accuracy"):.4f}"
                )
                if early_stop(model, history):
                    break
            model_test.test(model, X_test, test_df, history)
