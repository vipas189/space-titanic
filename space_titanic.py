import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import optuna

from pytorchmodel import PyTorchModel
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder


train_df = pd.read_csv("csv_files/space_titanic_train.csv")


def prepare_train_data():
    train_df = pd.read_csv("csv_files/space_titanic_train.csv").copy()
    train_df["Deck"] = train_df["Cabin"].str.split("/").str.get(0)
    train_df["Deck_Num"] = pd.to_numeric(
        train_df["Cabin"].str.split("/").str.get(1), errors="coerce"
    )
    train_df["Deck_Side"] = train_df["Cabin"].str.split("/").str.get(2)

    train_df["Group"] = train_df["PassengerId"].str.split("_").str.get(0).astype(int)
    train_df["GroupSize"] = (
        train_df["PassengerId"].str.split("_").str.get(1).astype(int)
    )

    train_df.fillna(
        {
            "HomePlanet": "Unknown",
            "CryoSleep": False,
            "Deck": train_df["Deck"].mode()[0],
            "Deck_Num": train_df["Deck_Num"].mode()[0],
            "Deck_Side": train_df["Deck_Side"].mode()[0],
            "Destination": "Unknown",
            "Age": train_df["Age"].mean(),
            "VIP": False,
            "RoomService": 0,
            "FoodCourt": 0,
            "ShoppingMall": 0,
            "Spa": 0,
            "VRDeck": 0,
        },
        inplace=True,
    )

    train_df = train_df.drop(["PassengerId", "Name", "Cabin"], axis=1)
    train_df.to_csv("csv_files/space_titanic_train_copy.csv", index=False)

    return train_df


def prepare_test_data():
    test_df = pd.read_csv("csv_files/space_titanic_test.csv").copy()
    test_df["Deck"] = test_df["Cabin"].str.split("/").str.get(0)
    test_df["Deck_Num"] = pd.to_numeric(
        test_df["Cabin"].str.split("/").str.get(1), errors="coerce"
    )
    test_df["Deck_Side"] = test_df["Cabin"].str.split("/").str.get(2)

    test_df["Group"] = test_df["PassengerId"].str.split("_").str.get(0).astype(int)
    test_df["GroupSize"] = test_df["PassengerId"].str.split("_").str.get(1).astype(int)

    test_df.fillna(
        {
            "HomePlanet": "Unknown",
            "CryoSleep": False,
            "Deck": test_df["Deck"].mode()[0],
            "Deck_Num": test_df["Deck_Num"].mode()[0],
            "Deck_Side": test_df["Deck_Side"].mode()[0],
            "Destination": "Unknown",
            "Age": test_df["Age"].mean(),
            "VIP": False,
            "RoomService": 0,
            "FoodCourt": 0,
            "ShoppingMall": 0,
            "Spa": 0,
            "VRDeck": 0,
        },
        inplace=True,
    )

    test_df = test_df.drop(["Name", "Cabin"], axis=1)
    test_df.to_csv("csv_files/space_titanic_test_copy.csv")

    return test_df


def scale_data(df_train, df_test):
    X_train, X_vall, y_train, y_val = train_test_split(
        df_train.drop("Transported", axis=1),
        df_train["Transported"],
        test_size=0.25,
        stratify=df_train["Transported"],
        random_state=42,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                X_train.select_dtypes(include=["float64", "int64"]).columns.tolist(),
            ),
            (
                "cat",
                OneHotEncoder(
                    drop="first", handle_unknown="ignore", sparse_output=False
                ),
                X_train.select_dtypes(include="object").columns.tolist(),
            ),
        ],
        remainder="passthrough",
    )

    X_train = torch.tensor(preprocessor.fit_transform(X_train), dtype=torch.float32)
    X_val = torch.tensor(preprocessor.transform(X_vall), dtype=torch.float32)
    X_test = torch.tensor(
        preprocessor.transform(df_test.drop("PassengerId", axis=1)), dtype=torch.float32
    )
    y_train = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    y_val = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)

    return X_train, X_val, X_test, y_train, y_val


def pytorch_model(
    trial,
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    test_df,
    epochs=999,
    batch_size=128,
    num_workers=0,
    stop_val_loss_not_improving=20,
    lr=0.004,
):
    train_data = torch.utils.data.TensorDataset(X_train, y_train)
    val_data = torch.utils.data.TensorDataset(X_val, y_val)
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=batch_size, num_workers=num_workers, pin_memory=True
    )

    device = torch.device("cpu")
    model = PyTorchModel(X_train.shape[1]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.001)
    # loss_fn = nn.BCEWithLogitsLoss()
    loss_fn = nn.BCELoss()
    print(os.cpu_count())
    print(torch.__version__)
    print(torch.version.cuda)

    best_val_accuracy = 0.0
    best_val_loss = float("inf")
    not_improved_val_acc = 0
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "best_val_scores": [],
    }

    print("Starting Training...")
    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0
        correct_train = 0
        total_train = 0

        for _, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            running_train_loss += loss.item() * inputs.size(0)
            predicted = (outputs > 0.5).float()
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        epoch_train_acc = correct_train / total_train
        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)

        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = loss_fn(outputs, labels)

                running_val_loss += loss.item() * inputs.size(0)
                predicted = (outputs > 0.5).float()
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        epoch_val_acc = correct_val / total_val
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)
        print(
            f"Epoch [{epoch+1}/{epochs}] | "
            f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f} | "
            f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f}"
        )
        if epoch_val_acc > best_val_accuracy:
            best_val_loss = epoch_val_loss
            best_val_accuracy = epoch_val_acc
            not_improved_val_acc = 0
            best_val_acc_list = history.get("best_val_scores")
            best_val_acc_list.append(best_val_accuracy)
            torch.save(
                model,
                f"models/model_val_acc-{best_val_accuracy:.6f}lr-{lr}batch{batch_size}.pth",
            )
            try:
                os.remove(
                    f"models/model_val_acc-{best_val_acc_list[len(best_val_acc_list)-2]:.6f}lr-{lr}batch{batch_size}.pth"
                )
            except FileNotFoundError:
                pass
            except IndexError:
                pass
            continue

        not_improved_val_acc += 1

        if not_improved_val_acc == stop_val_loss_not_improving:
            print("Model stopped, no more improvement in validation accuracy")
            break
    print(f"Best\nVal_Acc: {best_val_accuracy}\nVal_Loss: {best_val_loss}")
    with torch.no_grad():
        model = torch.load(
            f"models/model_val_acc-{best_val_accuracy:.6f}lr-{lr}batch{batch_size}.pth",
            weights_only=False,
        )
        model.eval()
        outputs = model(X_test)
        predicted = (outputs > 0.5).bool()
    test_df["Transported"] = predicted.cpu().numpy()
    test_df[["PassengerId", "Transported"]].to_csv(
        f"csv_files/submition_val_acc-{best_val_accuracy:.6f}lr-{lr}batch{batch_size}.csv",
        index=False,
    )


def optimization(X_train, X_val, X_test, y_train, y_val, test_df):
    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: pytorch_model(
            trial, X_train, X_val, X_test, y_train, y_val, test_df
        ),
        n_trials=20,
    )


def main():
    train_df = prepare_train_data()
    test_df = prepare_test_data()
    X_train, X_val, X_test, y_train, y_val = scale_data(train_df, test_df)
    optimization(X_train, X_val, X_test, y_train, y_val, test_df)


if __name__ == "__main__":
    main()
