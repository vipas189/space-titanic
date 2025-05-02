import pandas as pd
import numpy as np

import optuna

import tensorflow as tf

import os

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from tensorflow.keras import Model, Input  # pyright: ignore
from tensorflow.keras.layers import Dense  # pyright: ignore
from tensorflow.keras.models import Sequential  # pyright: ignore
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint  # pyright: ignore
from tensorflow.keras.optimizers import Adam # pyright: ignore

from optuna.integration import TFKerasPruningCallback

import xgboost as xgb
from sklearn.metrics import accuracy_score


train_df = pd.read_csv("csv_files/space_titanic_train.csv")


def prepare_train_data():
    train_df = pd.read_csv("csv_files/space_titanic_train.csv").copy()
    train_df["Deck"] = train_df["Cabin"].str.split("/").str.get(0)
    train_df["Deck_Num"] = pd.to_numeric(train_df["Cabin"].str.split("/").str.get(1), errors='coerce')
    train_df["Deck_Side"] = train_df["Cabin"].str.split("/").str.get(2)
    
    train_df["Group"] = train_df["PassengerId"].str.split("_").str.get(0).astype(int)
    train_df["GroupSize"] = train_df["PassengerId"].str.split("_").str.get(1).astype(int)
    
    train_df.fillna({"HomePlanet": "Unknown",
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
                     }, inplace=True)

    train_df = train_df.drop(["PassengerId", "Name", "Cabin"], axis=1)
    train_df.to_csv("csv_files/space_titanic_train_copy.csv", index=False)

    return train_df


def prepare_test_data():
    test_df = pd.read_csv("csv_files/space_titanic_test.csv").copy()
    test_df["Deck"] = test_df["Cabin"].str.split("/").str.get(0)
    test_df["Deck_Num"] = pd.to_numeric(test_df["Cabin"].str.split("/").str.get(1), errors='coerce')
    test_df["Deck_Side"] = test_df["Cabin"].str.split("/").str.get(2)
    
    test_df["Group"] = test_df["PassengerId"].str.split("_").str.get(0).astype(int)
    test_df["GroupSize"] = test_df["PassengerId"].str.split("_").str.get(1).astype(int)
    
    test_df.fillna({"HomePlanet": "Unknown",
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
                     }, inplace=True)

    test_df = test_df.drop(["Name", "Cabin"], axis=1)
    test_df.to_csv("csv_files/space_titanic_test_copy.csv")

    return test_df


def scale_data(df_train, df_test):
    X_train, X_val, y_train, y_val = train_test_split(df_train.drop("Transported", axis=1), df_train["Transported"], test_size=0.10, stratify=df_train["Transported"], random_state=42)
    
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
    
    X_train = preprocessor.fit_transform(X_train)
    X_val = preprocessor.transform(X_val)
    X_test = preprocessor.transform(df_test.drop("PassengerId", axis=1))
    return X_train, X_val, y_train, y_val, X_test


def sequential_model(trial, X_train, X_val, y_train, y_val):
    epochs = trial.suggest_categorical("epochs", [8, 10, 12, 14])
    batch_size = trial.suggest_categorical("batch_size", [18, 20, 24])
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        ),
        TFKerasPruningCallback(trial, "val_loss"),
        TensorBoard(
            log_dir="logs/fit/" + str(trial.number),
            histogram_freq=1,
            write_graph=True,
            write_images=True,
        ),
    ]

    model = Sequential()
    model.add(Dense(128, input_dim=X_train.shape[1], activation="relu"))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        validation_data=(X_val, y_val)
    )
    best_val_accuracy = max(history.history['val_accuracy'])
    return best_val_accuracy

def build_functional_model(trial, X_train, X_val, y_train, y_val):
    epochs = trial.suggest_categorical("epochs", [8, 10, 12, 14])
    batch_size = trial.suggest_categorical("batch_size", [18, 20, 24])
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        ),
        TFKerasPruningCallback(trial, "val_loss"),
        TensorBoard(
            log_dir="logs/fit/" + str(trial.number),
            histogram_freq=1,
            write_graph=True,
            write_images=True,
        ),
    ]
    inputs = Input(shape=(X_train.shape[1],))
    x = Dense(128, activation="relu")(inputs)
    x = Dense(128, activation="relu")(x)
    outputs = Dense(1, activation="sigmoid")(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        validation_data=(X_val, y_val)
    )
    best_val_accuracy = max(history.history['val_accuracy'])
    return best_val_accuracy

def build_xgboost_model(trial, X_train, X_val, y_train, y_val):
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "use_label_encoder": False,
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
    }

    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)

    # preds = model.predict(X_val)
    # acc = accuracy_score(y_val, preds)
    y_pred = model.predict(X_val)
    y_pred = (y_pred > 0.5)  # For binary classification
    acc = accuracy_score(y_val, y_pred)
    return acc


def optuna_optimization(X_train, X_val, y_train, y_val):
    study = optuna.create_study(
        direction="maximize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
    )
    # study.optimize(
    #     lambda trial: sequential_model(trial, X_train, X_val, y_train, y_val),
    #     n_trials=12, n_jobs=-1
    # )
    # study.optimize(
    #     lambda trial: build_functional_model(trial, X_train, X_val, y_train, y_val),
    #     n_trials=12, n_jobs=-1
    # )
    study.optimize(
        lambda trial: build_xgboost_model(trial, X_train, X_val, y_train, y_val),
        n_trials=128, n_jobs=-1
    )
    print(f"Best params: {study.best_trial.params}")
    print("Best val accuracy: {:.4f}".format(study.best_value))
    

def main():
    train_df = prepare_train_data()
    test_df = prepare_test_data()
    X_train, X_val, y_train, y_val, X_test = scale_data(train_df, test_df)
    optuna_optimization(X_train, X_val, y_train, y_val)
    # submition(X_test, test_df)
    # final(X_train, X_val, y_train, y_val, X_test, test_df)


if __name__ == "__main__":
    main()
