import pandas as pd
import numpy as np

import optuna

import tensorflow as tf

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from tensorflow.keras.layers import Dense  # pyright: ignore
from tensorflow.keras.models import Sequential  # pyright: ignore
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint  # pyright: ignore
from tensorflow.keras.optimizers import Adam # pyright: ignore

from optuna.integration import TFKerasPruningCallback


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


def split_data(train_df):
    X = train_df.drop("Transported", axis=1)
    y = train_df["Transported"]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    return X_train, X_val, y_train, y_val


def scale_data(X_train, X_val, test_df):
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
    X_test = preprocessor.transform(test_df.drop("PassengerId", axis=1))
    return X_train, X_val, X_test


def sequential_model(trial, X_train, X_val, y_train, y_val):
    units = trial.suggest_int("units", 32, 128, step=32)
    epochs = trial.suggest_int("epochs", 8, 48, step=8)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 24, 32])
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
    #     ModelCheckpoint(
    #     filepath="models/sequential_model_NN2.keras",
    #     monitor='val_accuracy',
    #     mode='max',
    #     save_best_only=True,
    #     save_weights_only=False,
    #     verbose=1
    # )
    ]

    model = Sequential()
    model.add(Dense(units, input_dim=X_train.shape[1], activation="relu"))
    model.add(Dense(units, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        validation_data=(X_val, y_val),
    )
    best_val_accuracy = max(history.history['val_accuracy'])
    return best_val_accuracy


def optuna_optimization(X_train, X_val, y_train, y_val):
    study = optuna.create_study(
        direction="maximize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
    )
    study.optimize(
        lambda trial: sequential_model(trial, X_train, X_val, y_train, y_val),
        n_trials=20
    )
    print(f"Best params: {study.best_trial.params}")
    print("Best val accuracy: {:.4f}".format(study.best_value))
    
    best_params = study.best_trial.params
    print("\nBest hyperparameters found:")
    print(best_params)


    best_units = best_params['units']
    best_learning_rate = best_params['learning_rate']
    final_epochs = 50
    # Build the final model with the best hyperparameters
    final_model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train.shape[1],)), # Use total feature count
        tf.keras.layers.Dense(best_units, activation='relu'),
        tf.keras.layers.Dropout(0.2), # Keep dropout if used in tuning
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    final_optimizer = tf.keras.optimizers.Adam(learning_rate=best_learning_rate)
    final_model.compile(optimizer=final_optimizer,
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

    print("\nTraining final model with best hyperparameters...")

    final_model.fit(X_train, y_train,
                    epochs=final_epochs,
                    batch_size=best_params['batch_size'],
                    verbose=1)

    final_model_save_path = "models/best_overall_sequential_model.keras"
    final_model.save(final_model_save_path)
    print(f"\nSaved the best overall model to: {final_model_save_path}")
    
def submition(X_test, test_df):
    loaded_model = tf.keras.models.load_model("models/sequential_model_NN.keras")
    predict = loaded_model.predict(X_test)
    predictions_bool = (predict > 0.5).flatten()

    submission_df = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Transported": predictions_bool
    })

    print(f"Saving submission file to {"csv_files/submition.csv"}")
    submission_df.to_csv("csv_files/submition.csv", index=False)
    print("Submission file created successfully!")

def final(X_train, X_val, y_train, y_val, X_test, test_df):
    units = 96
    epochs = 8
    batch_size = 8
    model = Sequential()
    model.add(Dense(units, input_dim=X_train.shape[1], activation="relu"))
    model.add(Dense(units, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])
    
    model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
    )
    predict = model.predict(X_test)
    predictions_bool = (predict > 0.5).flatten()

    submission_df = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Transported": predictions_bool
    })

    print(f"Saving submission file to {"csv_files/submition3.csv"}")
    submission_df.to_csv("csv_files/submition.csv3", index=False)
    print("Submission file created successfully!")
    

def main():
    train_df = prepare_train_data()
    test_df = prepare_test_data()
    X_train, X_val, y_train, y_val = split_data(train_df)
    X_train, X_val, X_test = scale_data(X_train, X_val, test_df)
    # optuna_optimization(X_train, X_val, y_train, y_val)
    # submition(X_test, test_df)
    final(X_train, X_val, y_train, y_val, X_test, test_df)


if __name__ == "__main__":
    main()
