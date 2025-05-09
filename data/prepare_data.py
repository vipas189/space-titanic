import pandas as pd


def prepare_train_data():
    train_df = pd.read_csv("csv_files/space_titanic_train.csv")
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
