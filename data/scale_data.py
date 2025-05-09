import torch


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def scale_data(df_train, df_test):
    X_train, y_train = df_train.drop("Transported", axis=1), df_train["Transported"]

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
    X_train = torch.tensor(preprocessor.fit_transform(X_train), dtype=torch.float64)
    X_test = torch.tensor(
        preprocessor.transform(df_test.drop("PassengerId", axis=1)), dtype=torch.float64
    )
    y_train = torch.tensor(y_train.values, dtype=torch.float64).unsqueeze(1)

    return X_train, X_test, y_train
