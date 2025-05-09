from data.prepare_data import prepare_train_data, prepare_test_data
from data.scale_data import scale_data
from model.model_process import model_start


def main():
    train_df = prepare_train_data()
    test_df = prepare_test_data()
    X_train, X_test, y_train = scale_data(train_df, test_df)
    model_start(X_train, X_test, y_train, test_df)


if __name__ == "__main__":
    main()
