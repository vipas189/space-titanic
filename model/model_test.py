import torch
from configs import Configs


def test(model, X_test, test_df, history):
    with torch.no_grad():
        model.load_state_dict(history.get("best_model_wts"))
        model.eval()
        outputs = model(X_test)
        predicted = (outputs > 0.5).bool()
        test_df["Transported"] = predicted.cpu().numpy()
        test_df[["PassengerId", "Transported"]].to_csv(
            f"csv_files/submition_val_acc-{history.get('best_val_accuracy'):.4f}lr-{Configs.LR}batch{Configs.BATCH_SIZE}.csv",
            index=False,
        )
