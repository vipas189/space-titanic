import torch
from configs import Configs


def val(model, loss_fn, val_loader, history):
    history["running_val_loss"] = 0.0
    history["correct_val"] = 0
    history["total_val"] = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(Configs.DEVICE), labels.to(Configs.DEVICE)

            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            history["running_val_loss"] += loss.item() * inputs.size(0)
            predicted = (outputs > 0.5).float()
            history["total_val"] += labels.size(0)
            history["correct_val"] += (predicted == labels).sum().item()

    history["epoch_val_loss"] = history.get("running_val_loss") / len(
        val_loader.dataset
    )
    history["epoch_val_acc"] = history.get("correct_val") / history.get("total_val")
    history["val_loss"].append(history.get("epoch_val_loss"))
    history["val_acc"].append(history.get("epoch_val_acc"))


# print(f"Best\nVal_Acc: {best_val_accuracy}\nVal_Loss: {best_val_loss}")
# with torch.no_grad():
#     model = torch.load(
#         f"models/model_val_acc-{best_val_accuracy:.6f}lr-{lr}batch{batch_size}.pth",
#         weights_only=False,
#     )
#     model.eval()
#     outputs = model(X_test)
#     predicted = (outputs > 0.5).bool()
# test_df["Transported"] = predicted.cpu().numpy()
# test_df[["PassengerId", "Transported"]].to_csv(
#     f"csv_files/submition_val_acc-{best_val_accuracy:.6f}lr-{lr}batch{batch_size}.csv",
#     index=False,
# )
