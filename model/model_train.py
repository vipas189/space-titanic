from configs import Configs


def train(model, loss_fn, optimizer, train_loader, history):
    history["running_train_loss"] = 0.0
    history["correct_train"] = 0
    history["total_train"] = 0
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(Configs.DEVICE), labels.to(Configs.DEVICE)

        outputs = model(inputs)
        loss = loss_fn(outputs, labels)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        history["running_train_loss"] += loss.item() * inputs.size(0)
        predicted = (outputs > 0.5).float()
        history["total_train"] += labels.size(0)
        history["correct_train"] += (predicted == labels).sum().item()

    history["epoch_train_loss"] = history.get("running_train_loss") / len(
        train_loader.dataset
    )
    history["epoch_train_acc"] = history.get("correct_train") / history.get(
        "total_train"
    )
    history["train_loss"].append(history.get("epoch_train_loss"))
    history["train_acc"].append(history.get("epoch_train_acc"))
