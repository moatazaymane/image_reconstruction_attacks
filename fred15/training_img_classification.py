import random
from code.fred15.models import Softmax_Model
import torch
from config import at_n_people as nb_pers
from torch import optim
from torch.utils.data import DataLoader
from utils import prepare_data_fred_at2


def training_loop(X, y, X_test, y_test, iterations, learning_rate, log_freq=5, seed=-1):

    ml_model = Softmax_Model(X.shape[1], 40)
    optimizer = optim.Adam(params=ml_model.parameters(), lr=learning_rate)
    criterion = torch.nn.NLLLoss()

    DL = list(zip(X, y))
    torch.random.manual_seed(seed)
    random.Random(seed).shuffle(DL)
    dl = DataLoader(DL, batch_size=16, shuffle=True)
    accuracies = []
    training_accuracies = []
    epochs = []
    st = 0
    for epoch in range(iterations):
        for xb, yb in dl:

            outputs = ml_model(xb)
            loss = criterion(outputs.view(-1, 40), yb.view(-1).long())

            loss.backward()
            optimizer.step()
            st += 1
            optimizer.zero_grad()

        if epoch % log_freq == 0:
            print(
                f"Training Loss after iteration {epoch} of gradient descent is {loss.item():.2f}"
            )
            accuracy = val_loop(ml_model, X_test, y_test, epoch)
            train_accuracy = val_loop(ml_model, xb, yb, epoch, train=True)

            epochs.append(epoch)
            accuracies.append(accuracy)
            training_accuracies.append(train_accuracy)
            if len(accuracies) >= 5 and abs(accuracies[-5] - accuracies[-1]) < 1e-2:
                break

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": ml_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": st,
        },
        "models/softmax_model",
    )

    return epochs, training_accuracies, accuracies


def val_loop(ml_model, X_test, y_test, iteration: int, train=False):

    with torch.no_grad():
        outputs = ml_model(X_test)

        sftmax = torch.exp(outputs)

        predictions = torch.argmax(sftmax, dim=1)

        accuracy = torch.sum(
            torch.where(predictions.float().view(-1) == y_test.view(-1).float(), 1, 0)
        )

        if not train:
            print(
                f"Test Set Accuracy after iteration {iteration} of gradient descent is : {accuracy / X_test.shape[0]:.2f}"
            )
        return accuracy


if __name__ == "__main__":

    X, y, X_test, y_test = prepare_data_fred_at2(1, test_size=100)

    X, X_test = X.reshape((X.shape[0], X.shape[1] * X.shape[2])), X_test.reshape(
        (X_test.shape[0], X.shape[1] * X.shape[2])
    )

    _, _, _ = training_loop(
        torch.from_numpy(X).float(),
        torch.from_numpy(y),
        torch.from_numpy(X_test).float(),
        torch.from_numpy(y_test),
        iterations=1000,
        learning_rate=2e-4,
        log_freq=10,
        seed=0,
    )
