from datetime import datetime
from pathlib import Path
import os
import tempfile

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from ray import train, tune
from ray.train import Checkpoint, RunConfig
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch

from data import get_datasets
from models import RNN

global device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def train_epoch(model, dataloader: DataLoader, criterion, optimizer: torch.optim, epoch_i: int, verbose=False):
    running_loss = 0.0
    correct, total = 0, 0

    model.train()
    for i, data in enumerate(dataloader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        # Forward + backward + optimize
        out = model(inputs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        # Compute training accuracy
        _, predictions = torch.max(out, 1)
        total += labels.size(0)
        correct += (predictions == labels).sum().item()
        running_loss += loss.item()

        if i % 5 == 0:
            print(f'[{epoch_i}, {i + 1:5d}] loss: {running_loss / 100:.3f}') if verbose else 0
            running_loss = 0.0
    accuracy = correct / total
    print(f"Train Acc (epoch={epoch_i+1}): {accuracy}") if verbose else 0
    return {"accuracy": accuracy}

def validate_epoch(model, dataloader: DataLoader, criterion):
    validation_loss = 0.0
    correct, total = 0, 0
    val_steps = 0

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            out = model(inputs)
            _, predictions = torch.max(out, 1)
            total += labels.size(0)
            correct += (predictions == labels).sum().item()

            loss = criterion(out, labels)
            validation_loss += loss.cpu().numpy()
            val_steps += 1

    accuracy = correct / total
    print(f"Validation: \n Test Accuracy = {accuracy} \n Total loss = {validation_loss}")
    return {"accuracy": accuracy, "loss": validation_loss/val_steps}

global NR_CLASSES   
NR_CLASSES = 4

def train_rnn(config):
    rnn = RNN(input_size=248, 
              hidden_size=config["hidden_size"],
              num_layers=config["nr_layers"],
              output_size=NR_CLASSES)
    rnn.to(device)

    criterion = CrossEntropyLoss()
    optimizer = SGD(rnn.parameters(), lr=config["learning_rate"], momentum=0.9)

    dataloader_train = DataLoader(dataset=dataset_train, batch_size=int(config["batch_size"]))
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=int(config["batch_size"]))

    # Load existing checkpoint through `get_checkpoint()` API
    if train.get_checkpoint():
        loaded_checkpoint = train.get_checkpoint()
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
            model_state, optimizer_state = torch.load(
                os.path.join(loaded_checkpoint_dir, "checkpoint.pt")
            )
            rnn.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)

    for epoch in range(config["nr_epochs"]):
        train_result = train_epoch(model=rnn, dataloader=dataloader_train, criterion=criterion,
                    optimizer=optimizer, epoch_i=epoch)
        test_result = validate_epoch(model=rnn, dataloader=dataloader_test, criterion=criterion)
 
        # Save a checkpoint. Automatically registered with Ray Tune and will potentially be accessed through in ``get_checkpoint()`` in future iterations.
        # Note to save a file like checkpoint, you still need to put it under a directory to construct a checkpoint.
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            path = os.path.join(temp_checkpoint_dir, "checkpoint.pt")
            torch.save(
                (rnn.state_dict(), optimizer.state_dict()), path
            )
            checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
            train.report(
                metrics={"train_accuracy": train_result["accuracy"], "test_accuracy": test_result["accuracy"], "loss": test_result["loss"]},
                checkpoint=checkpoint,
            )

def test_best_rnn(best_result):
    best_model = RNN(input_size=248, 
                    hidden_size=best_result.config["hidden_size"],
                    num_layers=best_result.config["num_layers"],
                    output_size=NR_CLASSES)
    best_model.to(device)

    checkpoint_path = os.path.join(best_result.checkpoint.to_directory(), "checkpoint.pt")
    print(f"Best model checkpoint path: {checkpoint_path}")
    model_state, optimizer_state = torch.load(checkpoint_path)
    best_model.load_state_dict(model_state)

    dataloader_test = DataLoader(dataset=dataset_test, batch_size=int(best_result.config["batch_size"]))
    test_result = validate_epoch(model=best_model, dataloader=dataloader_test, criterion=CrossEntropyLoss())
    print(f"Best trial test set accuracy: {test_result['accuracy']}")


def main(subject="intra-subject", num_samples=10, max_num_epochs=10, grace_period=2):
    global dataset_train
    global dataset_test
    dataset_train, dataset_test = get_datasets(type=subject)

    # Hyper-parameter grid/distributions of the model
    """ config = {
        "subject": tune.choice([subject]),
        "hidden_size": tune.qrandint(4, 249, 4),
        "nr_layers": tune.randint(2, 5),
        "kernel_size": tune.choice([2, 3, 4]),
        "learning_rate": tune.loguniform(1e-4, 1e-2),
        "batch_size": tune.choice([2]),
        "nr_epochs": tune.randint(2,5) if subject=="intra-subject" else tune.randint(3,7) 
    } """
    config = {
        "subject": tune.choice([subject]),
        "hidden_size": tune.choice([128]),
        "nr_layers": tune.choice([1,2]),
        "kernel_size": tune.choice([2, 3, 4]),
        "learning_rate": tune.loguniform(1e-4, 1e-2),
        "batch_size": tune.choice([2]),
        "nr_epochs": tune.randint(2,5) if subject=="intra-subject" else tune.randint(3,7) 
    }
    # AsyncHyperBand enables aggressive early stopping of bad trials.
    scheduler = ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=grace_period,
        reduction_factor=2)
    # Default algorithm: BasicVariantGenerator
    algo = OptunaSearch() # Optuma algorithm: SearchGenerator

    # Define the tuner that optimizes train_tcn function
    tuner = tune.Tuner(
        tune.with_resources(train_rnn, {"cpu":6, "gpu":1}), 
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            search_alg = algo,
            scheduler=scheduler,
            num_samples=num_samples,
        ),
        param_space=config,
        run_config=RunConfig(name="tcn_"+subject+"_tuning_"+datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    )
    results = tuner.fit()

    # Retrieve and show best result
    best_result = results.get_best_result("loss", "min")
    print("Best trial config: {}".format(best_result.config))
    print("Best trial final validation loss: {}".format(
        best_result.metrics["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_result.metrics["test_accuracy"]))
    test_best_rnn(best_result)

main(subject="cross-subject", num_samples=3, max_num_epochs=10, grace_period=2)


"""

╭─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ Trial name           status       subject           hidden_size     nr_layers     kernel_size     learning_rate     batch_size     nr_epochs     iter     total time (s)     train_accuracy     test_accuracy      loss │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ train_tcn_d49fefa0   TERMINATED   cross-subject             128             2               4       0.002939                 2             5        5            3930.46           1                 0.479167   1.20172 │
│ train_tcn_257386be   TERMINATED   cross-subject             128             2               2       0.00398008               2             5        5            3207.73           1                 0.5        1.19241 │
│ train_tcn_89f80007   TERMINATED   cross-subject             128             1               3       0.000708298              2             5        2            4148.78           0.546875          0.583333   1.24529 │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

Best trial config: {'subject': 'cross-subject', 'hidden_size': 128, 'nr_layers': 2, 'kernel_size': 2, 'learning_rate': 0.003980076208247994, 'batch_size': 2, 'nr_epochs': 5}
Best trial final validation loss: 1.1924052213629086
Best trial final validation accuracy: 0.5

"""