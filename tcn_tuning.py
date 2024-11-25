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
from models import TCN

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

def train_tcn(config):
    tcn = TCN(input_size=248, output_size=NR_CLASSES, 
              num_channels=[config["hidden_units_per_level"]]*config["nr_levels"], 
              kernel_size=config["kernel_size"], dropout=0.2)
    tcn.to(device)

    criterion = CrossEntropyLoss()
    optimizer = SGD(tcn.parameters(), lr=config["learning_rate"], momentum=0.9)

    dataloader_train = DataLoader(dataset=dataset_train, batch_size=int(config["batch_size"]))
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=int(config["batch_size"]))

    # Load existing checkpoint through `get_checkpoint()` API
    if train.get_checkpoint():
        loaded_checkpoint = train.get_checkpoint()
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
            model_state, optimizer_state = torch.load(
                os.path.join(loaded_checkpoint_dir, "checkpoint.pt")
            )
            tcn.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)

    for epoch in range(config["nr_epochs"]):
        train_result = train_epoch(model=tcn, dataloader=dataloader_train, criterion=criterion,
                    optimizer=optimizer, epoch_i=epoch)
        test_result = validate_epoch(model=tcn, dataloader=dataloader_test, criterion=criterion)
 
        # Save a checkpoint. Automatically registered with Ray Tune and will potentially be accessed through in ``get_checkpoint()`` in future iterations.
        # Note to save a file like checkpoint, you still need to put it under a directory to construct a checkpoint.
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            path = os.path.join(temp_checkpoint_dir, "checkpoint.pt")
            torch.save(
                (tcn.state_dict(), optimizer.state_dict()), path
            )
            checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
            train.report(
                metrics={"train_accuracy": train_result["accuracy"], "test_accuracy": test_result["accuracy"], "loss": test_result["loss"]},
                checkpoint=checkpoint,
            )

def test_best_tcn(best_result):
    best_model = TCN(input_size=248, output_size=NR_CLASSES, 
                     num_channels=[best_result.config["hidden_units_per_level"]]*best_result.config["nr_levels"],
                     kernel_size=best_result.config["kernel_size"], dropout=0.2)
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
    config = {
        "subject": tune.choice([subject]),
        "hidden_units_per_level": tune.qrandint(40, 65, 5),
        "nr_levels": tune.randint(2, 5),
        "kernel_size": tune.choice([2, 3, 4]),
        "learning_rate": tune.loguniform(5e-4, 5e-3),
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
        tune.with_resources(train_tcn, {"cpu":6, "gpu":1}), 
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
    test_best_tcn(best_result)

main(subject="cross-subject", num_samples=10, max_num_epochs=10, grace_period=2)



"""
Number samples in experiment: 100
Trial name           status       subject           ...n_units_per_level     nr_levels     kernel_size     learning_rate     batch_size     nr_epochs     iter     total time (s)     train_accuracy     test_accuracy            loss

LOW LOSS
train_tcn_a3d4cfe6   TERMINATED   cross-subject                       45             3               2       0.000561879              2             4        4            511.075           1                 0.5625       0.869425
train_tcn_2d2b0b7a   TERMINATED   cross-subject                       45             3               3       0.000351467              2             5        5            625.738           1                 0.625        0.873043 
train_tcn_cd48832c   TERMINATED   cross-subject                       40             4               2       0.0004756                2             4        4            497.66            0.96875           0.625        0.880267
train_tcn_3ab7c65f   TERMINATED   cross-subject                       40             3               2       0.000630519              2             4        4            495.938           1                 0.604167     0.897765
train_tcn_eee70c29   TERMINATED   cross-subject                       40             5               2       0.000517272              2             4        4            506.613           0.984375          0.520833     0.902691

HIGH ACCURACY
train_tcn_5130973f   TERMINATED   cross-subject                       40             3               3       0.000270887              2             5        2            249.918           0.890625          0.666667     1.10665

Number samples in experiment: 10
╭───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ Trial name           status       subject           ...n_units_per_level     nr_levels     kernel_size     learning_rate     batch_size     nr_epochs     iter     total time (s)     train_accuracy     test_accuracy       loss │
├───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ train_tcn_e5cc7ae2   TERMINATED   cross-subject                       50             4               2       0.00114069               2             4        4            793.391           0.9375            0.520833   1.1309   │
│ train_tcn_56aa05d0   TERMINATED   cross-subject                       65             2               4       0.00200306               2             4        2            382.861           0.5               0.520833   1.33894  │
│ train_tcn_a0433374   TERMINATED   cross-subject                       40             4               2       0.00105122               2             4        4            726.64            0.984375          0.5        1.0764   │
│ train_tcn_46bb1854   TERMINATED   cross-subject                       60             4               2       0.000629492              2             4        4            793.338           1                 0.5625     0.943731 │
│ train_tcn_ae1f667b   TERMINATED   cross-subject                       55             2               4       0.0013796                2             5        2            383.378           0.609375          0.520833   1.078    │
│ train_tcn_59614658   TERMINATED   cross-subject                       45             2               3       0.00221983               2             6        2            374.858           0.484375          0.541667   1.28038  │
│ train_tcn_6cf50b57   TERMINATED   cross-subject                       60             3               4       0.000654624              2             6        6           1288.28            1                 0.583333   1.01687  │
│ train_tcn_7503b670   TERMINATED   cross-subject                       40             3               4       0.000510245              2             4        4            826.187           1                 0.541667   0.931713 │
│ train_tcn_43581a0a   TERMINATED   cross-subject                       45             4               4       0.00062295               2             6        6           1334.39            1                 0.5625     0.945393 │
│ train_tcn_e3afcbbf   TERMINATED   cross-subject                       45             3               2       0.00131566               2             3        2            426.078           0.484375          0.5        1.20727

Best trial config: {'subject': 'cross-subject', 'hidden_units_per_level': 40, 'nr_levels': 3, 'kernel_size': 4, 'learning_rate': 0.0005102452601658585, 'batch_size': 2, 'nr_epochs': 4}
Best trial final validation loss: 0.9317130471269289
Best trial final validation accuracy: 0.5416666666666666



Number samples in experiment: 40
Trial name           status       subject           ...n_units_per_level     nr_levels     kernel_size     learning_rate     batch_size     nr_epochs     iter     total time (s)     train_accuracy     test_accuracy            loss

train_tcn_d4af78fa   TERMINATED   intra-subject                       55             4               4       0.00403549               2             3        3           179.762             1                   1       0.000346025
train_tcn_efb0aba5   TERMINATED   intra-subject                       55             4               4       0.00430998               2             3        3           179.753             1                   1       0.0010093
train_tcn_4fe1cb9f   TERMINATED   intra-subject                       35             2               4       0.00522736               2             3        3           158.39              0.875               1       0.00226925
train_tcn_56247471   TERMINATED   intra-subject                       55             4               4       0.00440681               2             3        3           168.11              0.875               1       0.0023855

Best trial config: {'subject': 'intra-subject', 'hidden_units_per_level': 55, 'nr_levels': 4, 'kernel_size': 4, 'learning_rate': 0.004035491223241612, 'batch_size': 2, 'nr_epochs': 3}
Best trial final validation loss: 0.0003460252949594178
Best trial final validation accuracy: 1.0

"""