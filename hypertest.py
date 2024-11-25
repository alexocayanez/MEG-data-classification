import itertools
import json
from collections import defaultdict
from pathlib import Path
from pprint import pprint

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import ExponentialLR

from data import get_dataloaders
from models import RNN
from train import train_and_validate

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

data_folder = Path("meters").resolve()
data_folder.mkdir(parents=True, exist_ok=True)

subjects = ["intra-subject", "cross-subject"]
optimizers = [torch.optim.SGD]
hidden_sizes = (np.arange(1, 4) * 64).tolist()
num_layers = np.arange(1, 3).tolist()
gammas = [0.9]
rates = (np.arange(1, 4) * 0.005).round(3).tolist()
BATCH_SIZE = 8
INPUT_SIZE = 248
NUM_CLASSES = 4
EPOCHS = 5

hyperparameters = list(itertools.product(subjects, optimizers, hidden_sizes, num_layers, gammas, rates))


def create():
    for i, (subject, optimizer_class, hidden_size, layers, gamma, rate) in enumerate(hyperparameters, start=1):
        json_file = data_folder.joinpath(
            f"{subject}-{optimizer_class.__name__}-{hidden_size}-{layers}-{gamma}-{rate}.json")
        if json_file.exists():
            continue
        print(f"{i:02}/{len(hyperparameters)}: ", json_file.stem)
        batch_size = BATCH_SIZE
        while batch_size > 1 and not json_file.exists():
            dl_train, dl_test = get_dataloaders(type=subject, batch_size=batch_size)
            model = RNN(
                input_size=INPUT_SIZE, hidden_size=hidden_size, num_layers=layers, output_size=NUM_CLASSES
            ).to(device)

            optimizer = optimizer_class(model.parameters(), lr=rate)
            scheduler = ExponentialLR(optimizer, gamma=gamma)
            criterion = CrossEntropyLoss()
            try:
                meter = train_and_validate(
                    criterion=criterion, optimizer=optimizer, model=model, epochs=EPOCHS, verbose=False,
                    scheduler=scheduler, dataloader_train=dl_train, dataloader_test=dl_test,
                )
            except torch.cuda.OutOfMemoryError:
                batch_size -= 1
                print(f"Restarting with batch size: {batch_size}")
            else:
                json_file.write_text(json.dumps(meter.to_dict(), indent=3))
            finally:
                del model, optimizer, scheduler, criterion
                torch.cuda.empty_cache()


def analyze():
    collected_data = [defaultdict(lambda: defaultdict(int)) for _ in range(6)]
    for params in hyperparameters:
        subject, optimizer_class, hidden_size, layers, gamma, rate = params
        json_file = data_folder.joinpath(
            f"{subject}-{optimizer_class.__name__}-{hidden_size}-{layers}-{gamma}-{rate}.json")
        if json_file.exists():
            data = json.loads(json_file.read_text())
            sums = defaultdict(int)
            for i, meter in enumerate(data.values()):
                for epoch in meter:
                    for k, v in epoch.items():
                        sums[k] += v * (1 + 5 * i)
            for i, value in enumerate(params):
                p_data = collected_data[i][value]
                for k, v in sums.items():
                    p_data[k] += v
    final_data = [dict(sorted(
        ((k, v["correct"] / v["total"]) for k, v in p.items())
        , key=lambda x: x[1], reverse=True)) for p in collected_data if len(p) > 1]
    pprint(final_data)


"""
Hyper-tuning Run 1: Tested only 109/328 simulations :(
[{'intra-subject': 0.4323148148148148},
{<class 'torch.optim.adam.Adam'>: 0.2804320987654321,
<class 'torch.optim.sgd.SGD'>: 0.887962962962963},
{128: 0.6203703703703703, 256: 0.24462962962962964, 384: 0.24388888888888888},
{1: 0.47291666666666665, 2: 0.4276388888888889, 3: 0.3963888888888889},
{0.85: 0.42319444444444443,
0.9: 0.44916666666666666,
0.95: 0.4245833333333333},
{0.01: 0.4716666666666667,
0.02: 0.42083333333333334,
0.03: 0.40444444444444444}]
"""
"""
Hyper-tuning Run 2: Tested 36 simulations
[{'cross-subject': 0.5034406565656566, 'intra-subject': 0.8608333333333333},
{64: 0.5192129629629629, 128: 0.5753858024691358, 192: 0.6142746913580247},
{1: 0.5847222222222223, 2: 0.5545267489711934},
{0.005: 0.5295524691358025,
0.01: 0.5783179012345679,
0.015: 0.6010030864197531}]
"""
"""
Final hyperparameters:
hidden_sizes: 192,
num_layers: 1,
gamma: 0.9,
learning_rate: 0.015
"""

if __name__ == '__main__':
    create()
    analyze()
