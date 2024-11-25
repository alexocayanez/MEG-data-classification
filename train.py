import dataclasses

import torch
from torch.utils.data import DataLoader

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

@dataclasses.dataclass(slots=True)
class RunMeter:
    correct: int = 0
    total: int = 0

    def update(self, value: bool) -> None:
        if value:
            self.correct += 1
        self.total += 1

    @property
    def accuracy(self) -> float:
        return self.correct / self.total


@dataclasses.dataclass(slots=True)
class ModelMeter:
    trainings: list[RunMeter] = dataclasses.field(default_factory=list)
    validations: list[RunMeter] = dataclasses.field(default_factory=list)

    def update(self, training: RunMeter, validation: RunMeter):
        self.trainings.append(training)
        self.validations.append(validation)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

def train_epoch(model, dataloader: DataLoader, criterion, optimizer: torch.optim, epoch_i: int, verbose=False):
    running_loss = 0.0
    meter = RunMeter()

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
        running_loss += loss.item()

        for label, prediction in zip(labels, predictions):
            print(f"{label=}\t{prediction=}") if verbose else 0
            meter.update(label == prediction.item())
        
        if i % 5 == 0:
            print(f'[{epoch_i + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}') if verbose else 0
            running_loss = 0.0

    print(f"Train Acc (epoch={epoch_i+1}): {meter.correct}/{meter.total}\t{meter.accuracy:.3f}")
    return meter


def validate_epoch(model, dataloader: DataLoader, criterion):
    validation_loss = 0.0
    meter = RunMeter()

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            out = model(inputs)
            _, predictions = torch.max(out, 1)
            
            for label, prediction in zip(labels, predictions):
                # print(f"{label=}\t{prediction=}")
                meter.update(label == prediction.item())

            loss = criterion(out, labels)
            validation_loss += loss.cpu().numpy()

    print(f"Validation: \n Test Accuracy = {meter.correct}/{meter.total}\t{meter.accuracy:.3f} \n Total loss = {validation_loss}")
    return meter

def train_and_validate(model, criterion, optimizer: torch.optim, scheduler, dataloader_train,
                       dataloader_test, epochs: int, verbose=True):
    meter = ModelMeter()
    for i in range(epochs):
        train_meter = train_epoch(model=model, dataloader=dataloader_train, criterion=criterion, optimizer = optimizer, epoch_i = i, verbose=False)
        validation_meter = validate_epoch(model=model, dataloader=dataloader_test, criterion=criterion)
        scheduler.step()
        meter.update(train_meter, validation_meter)
    return meter
