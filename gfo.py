import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score


class GradientFreeOptimization:
    def __init__(
        self,
        neural_network=None,
        weights=None,
        num_classes=10,
        data_loader=None,
        data_shape=None,
        val_loader=None,
        metric="f1",
        DEVICE="cpu",
    ):
        self.DEVICE = DEVICE
        self.network = neural_network
        self.model = neural_network(weights=weights).to(DEVICE)
        # for param in self.model.parameters():
        #     param.requires_grad = False
        self.num_classes = num_classes

        if hasattr(self.model, "fc"):
            if self.model.fc.out_features != self.num_classes:
                self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)
                # self.model.fc.weight = nn.init.normal_(
                #     self.model.fc.weight, mean=0.0, std=0.01
                # )
                # self.model.fc.bias = nn.init.zeros_(self.model.fc.bias)
        elif hasattr(self.model, "classifier"):
            if self.model.classifier[-1].out_features != self.num_classes:
                self.model.classifier[-1] = nn.Linear(
                    self.model.classifier[-1].in_features, self.num_classes
                )

        self.model.to(DEVICE)
        self.weights = weights
        self.data_loader = data_loader
        self.val_loader = val_loader
        self.metric = metric.lower()
        self.params_sizes = {}

    def find_param_sizes(self):
        for p in self.model.state_dict():
            # if ('weight' in p or 'bias' in p) and (not 'num_batches_tracked' in p):
            self.params_sizes[p] = self.model.state_dict()[p].size()

    def get_parameters(self, model):
        model_state = model.state_dict()
        params = []
        for p in model_state:
            # if ('weight' in p or 'bias' in p or 'running' in p) and (not 'num_batches_tracked' in p):
            # if (not 'num_batches_tracked' in p):
            if "weight" in p or "bias" in p:
                params.append(model_state[p].view(-1))
        params = torch.cat(params).cpu().detach().numpy()
        return params

    def set_weights(self, model_state, all_parameters):
        counted_params = 0
        for p in model_state:
            # if ('weight' in p or 'bias' in p or 'running' in p) and (not 'num_batches_tracked' in p):
            # if (not 'num_batches_tracked' in p):
            if "weight" in p or "bias" in p:
                model_state[p] = torch.tensor(
                    all_parameters[
                        counted_params : self.params_sizes[p].numel() + counted_params
                    ]
                ).reshape(self.params_sizes[p])
                counted_params += self.params_sizes[p].numel()
        return model_state

    def score_in_optimization(self, model=None):
        if model == None:
            model = self.model

        true_labels = []
        predicted_labels = []
        correct_1 = 0.0
        correct_5 = 0.0

        with torch.no_grad():
            for batch_idx, (data, label) in enumerate(self.data_loader):
                data, label = data.to(self.DEVICE), label.to(self.DEVICE)
                output = model(data)
                # loss = criterion(output, label)
                # running_loss += loss.item()
                _, pred = torch.max(output.data, 1)

                true_labels.extend(label.tolist())
                predicted_labels.extend(pred.tolist())

                _, top5pred = output.topk(5, 1, largest=True, sorted=True)

                label = label.view(label.size(0), -1).expand_as(top5pred)
                correct = top5pred.eq(label).float()

                # compute top 5
                correct_5 += correct[:, :5].sum()

                # compute top1
                correct_1 += correct[:, :1].sum()

        score = 0
        if self.metric == "f1":
            score = f1_score(true_labels, predicted_labels, average="macro")
        elif self.metric == "top1":
            score = correct_1 / len(self.data_loader.dataset)
        elif self.metric == "top5":
            score = correct_5 / len(self.data_loader.dataset)
        return score

    def fitness_func(self, parameters):
        if len(parameters) != len(self.get_parameters(self.model)):
            error_msg = f"Not matched sizes of parameters, given parameters length: {len(parameters)}, model parameters length: {len(self.get_parameters(self.model))}"
            raise Exception(error_msg)
        self.model.load_state_dict(
            self.set_weights(self.model.state_dict(), parameters)
        )
        self.model.to(self.DEVICE)
        self.model.eval()
        fitness = 1 - self.score_in_optimization()
        return fitness

    def validation_func(self, parameters):
        if len(parameters) != len(self.get_parameters(self.model)):
            error_msg = f"Not matched sizes of parameters, given parameters length: {len(parameters)}, model parameters length: {len(self.get_parameters(self.model))}"
            raise Exception(error_msg)
        val = self.evaluate_params(parameters, self.val_loader)
        return val

    def evaluate_params(self, parameters, data_loader, model=None, metric=None):
        if model == None:
            model = self.model
        else:
            self.model = model
        if len(parameters) != len(self.get_parameters(model)):
            error_msg = f"Not matched sizes of parameters, given parameters length: {len(parameters)}, model parameters length: {len(self.get_parameters(self.model))}"
            raise Exception(error_msg)
        # prev_loader = self.data_loader
        # self.data_loader = data_loader

        model.load_state_dict(self.set_weights(model.state_dict(), parameters))
        model.to(self.DEVICE)
        model.eval()

        if metric == None:
            metric = self.metric

        true_labels = []
        predicted_labels = []
        correct_1 = 0.0
        correct_5 = 0.0

        with torch.no_grad():
            for batch_idx, (data, label) in enumerate(data_loader):
                data, label = data.to(self.DEVICE), label.to(self.DEVICE)
                output = model(data)
                # loss = criterion(output, label)
                # running_loss += loss.item()
                _, pred = torch.max(output.data, 1)

                true_labels.extend(label.tolist())
                predicted_labels.extend(pred.tolist())

                _, top5pred = output.topk(5, 1, largest=True, sorted=True)

                label = label.view(label.size(0), -1).expand_as(top5pred)
                correct = top5pred.eq(label).float()

                # compute top 5
                correct_5 += correct[:, :5].sum()

                # compute top1
                correct_1 += correct[:, :1].sum()

        score = 0
        if metric == "f1":
            score = f1_score(true_labels, predicted_labels, average="macro")
        elif metric == "top1":
            score = correct_1.cpu().numpy() / len(data_loader.dataset)
        elif metric == "top5":
            score = correct_5.cpu().numpy() / len(data_loader.dataset)
        return score

    def population_initializer(self, popsize, seed=42):
        torch.manual_seed(seed)
        initial_population = []
        for i in range(popsize):
            model = self.network(weights=self.weights)
            if hasattr(model, "fc"):
                if model.fc.out_features != self.num_classes:
                    model.fc = nn.Linear(model.fc.in_features, self.num_classes)
                    # model.fc.weight = nn.init.normal_(
                    #     model.fc.weight, mean=0.0, std=0.01
                    # )
                    # model.fc.bias = nn.init.zeros_(model.fc.bias)
            elif hasattr(model, "classifier"):
                if model.classifier[-1].out_features != self.num_classes:
                    model.classifier[-1] = nn.Linear(
                        model.classifier[-1].in_features, self.num_classes
                    )
            model.to(self.DEVICE)
            params = self.get_parameters(model)
            initial_population.append([params])
        return np.concatenate(initial_population, axis=0)

    def population_initializer_by_blocked_intervals(
        self, popsize, blocked_dimensions, blocks_mask, seed=42
    ):
        rng = np.random.default_rng(seed)
        params = self.get_parameters(self.model)

        initial_population = []
        for i in range(popsize):
            params_blocked = np.zeros((blocked_dimensions))
            for i in range(blocked_dimensions):
                block_params = params[blocks_mask[i]]
                if len(block_params) != 0:
                    params_blocked[i] = rng.uniform(
                        low=block_params.min(), high=block_params.max()
                    )
            initial_population.append([params_blocked])

        return np.concatenate(initial_population, axis=0)
