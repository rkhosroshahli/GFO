import copy
import os
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    top_k_accuracy_score,
)

from pymoo.core.problem import Problem
from collections import Counter

# from block import Block
from data_loader import GFO_data


class GradientFreeOptimization:
    def __init__(
            self,
            network=None,
            weights=None,
            data: GFO_data = None,
            metric="f1",
            DEVICE="gpu",
            model_save_path=None,
            dataset=None,
            model_name=None,
    ):
        self.DEVICE = DEVICE
        self.network = network
        self.model = network
        if weights != "hugging_face":
            self.model = network(weights=weights)
        # for param in self.model.parameters():
        #     param.requires_grad = False
        self.num_classes = data.num_classes

        if hasattr(self.model, "fc"):
            print("fc")
            if self.model.fc.out_features != self.num_classes:
                self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)
                # self.model.fc.weight = nn.init.normal_(
                #     self.model.fc.weight, mean=0.0, std=0.01
                # )
                # self.model.fc.bias = nn.init.zeros_(self.model.fc.bias)
            else:
                print("The model loaded has a classifier output with the same size of classes")
        elif hasattr(self.model, "classifier"):
            print("Classifier")
            if self.model.classifier[-1].out_features != self.num_classes:
                self.model.classifier[-1] = nn.Linear(
                    self.model.classifier[-1].in_features, self.num_classes
                )
            else:
                print("The model loaded has a classifier output with the same size of classes")

        self.model.to(DEVICE)
        self.weights = weights
        self.data = data
        self.data_loader = data.generate_balanced_dataloader(seed=42)
        self.train_loader = data.train_loader
        self.val_loader = data.test_loader
        self.metric = metric.lower()
        self.params_sizes = {}
        self.model_save_path = model_save_path
        self.dataset = dataset
        self.model_name = model_name

    def find_param_sizes(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.params_sizes[name] = param.size()

    def get_parameters(self, model):
        if model is None:
            self.model = model
        params = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                params.append(torch.flatten(param).cpu().detach().numpy())

        return np.concatenate(params)

    def set_model_state(self, state, parameters):
        counted_params = 0
        torch_parameters = torch.from_numpy(parameters).to(self.DEVICE)
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                state[name] = torch.tensor(
                    parameters[counted_params: param.size().numel() + counted_params]
                ).reshape(param.size())
                counted_params += param.size().numel()
        return state

    def score_in_optimization(self, model=None):
        if model == None:
            model = self.model

        true_labels = []
        predicted_labels = []

        with torch.no_grad():
            for batch_idx, (data, label) in enumerate(self.data_loader):
                data, label = data.to(self.DEVICE), label.to(self.DEVICE)
                output = model(data)

                _, pred = torch.max(output, dim=1)

                true_labels.extend(label.cpu().detach().numpy())
                predicted_labels.extend(pred.cpu().detach().numpy())

        score = 0
        if self.metric == "f1":
            score = f1_score(true_labels, predicted_labels, average="weighted")
        elif self.metric == "top1":
            score = top_k_accuracy_score(true_labels, predicted_labels, k=1)
        elif self.metric == "top5":
            score = top_k_accuracy_score(true_labels, predicted_labels, k=5)
        return score

    def fitness_func(self, parameters):
        if len(parameters) != len(self.get_parameters(self.model)):
            error_msg = f"Not matched sizes of parameters, given parameters length: {len(parameters)}, model parameters length: {len(self.get_parameters(self.model))}"
            raise Exception(error_msg)

        self.model.load_state_dict(
            self.set_model_state(self.model.state_dict(), parameters)
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

    def train_func(self, parameters):
        if len(parameters) != len(self.get_parameters(self.model)):
            error_msg = f"Not matched sizes of parameters, given parameters length: {len(parameters)}, model parameters length: {len(self.get_parameters(self.model))}"
            raise Exception(error_msg)
        val = self.evaluate_params(parameters, self.train_loader)
        return val

    def evaluate_params(self, parameters, data_loader, model=None, metric=None):
        if model == None:
            model = self.model
        else:
            self.model = model
        if len(parameters) != len(self.get_parameters(model)):
            error_msg = f"Not matched sizes of parameters, given parameters length: {len(parameters)}, model parameters length: {len(self.get_parameters(self.model))}"
            raise Exception(error_msg)

        model.load_state_dict(self.set_model_state(model.state_dict(), parameters))
        model.eval()

        if metric == None:
            metric = [self.metric]
        elif isinstance(metric, str):
            metric = [metric]

        true_labels = []
        predicted_labels = []

        with torch.no_grad():
            for batch_idx, (data, label) in enumerate(data_loader):
                # print(batch_idx, data.size())
                data, label = data.to(self.DEVICE), label.to(self.DEVICE)
                output = model(data)
                # loss = criterion(output, label)
                # running_loss += loss.item()
                # out = nn.functional.softmax(output, dim=1)
                _, pred = torch.max(output, dim=1)

                true_labels.extend(label.cpu().detach().numpy())
                predicted_labels.extend(pred.cpu().detach().numpy())

        score = []
        for m in metric:
            s = 0
            if m == "f1":
                s = f1_score(true_labels, predicted_labels, average="macro")
            elif m == "precision":
                s = precision_score(true_labels, predicted_labels, average="macro")
            elif m == "recall":
                s = recall_score(true_labels, predicted_labels, average="macro")
            elif m == "top1":
                s = top_k_accuracy_score(true_labels, predicted_labels, k=1)
            elif m == "top5":
                s = top_k_accuracy_score(true_labels, predicted_labels, k=5)
            score.append(s)

        if len(score) == 1:
            return score[0]
        return score

    def random_population_init(self, dims:int, pop_size: int = 100, seed: int = 42):
        # torch.manual_seed(seed)
        # params = self.get_parameters(self.model)
        init_pop = np.random.normal(loc=0, scale=1, size=(pop_size, dims))
        return init_pop

    def block_population_init(self, pop_size, block, seed: int = 42):
        rng = np.random.default_rng(seed)
        params = self.get_parameters(self.load_model_from_path(self.model_save_path))
        blocks_mask = block.load_mask()
        blocked_dimensions = len(blocks_mask)

        initial_population = np.zeros((pop_size, blocked_dimensions))
        for i in range(pop_size):
            params_blocked = np.zeros((blocked_dimensions))
            for j in range(blocked_dimensions):
                block_params = params[blocks_mask[j]]
                # print(block_params.min(), block_params.max())
                if len(block_params) != 0:
                    params_blocked[j] = rng.uniform(
                        low=block_params.min(), high=block_params.max()
                    )
            initial_population[i, :] = params_blocked[:].copy()
        return initial_population  # np.concatenate(initial_population, axis=0)

    def block_local_search_boundaries(self, blocked_dimensions, block, seed=42):
        rng = np.random.default_rng(seed)
        params = self.get_parameters(self.load_model_from_path(self.model_save_path))
        blocks_mask = block.load_mask()

        var_min = np.zeros((blocked_dimensions))
        var_max = np.zeros((blocked_dimensions))

        for i in range(1, blocked_dimensions - 1):
            # block_params = params[blocks_mask[i]]
            if len(blocks_mask[i]) != 0:
                var_min[i] = params[blocks_mask[i - 1]].min()
                var_max[i] = params[blocks_mask[i + 1]].max()

        # ENABLE WHEN MERGE is not DONE!!!
        # var_min[0] = var_max[0] = params[blocks_mask[0]].copy()
        # var_min[-1] = var_max[-1] = params[blocks_mask[-1]].copy()

        return var_min, var_max

    def load_model_from_path(self, model_save_path, model=None):
        if model == None:
            model = self.model
        model.load_state_dict(torch.load(model_save_path + ".pth"))
        model.to(self.DEVICE)
        print("Saved model is loaded from:", model_save_path + ".pth")
        return model

    def pre_train(self, epochs=10, train_loader=None, model_save_path=None):
        model = self.model
        # print(model_save_path)
        import torch
        import torch.optim as optim
        import torch.nn as nn

        if os.path.exists(model_save_path + ".pth"):
            self.model = self.load_model_from_path(model_save_path)
            return self.model

        criterion = nn.CrossEntropyLoss()
        # criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        train_f1_history = []
        train_loss_history = []
        val_f1_history = []
        # Step 5: Train the network
        num_epochs = epochs
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            true_labels = []
            predicted_labels = []

            for batch_idx, (data, label) in enumerate(train_loader):
                data, label = data.to(self.DEVICE), label.to(self.DEVICE)
                output = model(data)
                # out = nn.functional.softmax(output, dim=1)
                _, pred = torch.max(output, dim=1)
                loss = criterion(output, label)
                # log_probs = nn.functional.log_softmax(output, dim=1)
                # loss = criterion(log_probs, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                true_labels.extend(label.cpu().detach().numpy())
                predicted_labels.extend(pred.cpu().detach().numpy())

            train_loss = running_loss / len(train_loader)
            train_loss_history.append(train_loss)
            train_f1score = f1_score(true_labels, predicted_labels, average="weighted")
            val_f1score = self.validation_func(self.get_parameters(model))
            train_f1_history.append(train_f1score)
            val_f1_history.append(val_f1score)

            print(
                f"Epoch: {epoch}| Train loss: {train_loss: .5f}| Train acc: {train_f1score: .5f}| Val acc: {val_f1score: .5f}"
            )
            # print(
            #     f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss:.4f}, F1-score: {train_f1score*100:.2f}%, Validation"
            # )
        import matplotlib.pyplot as plt

        points = np.linspace(1, num_epochs, num=num_epochs)
        plt.plot(points, train_f1_history, "o--", label="train")
        # plt.scatter(x=points, y=train_f1_history)
        plt.plot(points, val_f1_history, "^--", label="val")
        # plt.scatter(x=points, y=train_f1_history)
        plt.xlabel("epoch")
        plt.ylabel("f1 score")
        plt.legend()
        plt.grid()
        plt.savefig(model_save_path + ".png")
        # plt.show()
        plt.close()

        np.savez(
            model_save_path + ".npz",
            train_loss_history=train_loss_history,
            train_f1_history=train_f1_history,
            val_f1_history=val_f1_history,
        )

        self.model = model
        torch.save(model.state_dict(), model_save_path + ".pth")
        # params = gfo.get_parameters(model)
        print("Model is saved to:", model_save_path + ".pth")
        return model


class PercisionRecallProblem(Problem):
    def __init__(
            self,
            n_var=None,
            sample_size=None,
            gfo: GradientFreeOptimization = None,
            block= None,
            data: GFO_data = None,
    ):
        super().__init__(
            n_var=n_var,
            n_obj=2,
            n_ieq_constr=0,
            xl=-5.0,
            xu=5.0,
            vtype=float,
        )
        self.sample_size = sample_size
        self.dataset = data.dataset
        self.num_classes = data.num_classes
        self.gfo = gfo
        self.block = block
        self.data_loader = data.generate_balanced_dataloader(sample_size)
        self.data_loader_generator = data.generate_balanced_dataloader
        self.balanced_indices = []
        # self.next_batch()

    def next_batch(self):
        self.data_loader = self.data_loader_generator(self.sample_size)

    def _calc_pareto_front(self, n_pareto_points=100):
        x = np.linspace(0, 1, n_pareto_points)
        # return np.array([x, 1 - np.power(x, 2)]).T
        return np.ones((n_pareto_points, 2))

    def _evaluate(self, X, out, *args, **kwargs):

        # from sklearn.metrics import (
        #     f1_score,
        #     precision_score,
        #     recall_score,
        #     roc_auc_score,
        #     top_k_accuracy_score,
        # )

        NP = X.shape[0]

        f1 = np.zeros(NP)
        f2 = np.zeros(NP)
        f3 = np.zeros(NP)
        fn = np.zeros(NP)

        model = None
        if model == None:
            model = self.gfo.model

        # start_time = time.time()

        for i in range(NP):
            xi = X[i]
            uxi = xi.copy()
            if len(xi) != self.block.dims:
                uxi, = self.block.unblocker(np.array([xi]))
            p, r = self.gfo.evaluate_params(
                uxi, self.data_loader, metric=["precision", "recall"]
            )
            del uxi
            f1[i] = p
            f2[i] = r

        out["F"] = np.column_stack([1 - f1, 1 - f2])

        if "reset" in kwargs.keys():
            self.next_batch()
