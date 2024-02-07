import os
import time
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
        DEVICE="cuda",
    ):
        self.DEVICE = DEVICE
        self.network = neural_network
        self.model = neural_network(weights=weights).to(DEVICE)
        # for param in self.model.parameters():
        #     param.requires_grad = False
        self.num_classes = num_classes

        self.trainable_layer_name = ""

        if hasattr(self.model, "fc"):
            if self.model.fc.out_features != self.num_classes:
                self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)
            self.trainable_layer_name = "fc"
            # self.model.fc.weight = nn.init.normal_(
            #     self.model.fc.weight, mean=0.0, std=0.01
            # )
            # self.model.fc.bias = nn.init.zeros_(self.model.fc.bias)
        elif hasattr(self.model, "classifier"):
            self.trainable_layer_name = "classifier"
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
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if self.trainable_layer_name in name:
                    print(name)
                    self.params_sizes[name] = param.size()
                else:
                    param.requires_grad = False

    def get_parameters(self, model):
        params = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                # print(param.size())
                params.append(torch.flatten(param))
        params = torch.cat(params).cpu().detach().numpy()
        return params

    def set_weights(self, model_state, all_parameters):
        counted_params = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                model_state[name] = torch.tensor(
                    all_parameters[
                        counted_params : param.size().numel() + counted_params
                    ]
                ).reshape(param.size())
                counted_params += param.size().numel()
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
                # print(data.shape)
                data, label = data.to(self.DEVICE), label.to(self.DEVICE)
                output = model(data)
                # loss = criterion(output, label)
                # running_loss += loss.item()
                out = nn.functional.softmax(output, dim=1)
                _, pred = torch.max(out, dim=1)

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
        start_time = time.time()
        self.model.load_state_dict(
            self.set_weights(self.model.state_dict(), parameters)
        )
        end_time = time.time()
        # print(end_time-start_time)
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
                out = nn.functional.softmax(output, dim=1)
                _, pred = torch.max(out, dim=1)

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

    def random_population_init(self, popsize, seed=42):
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

    def optimized_population_init(
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

    def optimized_local_search_boundaries(
        self, blocked_dimensions, blocks_mask, seed=42
    ):
        rng = np.random.default_rng(seed)
        params = self.get_parameters(self.model)

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

    def pre_train(self, epochs=10, train_loader=None, model_save_path=None):
        model = self.model
        # print(model_save_path)
        import torch
        import torch.optim as optim
        import torch.nn as nn

        if os.path.exists(model_save_path + ".pth"):
            self.model.load_state_dict(torch.load(model_save_path + ".pth"))
            self.model.to(self.DEVICE)
            print("Saved model is loaded from:", model_save_path + ".pth")
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
                out = nn.functional.softmax(output, dim=1)
                _, pred = torch.max(out, dim=1)
                loss = criterion(output, label)
                # log_probs = nn.functional.log_softmax(output, dim=1)
                # loss = criterion(log_probs, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                _, pred = torch.max(output.data, 1)

                true_labels.extend(label.tolist())
                predicted_labels.extend(pred.tolist())

            train_loss = running_loss / len(train_loader)
            train_loss_history.append(train_loss)
            train_f1score = f1_score(true_labels, predicted_labels, average="macro")
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

        plt.plot(train_f1_history, label="train")
        plt.plot(val_f1_history, label="val")
        plt.ylabel("f1-score")
        plt.xlabel("epoch")
        plt.savefig(model_save_path + ".png")
        plt.show()
        plt.close()

        np.savez(
            model_save_path + ".npz",
            train_loss_history=train_loss_history,
            train_f1_history=train_f1_history,
            val_f1_history=val_f1_history,
        )

        gfo.model = model
        torch.save(model.state_dict(), model_save_path + ".pth")
        # params = gfo.get_parameters(model)
        print("Model is saved to:", model_save_path + ".pth")
        return model
