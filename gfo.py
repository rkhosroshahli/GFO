import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

# from model import ArtificialNeuralNetwork # , accuracy_score

class GradientFreeOptimization():
    def __init__(self, neural_network=None, weights=None, num_classes=10, data_loader=None, DEVICE="cpu"):
        self.DEVICE = DEVICE
        self.network = neural_network
        self.model = neural_network(weights=weights).to(DEVICE)
        num_ftrs = self.model.fc.in_features
        if self.model.fc.out_features != num_classes:
            self.model.fc = nn.Sequential(
                nn.Linear(num_ftrs, 512),  # Additional linear layer
                nn.ReLU(),
                nn.Linear(512, num_classes)
            )
        self.model.to(DEVICE)
        self.weights = weights
        self.num_classes = num_classes
        self.loader = data_loader
        self.params_sizes = {}
        


    def find_param_sizes(self):
        for p in self.model.state_dict():
            # if ('weight' in p or 'bias' in p) and (not 'num_batches_tracked' in p):
            self.params_sizes[p] = self.model.state_dict()[p].size()

    
    def set_weights(self, model_state, all_parameters):
        counted_params = 0
        for p in model_state:
            # if ('weight' in p or 'bias' in p or 'running' in p) and (not 'num_batches_tracked' in p):
            # if (not 'num_batches_tracked' in p):
            if 'weight' in p or 'bias' in p:
                model_state[p] = torch.tensor(all_parameters[counted_params:self.params_sizes[p].numel()+counted_params]).reshape(self.params_sizes[p])
                counted_params += self.params_sizes[p].numel()
        return model_state

    def f1score_in_optimization(self, model=None):
        if model == None:
            model = self.model
        # running_loss = 0.0
        true_labels = []
        predicted_labels = []

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.loader):
                data, target = data.to(self.DEVICE), target.to(self.DEVICE)
                output = model(data)
                # loss = criterion(output, target)
                # running_loss += loss.item()
                _, predicted = torch.max(output.data, 1)

                true_labels.extend(target.tolist())
                predicted_labels.extend(predicted.tolist())

        train_f1score = f1_score(true_labels, predicted_labels, average='macro')
        return train_f1score

    def fitness_func(self, parameters):
        if len(parameters) != len(self.get_parameters(self.model)):
            error_msg = f"Not matched sizes of parameters, given parameters length: {len(parameters)}, model parameters length: {len(self.get_parameters(self.model))}"
            raise Exception(error_msg)
        self.model.load_state_dict(self.set_weights(self.model.state_dict(), parameters))
        self.model.to(self.DEVICE)
        self.model.eval()
        fitness = -1 * self.f1score_in_optimization()
        return fitness
    
    def evaluate_params(self, parameters, data_loader):
        if len(parameters) != len(self.get_parameters(self.model)):
            error_msg = f"Not matched sizes of parameters, given parameters length: {len(parameters)}, model parameters length: {len(self.get_parameters(self.model))}"
            raise Exception(error_msg)
        # prev_loader = self.loader
        # self.loader = data_loader
        self.model.load_state_dict(self.set_weights(self.model.state_dict(), parameters))
        self.model.to(self.DEVICE)
        self.model.eval()
        
        true_labels = []
        predicted_labels = []

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_loader):
                data, target = data.to(self.DEVICE), target.to(self.DEVICE)
                output = self.model(data)
                # loss = criterion(output, target)
                # running_loss += loss.item()
                _, predicted = torch.max(output.data, 1)

                true_labels.extend(target.tolist())
                predicted_labels.extend(predicted.tolist())

        f1score = f1_score(true_labels, predicted_labels, average='macro')
        # self.loader = prev_loader
        return f1score

    def get_parameters(self, model):
        model_state = model.state_dict()
        params=[]
        for p in model_state:
            # if ('weight' in p or 'bias' in p or 'running' in p) and (not 'num_batches_tracked' in p):
            # if (not 'num_batches_tracked' in p):
            if 'weight' in p or 'bias' in p:
                params.append(model_state[p].view(-1))
        params = torch.cat(params).cpu().detach().numpy()
        return params

    def population_initializer(self, popsize, seed=42):
        torch.manual_seed(seed)
        initial_population = []
        for i in range(popsize):
            model = self.network(weights=self.weights, num_classes=self.num_classes).to(self.DEVICE)
            params = self.get_parameters(model)
            initial_population.append([params])
        return np.concatenate(initial_population, axis=0)
    
    def population_initializer_by_blocked_intervals(self, popsize, blocked_dimensions, blocks_mask, seed=42):
        rng = np.random.default_rng(seed)
        params = self.get_parameters(self.model)

        initial_population = []
        for i in range(popsize):
            params_blocked = np.zeros((blocked_dimensions))
            for i in range(blocked_dimensions):
                block_params = params[blocks_mask[i]]
                if len(block_params) != 0:
                    params_blocked[i] = rng.uniform(low=block_params.min(), high=block_params.max())
            initial_population.append([params_blocked])
        
        return np.concatenate(initial_population, axis=0)