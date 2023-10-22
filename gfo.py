import numpy as np
import torch
from sklearn.metrics import f1_score # , accuracy_score

class GradientFreeOptimization():
    def __init__(self, NeuralNetwork, loader, DEVICE):
        self.network = NeuralNetwork
        self.model = NeuralNetwork()
        self.loader = loader
        self.params_sizes={}
        for p in self.model.state_dict():
            self.params_sizes[p] = self.model.state_dict()[p].size()
        self.DEVICE = DEVICE

    
    def set_weights(self, model_state, all_parameters):
        counted_params = 0
        for p in model_state:
            if ('weight' in p or 'bias' in p) and (not 'num_batches_tracked' in p):
                model_state[p] = torch.tensor(all_parameters[counted_params:self.params_sizes[p].numel()+counted_params]).reshape(self.params_sizes[p])
                counted_params += self.params_sizes[p].numel()
        return model_state

    def f1score_in_optimization(self):
        # running_loss = 0.0
        true_labels = []
        predicted_labels = []

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.loader):
                data, target = data.to(self.DEVICE), target.to(self.DEVICE)
                output = self.model(data)
                # loss = criterion(output, target)

                # running_loss += loss.item()

                _, predicted = torch.max(output.data, 1)

                true_labels.extend(target.tolist())
                predicted_labels.extend(predicted.tolist())

        train_f1score = f1_score(true_labels, predicted_labels, average='macro')
        return train_f1score

    def fitness_func(self, parameters):
        self.model.load_state_dict(self.set_weights(self.model.state_dict(), parameters))
        self.model.to(self.DEVICE)
        self.model.eval();
        fitness = -1 * self.f1score_in_optimization()
        return fitness

    def get_parameters(self, model):
        model_state = model.state_dict()
        params=[]
        for p in model_state:
            if ('weight' in p or 'bias' in p) and (not 'num_batches_tracked' in p):
                params.append(model_state[p].view(-1))
        params = torch.cat(params).cpu().detach().numpy()
        return params

    def population_initializer(self, popsize, seed=42):
        torch.manual_seed(seed)
        # model = self.network().to(DEVICE)
        params = self.get_parameters(self.model)
        initial_population = np.array([params])
        for i in range(popsize-1):
            model = self.network().to(self.DEVICE)
            #print(f1score_in_optimization(model, train_loader))
            params = self.get_parameters(model)
            #print(fitness_func(params))
            initial_population = np.concatenate([initial_population, [params]], axis=0)
        return initial_population