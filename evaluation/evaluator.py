import torch
from sklearn.metrics import confusion_matrix
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import scienceplots
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np

class Evaluator():
    def __init__(self, testloader, model):
        self.testloader = testloader
        self.model = model

    def calculate_confusion(self):
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in self.testloader:
                _, res = self.model(inputs)
                outputs = F.softmax(res, dim=1)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        return np.array(all_preds), np.array(all_labels)
    
    def get_confusion(self, true_labels, predictions):
        return confusion_matrix(true_labels, predictions)
    
    def get_confusion_normalized(self, true_labels, predictions):
        return confusion_matrix(true_labels, predictions, normalize="true")
    

class Plot:
    def __init__(self) -> None:
        pass

    def sciencePlot(self, save_path: str = None, legend: bool = True, plot_title: str = "Titel", x_label: str = "X", y_label: str = "Y", graphs: list = None, inset_xlim: list = None, inset_ylim:list = None):
        """
        The list of graphs contains each information of its graphs.

        for each item in the list:
            graph[0] --> label: str
            graph[1] --> x-coordinates: np-Array
            graph[2] --> y-coordinates: np-Array
        """

        ####################
        # set style
        plt.style.use(["science", "ieee"])
        ####################

        fig, ax = plt.subplots()

        for graph in graphs: 
            plt.plot(graph[1], graph[2], label=graph[0])
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        if legend == True:
            plt.legend()
        plt.grid(False)
        plt.title(plot_title)

        # Set x ticks
        x_ticks = np.arange(np.floor(np.array(graphs[0][1]).min()), np.ceil(np.array(graphs[0][1]).max()) + 1, 2)
        plt.xticks(x_ticks)

        # Add inset plot
        if inset_xlim is not None and inset_ylim is not None:
            axins = inset_axes(ax, width='65%', height='30%', loc='right')
            axins.set_xlim(inset_xlim[0], inset_xlim[1])
            axins.set_ylim(inset_ylim[0], inset_ylim[1])
            axins.tick_params(axis='both', which='both', labelsize=4)

            for graph in graphs:
                axins.plot(graph[1], graph[2], label=graph[0])

            # axins.set_xlabel('X-Achse')
            # axins.set_ylabel('Y-Achse')
            axins.set_title('vergrößerte Teilansicht', fontsize=4)

        if save_path is not None:
            plt.savefig(save_path, format='png')

        plt.show()