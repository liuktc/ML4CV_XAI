import os
import json


class ResultMetrics:
    """
    The structure of the results is as follows:
    {
        "model_name": {
            "attribution_method": {
                "layer_name": {
                    "metric_name": value
                }
            }
        }
    }
    """

    def __init__(self, path: str):
        self.path = path
        self.results = {}
        self.load_results()

    def load_results(self):
        if os.path.exists(self.path):
            with open(self.path, "r") as f:
                self.results = json.load(f)
        else:
            print(f"Results file not found. Creating new results file {self.path}.")
            self.results = {}

        self.save_results()

    def add_results_all_layers(
        self, model: str, attribution_method: str, results: dict[str, dict[str, float]]
    ):
        """Add a result to the results dictionary.

        Args:
            model (str): The name of the model.
            attribution_method (str): The attribution method used to generate the results.
            results (dict[str, dict[str, float]]): Dictionary containing the results of each metric for each layer.
        """
        if model not in self.results:
            self.results[model] = {attribution_method: results}
        else:
            if attribution_method not in self.results[model]:
                self.results[model][attribution_method] = results
            for layer, metrics in results.items():
                if layer not in self.results[model][attribution_method]:
                    self.results[model][attribution_method][layer] = metrics

        self.save_results()

    def add_results_single_layer(
        self, model: str, attribution_method: str, layer: str, results: dict[str, float]
    ):
        """Add a result to the results dictionary.

        Args:
            model (str): The name of the model.
            attribution_method (str): The attribution method used to generate the results.
            layer (str): The name of the layer.
            results (dict[str, float]): Dictionary containing the results of each metric for the layer.
        """
        if model not in self.results:
            self.results[model] = {attribution_method: {layer: results}}
        else:
            if attribution_method not in self.results[model]:
                self.results[model][attribution_method] = {layer: results}
            elif layer not in self.results[model][attribution_method]:
                self.results[model][attribution_method][layer] = results

        self.save_results()

    def save_results(self):
        with open(self.path, "w") as f:
            json.dump(self.results, f, indent=4)
