import os
import pandas as pd


# class SingleResult:
#     def __init__(
#         self,
#         model_name,
#         attribution_method,
#         layer_name,
#         metric_name,
#         upscale_method,
#         value,
#     ):
#         self.model_name = model_name
#         self.attribution_method = attribution_method
#         self.layer_name = layer_name
#         self.metric_name = metric_name
#         self.upscale_method = upscale_method
#         self.value = value

#     def __str__(self):
#         return f"Model: {self.model_name}, Method: {self.attribution_method}, Layer: {self.layer_name}, Metric: {self.metric_name}, Upscale Method: {self.upscale_method}, Value: {self.value}"

#     def __repr__(self):
#         return self.__str__()

#     def to_dict(self):
#         return {
#             "model_name": self.model_name,
#             "attribution_method": self.attribution_method,
#             "layer_name": self.layer_name,
#             "metric_name": self.metric_name,
#             "upscale_method": self.upscale_method,
#             "value": self.value,
#         }

#     def to_csv(self):
#         return f"{self.model_name},{self.attribution_method},{self.layer_name},{self.metric_name},{self.upscale_method},{self.value}"


class ResultMetrics:
    """
    The results are saved in a csv file with the following header:
    Model,Attribution Method,Layer,Metric,Upscale Method,Value
    """

    def __init__(self, path: str):
        self.path = path
        self.HEADER = [
            "Model",
            "Attribution Method",
            "Layer",
            "Metric",
            "Upscale Method",
            "Value",
        ]

        self.results = pd.DataFrame(columns=self.HEADER)
        self.load_results()

    def load_results(self):
        if os.path.exists(self.path):
            self.results = pd.read_csv(self.path)
            print(f"Results loaded from {self.path}.")
        else:
            print(f"Results file not found. Creating new results file {self.path}.")
            self.results = pd.DataFrame(columns=self.HEADER)

        self.save_results()

    def add_result(
        self, model, attribution_method, layer, metric, upscale_method, value
    ):
        # Add result to the results dataframe
        self.results = pd.concat(
            [
                self.results,
                pd.DataFrame(
                    [[model, attribution_method, layer, metric, upscale_method, value]],
                    columns=self.HEADER,
                ),
            ]
        )

        self.save_results()

    # def add_results_all_layers(
    #     self, model: str, attribution_method: str, results: dict[str, dict[str, float]]
    # ):
    #     """Add a result to the results dictionary.

    #     Args:
    #         model (str): The name of the model.
    #         attribution_method (str): The attribution method used to generate the results.
    #         results (dict[str, dict[str, float]]): Dictionary containing the results of each metric for each layer.
    #     """
    #     if model not in self.results:
    #         self.results[model] = {attribution_method: results}
    #     else:
    #         if attribution_method not in self.results[model]:
    #             self.results[model][attribution_method] = results
    #         for layer, metrics in results.items():
    #             if layer not in self.results[model][attribution_method]:
    #                 self.results[model][attribution_method][layer] = metrics

    #     self.save_results()

    # def add_results_single_layer(
    #     self, model: str, attribution_method: str, layer: str, results: dict[str, float]
    # ):
    #     """Add a result to the results dictionary.

    #     Args:
    #         model (str): The name of the model.
    #         attribution_method (str): The attribution method used to generate the results.
    #         layer (str): The name of the layer.
    #         results (dict[str, float]): Dictionary containing the results of each metric for the layer.
    #     """
    #     if model not in self.results:
    #         self.results[model] = {attribution_method: {layer: results}}
    #     else:
    #         if attribution_method not in self.results[model]:
    #             self.results[model][attribution_method] = {layer: results}
    #         elif layer not in self.results[model][attribution_method]:
    #             self.results[model][attribution_method][layer] = results

    #     self.save_results()

    def save_results(self):
        self.results.to_csv(self.path, index=False)
        # with open(self.path, "w") as f:
        #     json.dump(self.results, f, indent=4)
