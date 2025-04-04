import os
import pandas as pd


class ResultMetrics:
    """
    The results are saved in a csv file with the following header:
    Model,Attribution Method,Layer,Metric,Upscale Method,Value
    """

    def __init__(self, path: str):
        self.path = path
        self.HEADER = [
            "Image Index",
            "Label",
            "Predicted Label",
            "Model",
            "Dataset",
            "Attribution Method",
            "Layer",
            "Metric",
            "Upscale Method",
            "Mixing Method",
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
        self,
        model,
        attribution_method,
        dataset,
        layer,
        metric,
        upscale_method,
        mixing_method,
        value,
        label=-1,
        predicted_label=-1,
        image_index=-1,
    ):
        new_row = pd.DataFrame(
            [
                {
                    "Image Index": image_index,
                    "Label": label,
                    "Predicted Label": predicted_label,
                    "Model": model,
                    "Dataset": dataset,
                    "Attribution Method": attribution_method,
                    "Layer": layer,
                    "Metric": metric,
                    "Upscale Method": upscale_method,
                    "Mixing Method": mixing_method,
                    "Value": value,
                }
            ]
        )

        self.results = pd.concat([self.results, new_row], ignore_index=True)

        self.save_results()

    def save_results(self):
        self.results.to_csv(self.path, index=False)

    def get_last_image_index(self):
        if self.results.empty:
            return -1
        else:
            return self.results["Image Index"].max()
