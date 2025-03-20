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
        self,
        model,
        attribution_method,
        layer,
        metric,
        upscale_method,
        value,
        image_index=-1,
    ):
        # Add result to the results dataframe
        self.results = pd.concat(
            [
                self.results,
                pd.DataFrame(
                    [
                        [
                            image_index,
                            model,
                            attribution_method,
                            layer,
                            metric,
                            upscale_method,
                            value,
                        ]
                    ],
                    columns=self.HEADER,
                ),
            ]
        )

        self.save_results()

    def save_results(self):
        self.results.to_csv(self.path, index=False)
