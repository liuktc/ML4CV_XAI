![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

# ML4CV Explainability

This repo contains the project for the exam of the course ["AI in Industry"](https://www.unibo.it/en/study/phd-professional-masters-specialisation-schools-and-other-programmes/course-unit-catalogue/course-unit/2024/446609) and ["Machine Learning for Computer Vision"](https://www.unibo.it/en/study/phd-professional-masters-specialisation-schools-and-other-programmes/course-unit-catalogue/course-unit/2024/446614). The goal of the project is to implement the ShapCAM algorithm explained in the paper [Shap-CAM: Visual Explanations for
Convolutional Neural Networks based on
Shapley Value](https://arxiv.org/abs/2208.03608v1).

## Installation

1. Clone the repository:

```bash
git clone https://github.com/liuktc/ML4CV_XAI
cd ML4CV_XAI
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Project Structure

The project is structured as follows:

```
├── data/                   # Dataset files
├── metrics/                # Evaluation metrics implementation
├── models/                 # Model architectures
├── notebooks/              # Jupyter notebooks
├── results/                # Metrics results storage
├── utils/                  # Utility functions
├── main.ipynb              # Main execution notebook
└── requirements.txt        # Project dependencies
```

## Usage

The main interface is provided through `main.ipynb`. The model weight are not stored here on github due to their size. Due to a OneDrive link sharing limitation, I can't provide a direct link to download the weights. If you want to run the code, please contact me. You can also train the model from scratch by running the `fine_tuning.ipynb` notebook.

## Results

Results are stored in JSON format with the following structure:

```json
{
    "model_name": {
        "attribution_method": {
            "layer_name": {
                "metric_name": value
            }
        }
    }
}
```

## Documentation

For all the details refer to [project report](https://github.com/liuktc/ML4CV_XAI/blob/main/report.pdf).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any question, feel free to contact me at luca.domeniconi5@studio.unibo.it
