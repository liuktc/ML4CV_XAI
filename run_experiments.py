import yaml
import torch
import numpy as np
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
import logging  # Logging added

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("evaluation.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)
logger.info("Starting program and setting up environment.")

# Import your modules
from utils import (
    _GradCAMPlusPlus,
    _ShapleyCAM,
    _ScoreCAM,
    _EigenCAM,
    _LayerCAM,
    SimpleUpsampling,
    ERFUpsamplingFast,
    min_max_normalize,
    MultiplierMix,
    IdentityMix,
)
from data import imagenettewoof, SynteticFigures, Binarize, imagenet

from results.results_metrics import ResultMetrics
from metrics import (
    ROC_AUC,
    DeletionCurveAUC,
    InsertionCurveAUC,
    Infidelity,
    AverageDrop,
    Coherency,
    Complexity,
    RoadCombined,
    Mass_IOU
)
from models import (
    vgg11_Imagenettewoof,
    vgg11_Syntetic,
    vgg11_Imagenet,
    vgg_preprocess,
    resnet18_Imagenettewoof,
    resnet50_Imagenettewoof,
    resnet18_Syntetic,
    resnet50_Syntetic,
    resnet18_Imagenet,
    resnet50_Imagenet,
    resnet_preprocess,
)

from tqdm.auto import tqdm

torch.manual_seed(123)
np.random.seed(123)

# Read the config filename from command line arguments
import argparse
parser = argparse.ArgumentParser(description="Run experiments with ML4CV_XAI.")
parser.add_argument(
    "--config",
    type=str,
    required=True,
    # default="config.yaml",
    help="Path to the configuration file.",
)
args = parser.parse_args()
logger.info(f"Using config file: {args.config}")


# Load config
with open(args.config, "r") as f:
    config = yaml.safe_load(f)
    config = {k.lower(): v for k, v in config.items()}
    for k, v in config.items():
        if "path" in k:
            continue
        if isinstance(v, list):
            config[k] = [x.lower() for x in v]
        elif isinstance(v, str):
            config[k] = v.lower()
        else:
            config[k] = v

device = config.get("device", "cpu")
if "cuda" in device and not torch.cuda.is_available():
    raise ValueError("CUDA is not available. Please set device to 'cpu'.")

logger.info("Loaded and processed configuration from config.yaml")

##################################################
# CONSTANTS
##################################################

MODELS = ["vgg11", "resnet18", "resnet50"]
DATASETS = ["imagenettewoof", "synthetic", "imagenet"]
ATTRIBUTION_METHODS = [
    "GradCAMPlusPlus",
    "ShapleyCAM",
    "ScoreCAM",
    "EigenCAM",
    "LayerCAM",
]
UPSCALE_METHODS = ["SimpleUpsampling", "ERFUpsamplingFast"]
METRICS = [
    "ROAD_combined",
    "ROC_AUC",
    "DeletionCurveAUC",
    "InsertionCurveAUC",
    "Infidelity",
    "AverageDrop",
    "Coherency",
    "Complexity",
    "Mass_IOU",	
]

# Convert all the constant lists to lowercase
MODELS = [m.lower() for m in MODELS]
DATASETS = [d.lower() for d in DATASETS]
ATTRIBUTION_METHODS = [m.lower() for m in ATTRIBUTION_METHODS]
UPSCALE_METHODS = [m.lower() for m in UPSCALE_METHODS]
METRICS = [m.lower() for m in METRICS]


###############################################
# CONFIGURATION VALIDATION
###############################################

if config["model"] not in MODELS:
    raise ValueError(f"Model {config['model']} not in {MODELS}.")
if config["dataset"] not in DATASETS:
    raise ValueError(f"Dataset {config['dataset']} not in {DATASETS}.")
if not all(m in ATTRIBUTION_METHODS for m in config["attribution_methods"]):
    raise ValueError(
        f"Attribution methods {config['attribution_methods']} not in {ATTRIBUTION_METHODS}."
    )
if not all(m in UPSCALE_METHODS for m in config["upscale_methods"]):
    raise ValueError(
        f"Upscale methods {config['upscale_methods']} not in {UPSCALE_METHODS}."
    )
if not all(m in METRICS for m in config["metrics"]):
    raise ValueError(f"Metrics {config['metrics']} not in {METRICS}.")
logger.info(
    f"Configuration validated: Model={config['model']}, Dataset={config['dataset']}, "
    f"Attribution Methods={config['attribution_methods']}, Upscale Methods={config['upscale_methods']}, "
    f"Metrics={config['metrics']}"
)


#################################################
# MODEL
#################################################

if config["dataset"] == "imagenettewoof":
    models_map = {
        "vgg11": vgg11_Imagenettewoof,
        "resnet18": resnet18_Imagenettewoof,
        "resnet50": resnet50_Imagenettewoof,
    }
elif config["dataset"] == "synthetic":
    models_map = {
        "vgg11": vgg11_Syntetic,
        "resnet18": resnet18_Syntetic,
        "resnet50": resnet50_Syntetic,
    }
elif config["dataset"] == "imagenet":
    models_map = {
        "vgg11": vgg11_Imagenet,
        "resnet18": resnet18_Imagenet,
        "resnet50": resnet50_Imagenet,
    }
else:
    raise ValueError(f"Dataset {config['dataset']} not supported.")

model = models_map[config["model"]]()
model.to(device)
model.eval()
logger.info(f"Model '{config['model']}' initialized.")
if "model_path" in config and config["model_path"]:
    model.load_state_dict(torch.load(config["model_path"], map_location=device))
    logger.info(f"Model weights loaded from {config['model_path']}.")
else:
    logger.warning("No model path provided. Using default weights.")
    logger.info("Model weights not loaded.")

preprocess_map = {
    "vgg11": vgg_preprocess,
    "resnet18": resnet_preprocess,
    "resnet50": resnet_preprocess,
}
preprocess = preprocess_map[config["model"]]

###############################################
# DATASET
###############################################

if config["dataset"] == "imagenettewoof":
    test_data = imagenettewoof(
        split="test", size="320px", download=False, transform=preprocess
    )
elif config["dataset"] == "synthetic":
    TRAIN_SIZE = 8
    TEST_SIZE = 6 * 50
    background_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(
                brightness=0.25, contrast=0.15, saturation=0.15, hue=0.15
            ),
        ]
    )
    mask_preprocess = transforms.Compose(
        [
            transforms.Resize((224, 224), interpolation=InterpolationMode.NEAREST),
            transforms.GaussianBlur(kernel_size=15),
            transforms.ToTensor(),
            Binarize(),
        ]
    )
    test_data = SynteticFigures(
        background_path="./data/WaldoNoise",
        num_images=TEST_SIZE,
        split="test",
        num_shapes_per_image=1,
        image_transform=preprocess,
        background_transform=background_transform,
        mask_preprocess=mask_preprocess,
        size_range=(80, 100),
    )
elif config["dataset"] == "imagenet":
    test_data = imagenet(root="/media/data/ldomeniconi/imagenet_root", split="val", transform=preprocess)
else:
    raise ValueError(f"Dataset {config['dataset']} not supported.")
logger.info(f"Dataset '{config['dataset']}' loaded with {len(test_data)} samples.")


#############################################
# ATTRIBUTION METHODS
#############################################

attr_methods_map = {
    "gradcamplusplus": _GradCAMPlusPlus,
    "shapleycam": _ShapleyCAM,
    "scorecam": _ScoreCAM,
    "eigencam": _EigenCAM,
    "layercam": _LayerCAM,
}
attr_methods = [attr_methods_map[m]() for m in config["attribution_methods"]]
logger.info(f"Attribution methods initialized: {config['attribution_methods']}")

# Upscale methods
upscale_map = {
    "simpleupsampling": lambda: SimpleUpsampling((224, 224)),
    "erupsamplingfast": ERFUpsamplingFast,
}
upscale_methods = [upscale_map[m]() for m in config["upscale_methods"]]
logger.info(f"Upscale methods initialized: {config['upscale_methods']}")


#############################################
# METRICS
#############################################

metric_map = {
    "road_combined": RoadCombined,
    "roc_auc": ROC_AUC,
    "deletioncurveauc": DeletionCurveAUC,
    "insertioncurveauc": InsertionCurveAUC,
    "infidelity": Infidelity,
    "averagedrop": AverageDrop,
    "coherency": Coherency,
    "complexity": Complexity,
    "mass_iou": Mass_IOU,
}
metrics = [metric_map[m]() for m in config["metrics"]]
logger.info(f"Evaluation metrics initialized: {config['metrics']}")

# Main evaluation loop
logger.info("Starting main evaluation loop.")


#########################################
# LAYERS
#########################################

if config["model"] == "vgg11":
    layers = [
        model.features[20],
        model.features[15],
        model.features[10],
        model.features[5],
    ]
elif config["model"] == "resnet18":
    layers = [model.layer4, model.layer3, model.layer2, model.layer1]
elif config["model"] == "resnet50":
    layers = [model.layer4, model.layer3, model.layer2, model.layer1]
else:
    raise ValueError(f"Model {config['model']} not supported.")
layers_names_map = {
    "vgg11": ["features.20", "features.15", "features.10", "features.5"],
    "resnet18": ["layer4", "layer3", "layer2", "layer1"],
    "resnet50": ["layer4", "layer3", "layer2", "layer1"],
}
layers_names = layers_names_map[config["model"]]


#######################################
# NUM_SAMPLES
#######################################

if "num_samples" in config:
    num_samples = config["num_samples"]
    if num_samples > len(test_data):
        raise ValueError(
            f"Number of samples {num_samples} exceeds dataset size {len(test_data)}."
        )
    # test_data = torch.utils.data.Subset(test_data, range(num_samples))
    logger.info(f"Using {num_samples} samples for evaluation.")
    INDICES = np.random.choice(
        len(test_data), num_samples, replace=False
    )
else:
    INDICES = np.arange(len(test_data))
    logger.info(f"Using all {len(test_data)} samples for evaluation.")


#########################################
# MAIN LOOP
#########################################

results = ResultMetrics(config["output_path"])

for index in tqdm(INDICES):
    try:
        logger.debug(f"Processing image index {index}")
        sample = test_data[index]
        if len(sample) == 2:
            images, labels = sample
            mask = None
        else:
            images, mask, labels = sample
            mask = mask.unsqueeze(0).to(device)

        images = images.unsqueeze(0).to(device)
        labels = torch.tensor([labels], dtype=torch.long).to(device)
        pred_label = model(images).argmax(dim=1)

        for attr in attr_methods:
            for upsampler in upscale_methods:
                layer_attributions = []
                for layer, layer_name in zip(layers, layers_names):
                    attr_map = attr.attribute(input_tensor=images,
                                                model=model,
                                                layer=layer,
                                                target=labels)
                    attr_map = upsampler(attribution=attr_map,
                                        image=images,
                                        device=device,
                                        model=model,
                                        layer=layer)

                    if (
                        torch.abs(
                            attr_map.amax((2, 3), keepdim=True)
                            - attr_map.amin((2, 3), keepdim=True)
                        )
                        < 1e-6
                    ).any():
                        logger.warning(
                            f"Skipping constant saliency map at image index {index}"
                        )
                        continue

                    attr_map = min_max_normalize(attr_map)
                    layer_attributions.append(attr_map)

                    mix = MultiplierMix("all")
                    mixed_map = mix(layer_attributions)
                    mixed_map = min_max_normalize(mixed_map)

                    for metric in metrics:
                        for name, map_in_use, mix_method in [
                            ("Normal", attr_map, IdentityMix()),
                            ("Mixed", mixed_map, mix),
                        ]:
                            result = metric(
                                model=model,
                                test_images=images,
                                saliency_maps=map_in_use,
                                class_idx=labels,
                                attribution_method=attr,
                                device=device,
                                layer=layer,
                                upsample_method=upsampler,
                                mixer=mix_method,
                                previous_attributions=layer_attributions[:-1],
                                mask=mask
                            )
                            if isinstance(result, torch.Tensor):
                                result = result.item()

                            results.add_result(
                                model=config["model"],
                                attribution_method=attr.name,
                                dataset=config["dataset"],
                                layer=layer_name,
                                metric=metric.name,
                                upscale_method=upsampler.name,
                                mixing_method=mix_method.name,
                                value=result,
                                image_index=index,
                                label=labels[0].item(),
                                predicted_label=pred_label[0].item(),
                            )
                            logger.debug(
                                f"Recorded result: Index={index}, Layer={layer_name}, Attr={attr.name}, "
                                f"Metric={metric.name}, Mix={mix_method.name}"
                            )
    except Exception as e:
        logger.error(f"Error processing index {index}: {e}", exc_info=True)

results.save_results()
logger.info("Results saved successfully.")
