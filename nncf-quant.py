#!/usr/bin/env python
# coding: utf-8

# Note: Excerpts of this file were copied from https://docs.openvino.ai/2023.3/notebooks/112-pytorch-post-training-quantization-nncf-with-output.html

import os
import time
import zipfile
from pathlib import Path
from typing import List, Tuple
import nncf
import openvino as ov
from openvino.runtime.ie_api import CompiledModel
import torch
from torchvision.datasets import ImageFolder
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms as transforms
from notebook_utils import download_file
from typing import Union

def download_tiny_imagenet_200(
    output_dir: Path,
    url: str = "http://cs231n.stanford.edu/tiny-imagenet-200.zip",
    tarname: str = "tiny-imagenet-200.zip",
):
    archive_path = output_dir / tarname
    download_file(url, directory=output_dir, filename=tarname)
    zip_ref = zipfile.ZipFile(archive_path, "r")
    zip_ref.extractall(path=output_dir)
    zip_ref.close()
    print(f"Successfully downloaded and extracted dataset to: {output_dir}")

def create_validation_dir(dataset_dir: Path):
    VALID_DIR = dataset_dir / "val"
    val_img_dir = VALID_DIR / "images"

    fp = open(VALID_DIR / "val_annotations.txt", "r")
    data = fp.readlines()

    val_img_dict = {}
    for line in data:
        words = line.split("\t")
        val_img_dict[words[0]] = words[1]
    fp.close()

    for img, folder in val_img_dict.items():
        newpath = val_img_dir / folder
        if not newpath.exists():
            os.makedirs(newpath)
        if (val_img_dir / img).exists():
            os.rename(val_img_dir / img, newpath / img)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name: str, fmt: str = ":f"):
        self.name = name
        self.fmt = fmt
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    """Displays the progress of validation process"""

    def __init__(self, num_batches: int, meters: List[AverageMeter], prefix: str = ""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch: int):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches: int):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def accuracy(output: torch.Tensor, target: torch.Tensor, topk: Tuple[int] = (1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))

        return res
    
def validate(val_loader: torch.utils.data.DataLoader, model: Union[torch.nn.Module, CompiledModel]):
    """Compute the metrics using data from val_loader for the model"""
    batch_time = AverageMeter("Time", ":3.3f")
    top1 = AverageMeter("Acc@1", ":2.2f")
    top5 = AverageMeter("Acc@5", ":2.2f")
    progress = ProgressMeter(len(val_loader), [batch_time, top1, top5], prefix="Test: ")
    start_time = time.time()
    # Switch to evaluate mode.
    if not isinstance(model, CompiledModel):
        model.eval()
        model.to(torch_device)

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.to(torch_device)
            target = target.to(torch_device)

            # Compute the output.
            if isinstance(model, CompiledModel):
                output_layer = model.output(0)
                output = model(images)[output_layer]
                output = torch.from_numpy(output)
            else:
                output = model(images)

            # Measure accuracy and record loss.
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # Measure elapsed time.
            batch_time.update(time.time() - end)
            end = time.time()

            print_frequency = 10
            if i % print_frequency == 0:
                progress.display(i)

        print(
            " * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} Total time: {total_time:.3f}".format(top1=top1, top5=top5, total_time=end - start_time)
        )
    return top1.avg

def create_model(model_path: Path):
    model = resnet50()

    # Update the last FC layer for Tiny ImageNet number of classes.
    NUM_CLASSES = 200
    model.fc = torch.nn.Linear(in_features=2048, out_features=NUM_CLASSES, bias=True)
    model.to(torch_device)

    if model_path.exists():
        checkpoint = torch.load(str(model_path), map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
    else:
        raise RuntimeError("There is no checkpoint to load")

    return model

def create_dataloaders(batch_size: int = 128):
    train_dir = DATASET_DIR / "train"
    val_dir = DATASET_DIR / "val" / "images"
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    train_dataset = ImageFolder(
        train_dir,
        transforms.Compose(
            [
                transforms.Resize(IMAGE_SIZE),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )
    val_dataset = ImageFolder(
        val_dir,
        transforms.Compose(
            [transforms.Resize(IMAGE_SIZE), transforms.ToTensor(), normalize]
        ),
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        sampler=None,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    return train_loader, val_loader

def transform_fn(data_item):
    images, _ = data_item
    return images

# ----

torch_device = "cpu"
print(f"Using {torch_device} device")

MODEL_DIR = Path("model")
OUTPUT_DIR = Path("output")
BASE_MODEL_NAME = "resnet50"
IMAGE_SIZE = [64, 64]

OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

# Paths where PyTorch and OpenVINO IR models will be stored.
fp32_checkpoint_filename = Path(BASE_MODEL_NAME + "_fp32").with_suffix(".pth")
fp32_ir_path = OUTPUT_DIR / Path(BASE_MODEL_NAME + "_fp32").with_suffix(".xml")
int8_ir_path = OUTPUT_DIR / Path(BASE_MODEL_NAME + "_int8").with_suffix(".xml")

fp32_pth_url = "https://storage.openvinotoolkit.org/repositories/nncf/openvino_notebook_ckpts/304_resnet50_fp32.pth"
download_file(fp32_pth_url, directory=MODEL_DIR, filename=fp32_checkpoint_filename)

# Download and prepare calibration dataset
DATASET_DIR = OUTPUT_DIR / "tiny-imagenet-200"
if not DATASET_DIR.exists():
    download_tiny_imagenet_200(OUTPUT_DIR)
    create_validation_dir(DATASET_DIR)

train_loader, val_loader = create_dataloaders()

# Load model and original fp32 execute
model = create_model(MODEL_DIR / fp32_checkpoint_filename)
acc1 = validate(val_loader, model)
print(f"Test accuracy of original FP32 model: {acc1:.3f}")

# Quantize model to int8 using NNCF
calibration_dataset = nncf.Dataset(train_loader, transform_fn)
quantized_model = nncf.quantize(model, calibration_dataset)

# Execute quantized int8 model
acc1 = validate(val_loader, quantized_model)
print(f"Accuracy of quantized INT8 model: {acc1:.3f}")

# Convert and execute models using OpenVINO IR and verify same accuracy
device = "CPU"
core = ov.Core()
dummy_input = torch.randn(128, 3, *IMAGE_SIZE)

model_ir = ov.convert_model(model, example_input=dummy_input, input=[-1, 3, *IMAGE_SIZE])
fp32_compiled_model = core.compile_model(model_ir, device)
acc1 = validate(val_loader, fp32_compiled_model)
print(f"Accuracy of OpenVINO FP32 IR model: {acc1:.3f}")

quantized_model_ir = ov.convert_model(quantized_model, example_input=dummy_input, input=[-1, 3, *IMAGE_SIZE])
int8_compiled_model = core.compile_model(quantized_model_ir, device)
acc1 = validate(val_loader, int8_compiled_model)
print(f"Accuracy of OpenVINO INT8 IR model: {acc1:.3f}")

# Export to static shape (for NPU)
dummy_input = torch.randn(1, 3, *IMAGE_SIZE)
model_ir = ov.convert_model(model, example_input=dummy_input,input=[[1, 3, *IMAGE_SIZE]])
ov.save_model(model_ir, fp32_ir_path)

quantized_model_ir = ov.convert_model(quantized_model, example_input=dummy_input, input=[[1, 3, *IMAGE_SIZE]])
ov.save_model(quantized_model_ir, int8_ir_path)