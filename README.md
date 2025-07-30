# Efficient LoFTR: Semi-Dense Local Feature Matching with Sparse-Like Speed

### [Project Page](https://zju3dv.github.io/efficientloftr) | [Paper](https://zju3dv.github.io/efficientloftr/files/EfficientLoFTR.pdf) 
<br/>

> Efficient LoFTR: Semi-Dense Local Feature Matching with Sparse-Like Speed  
> [Yifan Wang](https://github.com/wyf2020)<sup>\*</sup>, [Xingyi He](https://github.com/hxy-123)<sup>\*</sup>, [Sida Peng](https://pengsida.net), [Dongli Tan](https://github.com/Cuistiano), [Xiaowei Zhou](http://xzhou.me)  
> CVPR 2024 Highlight

https://github.com/zju3dv/EfficientLoFTR/assets/69951260/40890d21-180e-4e70-aeba-219178b0d824

## 🌟News🌟
[July 2025] EfficientLoFTR is now officially integrated into the 🤗 Hugging Face Transformers (thanks to [@sbucaille](https://github.com/sbucaille)!).
You can easily run inference with just a few lines of code using `pip install transformers`. ([model card](https://huggingface.co/zju-community/efficientloftr))

[Feb 2025] To enhance multi-modality matching with EfficientLoFTR and improve its applicability to UAV localization, autonomous driving, and beyond, check out our latest work, [MatchAnything](https://github.com/zju3dv/MatchAnything)! Try our demo and see it in action!
## TODO List
- [x] Inference code and pretrained models
- [x] Code for reproducing the test-set results
- [x] Add options of flash-attention for better performance
- [x] [jupyter notebook demo](https://github.com/zju3dv/EfficientLoFTR/blob/089f6665722398007908977891f47f2c002f2aec/notebooks/demo_single_pair.ipynb) for matching a pair of images
- [x] Training code

## Installation
```shell
conda env create -f environment.yaml
conda activate eloftr
pip install torch==2.0.0+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt 
```
The test and training can be downloaded by [download link](https://drive.google.com/drive/folders/1DOcOPZb3-5cWxLqn256AhwUVjBPifhuf?usp=sharing) provided by LoFTR

We provide our pretrained model in [download link](https://drive.google.com/drive/folders/1GOw6iVqsB-f1vmG6rNmdCcgwfB4VZ7_Q?usp=sharing)

## Match image pairs with EfficientLoFTR

<details>
<summary><b>[Basic Usage]</b></summary>

```python
import torch
import cv2
import numpy as np
from copy import deepcopy
from src.loftr import LoFTR, full_default_cfg, reparameter

# Initialize the matcher with default settings
_default_cfg = deepcopy(full_default_cfg)
matcher = LoFTR(config=_default_cfg)

# Load pretrained weights
matcher.load_state_dict(torch.load("weights/eloftr_outdoor.ckpt")['state_dict'])
matcher = reparameter(matcher)  # Essential for good performance
matcher = matcher.eval().cuda()

# Load and preprocess images
img0_raw = cv2.imread("path/to/image0.jpg", cv2.IMREAD_GRAYSCALE)
img1_raw = cv2.imread("path/to/image1.jpg", cv2.IMREAD_GRAYSCALE)

# Resize images to be divisible by 32
img0_raw = cv2.resize(img0_raw, (img0_raw.shape[1]//32*32, img0_raw.shape[0]//32*32))
img1_raw = cv2.resize(img1_raw, (img1_raw.shape[1]//32*32, img1_raw.shape[0]//32*32))

# Convert to tensors
img0 = torch.from_numpy(img0_raw)[None][None].cuda() / 255.
img1 = torch.from_numpy(img1_raw)[None][None].cuda() / 255.
batch = {'image0': img0, 'image1': img1}

# Inference
with torch.no_grad():
    matcher(batch)
    mkpts0 = batch['mkpts0_f'].cpu().numpy()  # Matched keypoints in image0
    mkpts1 = batch['mkpts1_f'].cpu().numpy()  # Matched keypoints in image1
    mconf = batch['mconf'].cpu().numpy()      # Matching confidence scores
```

</details>

<details>
<summary><b>[Advanced Usage]</b></summary>

```python
import torch
import cv2
import numpy as np
import matplotlib.cm as cm
from copy import deepcopy
from src.loftr import LoFTR, full_default_cfg, opt_default_cfg, reparameter
from src.utils.plotting import make_matching_figure

# Model configuration options
model_type = 'full'  # Choose: 'full' for best quality, 'opt' for best efficiency
precision = 'fp32'   # Choose: 'fp32', 'mp' (mixed precision), 'fp16' for best efficiency

# Load appropriate config
if model_type == 'full':
    _default_cfg = deepcopy(full_default_cfg)
elif model_type == 'opt':
    _default_cfg = deepcopy(opt_default_cfg)

# Set precision options
if precision == 'mp':
    _default_cfg['mp'] = True
elif precision == 'fp16':
    _default_cfg['half'] = True

# Initialize matcher
matcher = LoFTR(config=_default_cfg)
matcher.load_state_dict(torch.load("weights/eloftr_outdoor.ckpt")['state_dict'])
matcher = reparameter(matcher)

# Apply precision settings
if precision == 'fp16':
    matcher = matcher.half()

matcher = matcher.eval().cuda()

# Load and preprocess images
img0_raw = cv2.imread("path/to/image0.jpg", cv2.IMREAD_GRAYSCALE)
img1_raw = cv2.imread("path/to/image1.jpg", cv2.IMREAD_GRAYSCALE)
img0_raw = cv2.resize(img0_raw, (img0_raw.shape[1]//32*32, img0_raw.shape[0]//32*32))
img1_raw = cv2.resize(img1_raw, (img1_raw.shape[1]//32*32, img1_raw.shape[0]//32*32))

# Convert to tensors with appropriate precision
if precision == 'fp16':
    img0 = torch.from_numpy(img0_raw)[None][None].half().cuda() / 255.
    img1 = torch.from_numpy(img1_raw)[None][None].half().cuda() / 255.
else:
    img0 = torch.from_numpy(img0_raw)[None][None].cuda() / 255.
    img1 = torch.from_numpy(img1_raw)[None][None].cuda() / 255.

batch = {'image0': img0, 'image1': img1}

# Inference with different precision modes
with torch.no_grad():
    if precision == 'mp':
        with torch.autocast(enabled=True, device_type='cuda'):
            matcher(batch)
    else:
        matcher(batch)
    
    mkpts0 = batch['mkpts0_f'].cpu().numpy()
    mkpts1 = batch['mkpts1_f'].cpu().numpy()
    mconf = batch['mconf'].cpu().numpy()

# Post-process confidence scores for 'opt' model
if model_type == 'opt':
    mconf = (mconf - min(20.0, mconf.min())) / (max(30.0, mconf.max()) - min(20.0, mconf.min()))

# Visualize matches
color = cm.jet(mconf)
text = ['EfficientLoFTR', 'Matches: {}'.format(len(mkpts0))]
fig = make_matching_figure(img0_raw, img1_raw, mkpts0, mkpts1, color, text=text)
```

**Configuration Options:**
- `model_type`: 
  - `'full'`: Best matching quality
  - `'opt'`: Best efficiency with minimal quality loss
- `precision`:
  - `'fp32'`: Full precision (default)
  - `'mp'`: Mixed precision for better efficiency
  - `'fp16'`: Half precision for maximum efficiency (requires modern GPU)
- **Note**: Our model is trained on MegaDepth and works best for outdoor scenes. There may be a domain gap for indoor environments.

</details>

<details>
<summary><b>[Using Transformers]</b></summary>
EfficientLoFTR is now officially integrated into the 🤗 Hugging Face Transformers (thanks to [@sbucaille](https://github.com/sbucaille)!).
You can easily run inference with just a few lines of code using `pip install transformers`. ([model card](https://huggingface.co/zju-community/efficientloftr))
Note: The default processor resizes images to a resolution of 480x640 pixels.

```python
from transformers import AutoImageProcessor, AutoModel
import torch
from PIL import Image
import requests

# Load example images (same as in the original paper)
url_image1 = "https://raw.githubusercontent.com/magicleap/SuperGluePretrainedNetwork/refs/heads/master/assets/phototourism_sample_images/united_states_capitol_98169888_3347710852.jpg"
image1 = Image.open(requests.get(url_image1, stream=True).raw)
url_image2 = "https://raw.githubusercontent.com/magicleap/SuperGluePretrainedNetwork/refs/heads/master/assets/phototourism_sample_images/united_states_capitol_26757027_6717084061.jpg"
image2 = Image.open(requests.get(url_image2, stream=True).raw)

images = [image1, image2]

# Load processor and model
processor = AutoImageProcessor.from_pretrained("zju-community/efficientloftr")
model = AutoModel.from_pretrained("zju-community/efficientloftr")

# Process images and run inference
inputs = processor(images, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
# keypoints = outputs.keypoints        # Keypoints in both images
# matches = outputs.matches            # Matching indices 
# matching_scores = outputs.matching_scores  # Confidence scores
```


**Post-process and visualize results:**

```python
# Post-process to get keypoints and matches in a readable format
image_sizes = [[(image.height, image.width) for image in images]]
outputs = processor.post_process_keypoint_matching(outputs, image_sizes, threshold=0.2)

# Print matching results
for i, output in enumerate(outputs):
    print(f"Image pair {i}:")
    print(f"Found {len(output['keypoints0'])} matches")
    for keypoint0, keypoint1, matching_score in zip(
            output["keypoints0"], output["keypoints1"], output["matching_scores"]
    ):
        print(
            f"Keypoint {keypoint0.numpy()} ↔ {keypoint1.numpy()} (score: {matching_score:.3f})"
        )

# Visualize matches
processor.visualize_keypoint_matching(images, outputs)
```

For more details, visit the [Hugging Face model card](https://huggingface.co/zju-community/efficientloftr).

</details>

An jupyter notebook about advanced usage is given in notebooks/demo_single_pair.ipynb.


## Reproduce the testing results
You need to first set up the testing subsets of ScanNet and MegaDepth. We create symlinks from the previously downloaded datasets to `data/{{dataset}}/test`.

```shell
# set up symlinks
ln -s /path/to/scannet-1500-testset/* /path/to/EfficientLoFTR/data/scannet/test
ln -s /path/to/megadepth-1500-testset/* /path/to/EfficientLoFTR/data/megadepth/test
```
### Inference time
```shell
conda activate eloftr
bash scripts/reproduce_test/indoor_full_time.sh
bash scripts/reproduce_test/indoor_opt_time.sh
```

### Accuracy
```shell
conda activate eloftr
bash scripts/reproduce_test/outdoor_full_auc.sh
bash scripts/reproduce_test/outdoor_opt_auc.sh
bash scripts/reproduce_test/indoor_full_auc.sh
bash scripts/reproduce_test/indoor_opt_auc.sh
```

## Training
```shell
conda env create -f environment_training.yaml  # used a different version of pytorch, maybe slightly different from the inference environment
pip install -r requirements.txt
conda activate eloftr_training
bash scripts/reproduce_train/eloftr_outdoor.sh eloftr_outdoor
```

## Citation

If you find this code useful for your research, please use the following BibTeX entry.

```bibtex
@inproceedings{wang2024eloftr,
  title={{Efficient LoFTR}: Semi-Dense Local Feature Matching with Sparse-Like Speed},
  author={Wang, Yifan and He, Xingyi and Peng, Sida and Tan, Dongli and Zhou, Xiaowei},
  booktitle={CVPR},
  year={2024}
}
```
