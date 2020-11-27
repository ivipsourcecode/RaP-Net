# RaP-Net

RaP-Net apply **R**egion-wise weight, reflecting the semantic static attribute, to re-weight **P**oint-wise  reliability of each pixel and extract local features for robust indoor localization. Technical details are described in [this paper]() (submit to ICRA 2021)

> ```
> Dongjiang Li, Jinyu Miao, Xuesong Shi, Yuxin Tian, Qiwei Long, Ping Guo, Hongfei Yu, Wei Yang, Haosong Yue, Qi Wei, Fei Qiao, "RaP-Net: A Region-wise and Point-wise Weighting Network to Extract Robust Keypoints for Indoor Localization," 
> ```

If you use RaP-net in an academic work, please cite:
```
@InProceedings{li2020rapnet,
  author = {Dongjiang, Li and Jinyu, Miao and Xuesong, Shi and Yuxin, Tian and Qiwei, Long and Ping, Guo and Hongfei, Yu and Wei, Yang and Haosong, Yue and Qi, Wei and Fei, Qiao},
  title = {{RaP-Net: A Region-wise and Point-wise Weighting Network to Extract Robust Keypoints for Indoor Localization}},
  journal={},
  year = {2020},
}
```

 The code to hard detection is customized from [D2-Net](https://github.com/mihaidusmanu/d2-net)



## 1.Prerequisites

Python 3.7 is recommended for running our code. [Conda](https://docs.conda.io/en/latest/) can be used to install the required packages:

```
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
conda install h5py imageio imagesize matplotlib numpy scipy tqdm shutil
```



## 2. Models

The off-the-shelf weights of **basenet** and **overall RaP-Net** are also provided in `models/rapnet.basenet.pth` and `models/rapnet.overall.pth` respectively. They can be downloaded from release 1.0.



## 3. Training

We train the basenet and region-wise weight in RaP-Net separately. The related code will be open-sourced later.



## 4. Inference

`extract_features.py` can be used to extract D2 features for a given list of images. The output format can be either [`npz`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.savez.html) or `mat`. In either case, the feature files encapsulate three arrays: 

> \- `keypoints` [`N x 3`] array containing the positions of keypoints `x, y` and the scales `s`. The positions follow the COLMAP format, with the `X` axis pointing to the right and the `Y` axis to the bottom.
>
> \- `scores` [`N`] array containing the activations of keypoints (higher is better).
>
> \- `descriptors` [`N x 512`] array containing the L2 normalized descriptors.



```python extract_features.py --image_list /path/to/image/list/ --file_type jpg(png or jpeg) --model_file models/rapnet.overall.pth  ```
