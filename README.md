# Conditional Motion In-Betweening (CMIB)

This is a fork of the [official CMIB repo][repo] where the implementation
of the [Conditional Motion In-betweeening][paper] lives.

[repo]: https://github.com/jihoonerd/Conditional-Motion-In-Betweening/
[paper]: https://www.sciencedirect.com/science/article/pii/S0031320322003752

## Environments

This repo is tested on the following environment:

* Windows 10
* Python >= 3.7     # 3.10.11 works for this machine
* PyTorch == 2.3.1
* Cuda V12.3.52

## Prerequisites

- [Git LFS][glfs] for the LAFAN1 dataset

- [LAFAN1 dataset][lafan] if you wish to retrain the model yourself
  ```bash
  $ git clone https://github.com/ubisoft/ubisoft-laforge-animation-dataset LAFAN1
  $ cd LAFAN1
  $ python evaluate.py  # just to verify that the data you got is correct
  ```
- Check that the CUDA version on your hardware is the same as above.
  ```bash
  $ nvcc --version  # this lets you check the CUDA version installed
  $ nvidia-smi      # verify that your card supports at least that version
  ```
- This would be a good point to create and activate a virtual environment using
  the built-in `venv` or some other tool like Anaconda, so that the packages we
  install here are used only for this project. Here's how to do it with `venv`:
  ```bash
  $ python -m venv cmib             # or pick some other name
  $ source cmib/Scripts/activate    # activate the virtual environment
  (cmib) $                          # this prompt means you're in the virtual environment
  ```
  Once you're done with the virtual environment, you can `deactivate` it.

- Install the version of pytorch [closest to your CUDA version][cuda]. This can
  be lower than the one you have, but not higher. For 2.3.1 with CUDA 12.1:
  ```bash
  $ pip install torch torchvision torchaudio --index-url https://download.pytorch/org/whl/cu121
  ```

- Install the required packages. By default, this also contains pytorch; to change
  the version, you can loosen the requirements by changing the `torch==2.31+cu121`
  to just `torch`, and doing the same for the `torchaudio` and `torchvision` modules.

  If you've already installed PyTorch, torchaudio, and torchvision above, you can
  remove the `torch==<version>` lines completely to skip downloading these.

  Once you've made the necessary changes to the requirements file, install from it:
  ```bash
  $ pip install -r requirements.txt
  ```

[glfs]: https://git-lfs.com
[lafan]: https://github.com/ubisoft/ubisoft-laforge-animation-dataset
[cuda]: https://pytorch.org/get-started/locally/

## Trained Weights

You can download trained weights from [here](https://works.do/FCqKVjy).

## Train from Scratch

Training script is `trainer.py`. For convenience, `train.sh` has the following
arguments populated by default, so you can make changes there.

```bash
python trainer.py \
    --processed_data_dir="processed_data_80/" \
    --window=90 \
    --batch_size=32 \
    --epochs=5000 \
    --device=0 \
    --entity=cmib_exp \
    --exp_name="cmib_80" \
    --save_interval=50 \
    --learning_rate=0.0001 \
    --loss_cond_weight=1.5 \
    --loss_pos_weight=0.05 \
    --loss_rot_weight=2.0 \
    --from_idx=9 \
    --target_idx=88 \
    --interpolation='slerp'

```

## Inference

You can use `run_cmib.py` for inference. Please refer to help page of `run_cmib.py` for more details.

```python
python run_cmib.py --help
```

## Reference

* LAFAN1 Dataset
  ```
  @article{harvey2020robust,
  author    = {Félix G. Harvey and Mike Yurick and Derek Nowrouzezahrai and Christopher Pal},
  title     = {Robust Motion In-Betweening},
  booktitle = {ACM Transactions on Graphics (Proceedings of ACM SIGGRAPH)},
  publisher = {ACM},
  volume    = {39},
  number    = {4},
  year      = {2020}
  }
  ```

## Citation
```
@article{KIM2022108894,
title = {Conditional Motion In-betweening},
journal = {Pattern Recognition},
pages = {108894},
year = {2022},
issn = {0031-3203},
doi = {https://doi.org/10.1016/j.patcog.2022.108894},
url = {https://www.sciencedirect.com/science/article/pii/S0031320322003752},
author = {Jihoon Kim and Taehyun Byun and Seungyoun Shin and Jungdam Won and Sungjoon Choi},
keywords = {motion in-betweening, conditional motion generation, generative model, motion data augmentation},
abstract = {Motion in-betweening (MIB) is a process of generating intermediate
skeletal movement between the given start and target poses while preserving the
naturalness of the motion, such as periodic footstep motion while walking.
Although state-of-the-art MIB methods are capable of producing plausible motions
given sparse key-poses, they often lack the controllability to generate motions
satisfying the semantic contexts required in practical applications. We focus on
the method that can handle pose or semantic conditioned MIB tasks using a
unified model. We also present a motion augmentation method to improve the
quality of pose-conditioned motion generation via defining a distribution over
smooth trajectories. Our proposed method outperforms the existing
state-of-the-art MIB method in pose prediction errors while providing additional
controllability. Our code and results are available on our project web page:
https://jihoonerd.github.io/Conditional-Motion-In-Betweening}
}
```

## Author

* [Jihoon Kim](https://github.com/jihoonerd)
* [Taehyun Byun](https://github.com/childtoy)
