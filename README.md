# intrinsic-properties
Code for our ICLR 2024 paper "The Effect of Intrinsic Dataset Properties on Generalization: Unraveling Learning Differences Between Natural and Medical Images". Using this code you can 
1. measure intrinsic properties of your dataset: label sharpness $\hat{K}_F$ or intrinsic dimension $d_{\text{data}}$,
2. measure the intrinsic dimension of your model's learned representations $d_{\text{repr}}$ in some layer

We also provide all code used to reproduce the experiments in our paper:
1. `train.py`: Train multiple models on the different datasets.
2. `estimate_dataID_allmodels.py`: Estimate the intrinsic dimension of the training sets of multiple models.
3. `estimate_reprID_allmodels.py`: Estimate the intrinsic dimension of the learned representations of multiple models, for model layers of choice.
4. `adv_atk_allmodels.py`: Evaluate the robustness of multiple models to adversarial attack.