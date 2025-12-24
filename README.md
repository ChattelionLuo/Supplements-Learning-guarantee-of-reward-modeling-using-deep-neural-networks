This folder contains necessary code files to reproduce the experiments in the article **Learning guarantee of reward modeling using deep neural networks**

The structure of this repository is as follows:
<working directory>
```
├─README.md
├─plot_two.py
├─CodeR_BTmodel
    ├─ result/
    ├─ ready.py
    ├─ funcs.py
    ├─ sim1.py
    ├─ sim2.py
    ├─ sim3.py
    ├─ run_toy1.sh
    ├─ run_toy2.sh
    ├─ run_toy3.sh
    ├─ Regret.png
├─CodeR_THmodel
    ├─ result/
    ├─ ready.py
    ├─ funcs.py
    ├─ sim1.py
    ├─ sim2.py
    ├─ sim3.py
    ├─ run_toy1.sh
    ├─ run_toy2.sh
    ├─ run_toy3.sh
    ├─ Regret.png
```

- result/: the result files for all  tasks. It is used by the ready.py to generate figures.
- run_toy1.sh: scripts to train models and save output results for Figure 3.
- run_toy2.sh: scripts to train models and save output results for Figure 4.
- run_toy3.sh: scripts to train models and save output results for Figure 5.
- sim1.py: python code for running a single configuration for Figure 3.
- sim2.py: python code for running a single configuration for Figure 4.
- sim3.py: python code for running a single configuration for Figure 5.
- funcs.py: functions utilized to generate dataset and define the neural network structure.
- plot_figures.py: Given all results from both BT models and Thurstonian model are ready, this file visualize results from both models. User can adjust the code to generate different figures as needed.

**Workflow**

Preparations
- Install the PyTorch framework, following the installation guides at https://pytorch.org/get-started/locally/.
- Install necessary Python packages as in the Software Environment section.
- Check local working directory and structure of this repository.

Run the code for BT model under the working dictionary:
```
    cd ./CodeR_BTmodel
    bash run_toy1.sh
    bash run_toy2.sh
    bash run_toy3.sh
```

Run the result for BT model under the working dictionary:
```
    cd ./CodeR_THmodel
    bash run_toy1.sh
    bash run_toy2.sh
    bash run_toy3.sh
```

Plot the Figure under the working dictionary:
```
    python plot_two.py
```


Note.

- The experiments can be runned on a server with multiple cores. The programme run_toy1.sh may take several hours to finish depending on the hardware configurations. Other scripts may take less time.

**Development**
The repository is released and maintained by Yuanhang Luo(chattelion.luo@connect.polyu.hk) and Yeheng GE(geyh96@foxmail.com).

**Reference**
- Yuanhang Luo, Yeheng Ge, Ruijian Han, and Guohao Shen. 2026. Learning Guarantee of Reward Modeling Using Deep Neural Networks. In Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data
Mining (KDD ’26). https://doi.org/10.1145/3770845.3780316



