# CORUN 🏃 | Colabator 🏃🏻‍♂️
Official Code for "Real-world Image Dehazing with Coherence-based Label Generator and Cooperative Unfolding Network" [Arxiv](#)

[Chenyu Fang](https://cnyvfang.github.io/), [Chunming He](https://chunminghe.github.io/), Fengyang Xiao, [Yulun Zhang](https://yulunzhang.com), Longxiang Tang, Yuelin Zhang, [Kai Li](https://kailigo.github.io), and Xiu Li

**Abstract:** Real-world Image Dehazing (RID) aims to alleviate haze-induced degradation in real-world settings. This task remains challenging due to the complexities in accurately modeling real haze distributions and the scarcity of paired real-world data. To address these challenges, we first introduce a cooperative unfolding network that jointly models atmospheric scattering and image scenes, effectively integrating physical knowledge into deep networks to restore haze-contaminated details. Additionally, we propose the first RID-oriented iterative mean-teacher framework, termed the Coherence-based Label Generator, to generate high-quality pseudo labels for network training. Specifically, we provide an optimal label pool to store the best pseudo-labels during network training, leveraging both global and local coherence to select high-quality candidates and assign weights to prioritize haze-free regions. We verify the effectiveness of our method, with experiments demonstrating that it achieves state-of-the-art performance on RID tasks.  


<details>
<summary>🏃 The architecture of the proposed CORUN with the details at k-th stage (CORUN)</summary>
<center>
    <img
    src="figs/Arch.jpg">
</center>
</details>

<details>
<summary>🏃🏻‍♂️ The plug-and-play Coherence-based Pseudo Labeling paradigm (Colabator)</summary>
<center>
    <img
    src="figs/CPL.jpg">
    <br>
</center>
</details>



## 🔥 News
- **2024-06-12:** We release the results and acknowledgements of this work.
- **2024-05-28:** We release this repository, the preprint of full paper will be release soon.



## 🔧 Todo 
- [ ] Complete this repository
- [ ] Release the preprint


## 🔗 Contents

- [ ] Datasets
- [ ] Training
- [ ] Testing
- [x] [Results](https://github.com/cnyvfang/CORUN-Colabator/blob/main/README.md#-results)
- [ ] Citation
- [x] [Acknowledgements](https://github.com/cnyvfang/CORUN-Colabator/blob/main/README.md#-acknowledgements)


## 🔍 Results

We achieved state-of-the-art performance on *RTTS* and *Fattal's* datasets and corresponding downstream tasks. More results can be found in the paper.

<details>
<summary>Quantitative Comparison (click to expand)</summary>

- Quantitative results on RTTS
  <p align="center">
  <img width="900" src="figs/tab-1.png">
  </p>
- User study scores on RTTS and Fattal’s data
  <p align="center">
  <img width="900" src="figs/tab-2.png">
  </p>
- Object detection results on RTTS
  <p align="center">
  <img width="900" src="figs/tab-3.png">
  </p>  
  </details>

<details> 
<summary>Visual Comparison (click to expand)</summary>

- Results of cutting-edge methods based on deep unfolding networks.
  <p align="center">
  <img width="900" src="figs/DUN.jpg">
  </p>
- Visual comparison on RTTS
  <p align="center">
  <img width="900" src="figs/RTTS.jpg">
  </p>
- Visual comparison on Fattal’s data
  <p align="center">
  <img width="900" src="figs/Fattal.jpg">
  </p>
- Visual comparison of object detection on RTTS
  <p align="center">
  <img width="900" src="figs/detection.jpg">
  </p>
  
  </details>


## 📎 Citation

If you find the code helpful in your resarch or work, please cite the following paper(s).

```
The bib of this paper will be released soon.
```


## 💡 Acknowledgements
The codes are based on [BasicSR](https://github.com/XPixelGroup/BasicSR). Please also follow their licenses. Thanks for their awesome works.
