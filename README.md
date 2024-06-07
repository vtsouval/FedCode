# FedCode: Communication-Efficient Federated Learning via Transferring Codebooks

Federated Learning (FL) is a distributed machine learning paradigm that enables learning models from decentralized local data, offering significant benefits for clients' data privacy. Despite its appealing privacy properties, FL faces the challenge of high communication burdens, necessitated by the continuous exchange of model weights between the server and clients. To mitigate these issues, existing communication-efficient FL approaches employ model compression techniques, such as pruning and weight clustering; yet, the need to transmit the entire set of weight updates at each federated round — even in a compressed format — limits the potential for a substantial reduction in communication volume. In response, we propose~\method, a novel FL training regime directly utilizing codebooks, i.e., the cluster centers of updated model weight values, to significantly reduce the bidirectional communication load, all while minimizing computational overhead and preventing substantial degradation in model performance. To ensure a smooth learning curve and proper calibration of clusters between the server and clients through the periodic transfer of compressed model weights, following multiple rounds of exclusive codebook communication. Our comprehensive evaluations across various publicly available vision and audio datasets on diverse neural architectures demonstrate that~\method~achieves a $12.4$-fold reduction in data transmission on average, while maintaining models' performance on par with \textit{FedAvg}, incurring a mere average accuracy loss of just $1.65$\%.

A complete description of our work can be found in our paper [arXiv](https://arxiv.org/abs/2311.09270).

## Installation

Ensure that all required python packages are install by running the following:

```
pip3 install -r requirements.txt
```

## Execution Instructions

From the root of this repo, start an experiment by executing :
```
# FedAvg
python3 ./src/fedavg.py --dataset cifar10 --model mobilenet --num_clients 10 --num_rounds 100 --split iid
# FedAvg + Weight Clustering
python3 ./src/fedavg.py --dataset cifar10 --model mobilenet --num_clients 10 --num_rounds 100 --split iid --num_clusters 64
# Federated Learning with Codebook Transfer
python3 ./src/fedcode.py --dataset cifar10 --model mobilenet --num_clients 10 --num_rounds 100 --split iid --num_clusters 64 --r_cb 2 --f1 5 --f2 2
```

# References

If you find this work useful in your research, please consider citing our paper:

<pre>@misc{khalilian2023fedcode,
      title={FedCode: Communication-Efficient Federated Learning via Transferring Codebooks}, 
      author={Saeed Khalilian and Vasileios Tsouvalas and Tanir Ozcelebi and Nirvana Meratnia},
      year={2023},
      eprint={2311.09270},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
</pre>
