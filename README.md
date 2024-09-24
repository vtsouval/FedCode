# FedCode: Communication-Efficient Federated Learning via Transferring Codebooks

Federated Learning (FL) is a distributed machine learning paradigm that enables learning models from decentralized local data, offering significant benefits for clients' data privacy. Despite its appealing privacy properties, FL faces the challenge of high communication burdens, necessitated by the continuous exchange of model weights between the server and clients. To mitigate these issues, existing communication-efficient FL approaches employ model compression techniques, such as pruning and weight clustering; yet, the need to transmit the entire set of weight updates at each federated round — even in a compressed format — limits the potential for a substantial reduction in communication volume. In response, we propose FedCode, a novel FL training regime directly utilizing codebooks, i.e., the cluster centers of updated model weight values, to significantly reduce the bidirectional communication load, all while minimizing computational overhead and preventing substantial degradation in model performance. To ensure a smooth learning curve and proper calibration of clusters between the server and clients through the periodic transfer of compressed model weights, following multiple rounds of exclusive codebook communication. Our comprehensive evaluations across various publicly available vision and audio datasets on diverse neural architectures demonstrate that FedCode achieves a $12.4$-fold reduction in data transmission on average, while maintaining models' performance on par with FedAvg, incurring a mere average accuracy loss of just $1.65$\%.

<img src=./img/overview.png width=75%/>

A complete description of our work can be found in our paper [arXiv](https://arxiv.org/abs/2311.09270).

## Installation

Ensure that all required python packages are install by running the following:

```
pip3 install -r requirements.txt
```

## Execution Instructions

From the root of this repo, start an experiment by executing :
```
# Federated Learning with Codebook Transfer
cd ./src && python3 ./src/main.py --dataset cifar10 --model mobilenet --num_clients 10 --num_rounds 100 --split iid \
      --num_clusters 64 \ # number of clusters
      --start_rnd 2 \ # round to start clustering
      --broadcast_rate 5 \ # f1
      --request_rate 2 # f2
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
