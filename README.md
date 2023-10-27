# FedCode: Communication-Efficient Federated Learning via Transferring Codebooks

Federated Learning (FL) is a distributed machine learning paradigm that enables learning models from decentralized local datasets, where only models' weights are exchanged between a server and the clients. While FL offers appealing properties for clients' data privacy, it imposes high communication burdens, especially for resource-constrained clients. Existing approaches on communication-efficient FL rely on deep model compression techniques, such as pruning and weight clustering, to reduce the model size and, consequently, the size of communicated weight updates. However, transmitting the entire weight updates at each federated round, even in a compressed format, limits the potential for a substantial reduction in transmitted data volume. In this work, we propose~\method, which primarily focuses on transmitting cluster centers derived from clustering the model weight values. To ensure a smooth learning curve and proper calibration of clusters between the server and the clients, we periodically transfer the entire weights after a few rounds of solely communicating cluster centers. This approach allows a significant reduction in bidirectional transmitted data without imposing significant computational overhead on the clients or leading to major performance degradation of the models. We evaluated the effectiveness of~\method~using various publicly available datasets with ResNet-20 and MobileNet backbone model architectures. Our evaluations demonstrate an 11.7-fold data transmission reduction on average while maintaining a comparable model performance with an average accuracy loss of $1.29\%$ compared to \textit{FedAvg}. Further validation of~\method~performance under non-IID data distributions showcased an average accuracy loss of $0.91\%$ compared to \textit{FedAvg} while achieving approximately a 14-fold data transmission reduction. Our experiments have shown that~\method~has the potential to enhance communication efficiency and scalability in FL, particularly for resource-constrained clients facing limitations in terms of communication bandwidth and power resources.

# Installation

# Experiments

# References

If you find this work useful in your research, please consider citing our paper:

