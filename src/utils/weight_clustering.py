from sklearn.cluster import KMeans , MiniBatchKMeans
import torch
import copy, math
import numpy as np
import warnings
import numpy as np
import math
warnings.filterwarnings("ignore")
from concurrent.futures import ThreadPoolExecutor

def block_weight_clustering_per_section(model,layer_name,cluster_numebr,B):
    """
    The function iterates over the parameters of the model and collects the weights of the specified section.
    The collected weights are flattened and reshaped into blocks of size B.
    Weight clustering is performed using the K-means algorithm with the specified number of clusters.
    The resulting cluster centers are used to replace the original weights of the specified section.
    The compressed model with block weight clustering applied is returned.
    """
    model2 = copy.deepcopy(model)
    with torch.no_grad():
      model2.cpu()
      weights=np.array([])
      for name, params in model2.named_parameters():
                if  name in  layer_name:
                  param_shape=list(params.size()) 
                  w=params.reshape(-1,B)
                  if len(weights)==0:
                           weights=w
                  else:
                           weights = np.concatenate((weights,w ))
      for module_name, module in model2.named_children():
                    if module_name  in layer_name: #layer
                        for basic_block_name, basic_block in module.named_children():
                                    for sub_block_name, sub_block in basic_block.named_children():
                                        for params in sub_block.parameters():
                                                w=params.reshape(-1,B).numpy()
                                                if len(weights)==0:
                                                    weights=w
                                                else:
                                                    weights = np.concatenate((weights,w ))

      weights = weights.astype('double')
      kmeans = KMeans(n_clusters=cluster_numebr, init='k-means++', max_iter=100, n_init=64, random_state=0)
      kmeans.fit(weights)
      cluster_centers=torch.from_numpy(kmeans.cluster_centers_)
      start_index=0
      for name, params in model2.named_parameters():
                if  name in  layer_name:
                    param_shape=list(params.size())
                    w=params.reshape(-1,B)
                    cluster_list=[]
                    for i in range(len(w)):
                            ww=np.array(cluster_centers[kmeans.labels_[start_index]])
                            w[i]=torch.from_numpy(ww) 
                            start_index+=1
                    cluster_list=w
                    reshape_size_tuple=tuple(param_shape)
                    cluster_list=torch.tensor(cluster_list,dtype=torch.float)
                    cluster_list=cluster_list.reshape(reshape_size_tuple)
                    params.data=cluster_list.data.cuda()

      for module_name, module in model2.named_children():
                    if module_name  in layer_name: 
                        for basic_block_name, basic_block in module.named_children():
                                    for sub_block_name, sub_block in basic_block.named_children():
                                            for  params in sub_block.parameters():
                                                param_shape=list(params.size())
                                                w=params.reshape(-1,B)
                                                for i in range(len(w)):
                                                        ww=np.array(cluster_centers[kmeans.labels_[start_index]])
                                                        w[i]=torch.from_numpy(ww) 
                                                        start_index+=1
                                                cluster_list=w
                                                reshape_size_tuple=tuple(param_shape)
                                                cluster_list=torch.tensor(cluster_list,dtype=torch.float)
                                                cluster_list=cluster_list.reshape(reshape_size_tuple)
                                                params.data=cluster_list.data.cuda()
    return model2

def block_weight_clustering_per_section_compressing(model,layer_name,cluster_numebr,kmeans_centers,B):
    model2 = copy.deepcopy(model)
    with torch.no_grad():
      model2.cpu()
      for module_name, module in model2.named_children():
                    if layer_name in module_name: #layer
                        weights=np.array([])
                        for basic_block_name, basic_block in module.named_children():
                                    for sub_block_name, sub_block in basic_block.named_children():
                                        for params in sub_block.parameters():
                                                w=params.reshape(-1,B).numpy()
                                                if len(weights)==0:
                                                    weights=w
                                                else:
                                                    weights = np.concatenate((weights,w ))

                        weights = weights.astype('double')
                        kmeans = KMeans(n_clusters=cluster_numebr, init='k-means++', max_iter=100, n_init=64, random_state=0)
                        kmeans.fit(weights)
                        cluster_centers=torch.from_numpy(kmeans.cluster_centers_)
                        kmeans_centers.append(kmeans.cluster_centers_)
                        start_index=0
                        for basic_block_name, basic_block in module.named_children():
                                    for sub_block_name, sub_block in basic_block.named_children():
                                        for params in sub_block.parameters():
                                            param_shape=list(params.size())
                                            w=params.reshape(-1,B)
                                            for i in range(len(w)):
                                                ww=[]
                                                ww.append(w[i].numpy())
                                                predict=kmeans.predict(ww)
                                                ww=np.full((B),predict[0]) 
                                                w[i]=torch.from_numpy(ww)
                                            cluster_list=w
                                            reshape_size_tuple=tuple(param_shape)
                                            cluster_list=torch.tensor(cluster_list,dtype=torch.float)
                                            cluster_list=cluster_list.reshape(reshape_size_tuple)
                                            params.data=cluster_list.data.cuda()
    return model2




######################################################################################################################################################
'''
 global weight clustering
'''
def global_weight_clustering(model,cluster_numebr, B=1):
  model.to('cpu')
  weights=np.array([])
  skip_keywords = ("embeddings", "word_embeddings", "position_embeddings", "token_type_embeddings")
  with torch.no_grad():
    for name, params in model.named_parameters():
            
            if any(k in name for k in skip_keywords):
                continue
            if params.numel() % B != 0 or params.numel() < B:
                continue
            w = params.reshape(-1, B).numpy()
            if len(weights) == 0:
                weights = w
            else:
                weights = np.concatenate((weights, w))
    print('weight_clustering', weights.shape)
    weights = weights.astype('double')
    kmeans = KMeans(n_clusters=cluster_numebr, init='k-means++', max_iter=5, n_init='auto', random_state=0)
    kmeans.fit(weights)
    cluster_centers = torch.from_numpy(kmeans.cluster_centers_)
    start_index = 0
    for name, params in model.named_parameters():
                
                if any(k in name for k in skip_keywords):
                  continue

                if params.numel() % B != 0 or params.numel() < B:
                    continue
                param_shape = list(params.size())
                w = params.reshape(-1, B)
                for i in range(len(w)):
                        ww = np.array(cluster_centers[kmeans.labels_[start_index]])
                        w[i] = torch.from_numpy(ww)
                        start_index += 1
                cluster_list = w
                reshape_size_tuple = tuple(param_shape)
                cluster_list = torch.tensor(cluster_list, dtype=torch.float)
                cluster_list = cluster_list.reshape(reshape_size_tuple)
                params.data = cluster_list.data.cuda()
  return model, cluster_centers, kmeans.labels_


'''' layer wise weight clustering
'''
def layer_block_weight_clustering(model,n_clusters,B=1):
  uncompressed_size=0
  compressed_size=0
  model.to('cpu')
  skip_keywords = ("embeddings", "word_embeddings", "position_embeddings", "token_type_embeddings")
  bit_width = math.ceil(math.log2(n_clusters))
  print('layer wise weight clustering',n_clusters )
  with torch.no_grad():
    count=0
    for name, params in model.named_parameters():

      if any(k in name for k in skip_keywords):
                continue
      
      param_shape=list(params.size()) 
      flat_weights = params.reshape(-1)
      print(count, name.ljust(80), param_shape,  'uncompressed size: ',  (flat_weights.shape[0] *32)/ (8*1024 *1024) ,'compressed size: ', (flat_weights.shape[0] *bit_width)/ (8*1024 *1024 ))
      if flat_weights.shape[0] < n_clusters or flat_weights.numel()%B !=0 or flat_weights.numel() < B:
            print('can not be clustered: the layer size is very small')
            uncompressed_size+= (flat_weights.shape[0] *32)/ (8*1024 *1024)
            compressed_size+= (flat_weights.shape[0] *32)/ (8*1024 *1024)
            continue
      uncompressed_size+= (flat_weights.shape[0] *32)/ (8*1024 *1024)
      compressed_size+= (flat_weights.shape[0] *bit_width)/ (8*1024 *1024)+ (n_clusters * B *32)/ (8*1024 *1024)
      
      
      weights=params.reshape(-1,B)
      kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(weights)
      cluster_centers=torch.from_numpy(kmeans.cluster_centers_)
      count+=1
      cluster_list=[]
    #   for i in range(0,len(kmeans.labels_)):
    #        cluster_list.append(cluster_centers[kmeans.labels_[i]].view(1) )

      start_index=0
      for i in range(len(weights)):
            ww=np.array(cluster_centers[kmeans.labels_[start_index]])
            weights[i]=torch.from_numpy(ww)
            start_index+=1

      cluster_list=weights
      reshape_size_tuple=tuple(param_shape)
      cluster_list=torch.tensor(cluster_list,dtype=torch.float)
      cluster_list=cluster_list.reshape(reshape_size_tuple)
      params.data=cluster_list.data.cuda()
    model.cuda()
  return model, (uncompressed_size, compressed_size, uncompressed_size/compressed_size)


###################################################################################################################################################

def weight_clustering_compress(model,cluster_numebr, B=1):
  model.to('cpu')
  weights=np.array([])
  with torch.no_grad():
    count=0
    for name, params in model.named_parameters():
            w=params.reshape(-1,B).numpy()
            if len(weights)==0:
                    weights=w
            else:
                weights = np.concatenate((weights,w ))
            # print(name)
    print('weight_clustering', weights.shape)
    weights = weights.astype('double')
    kmeans = KMeans(n_clusters=cluster_numebr, init='k-means++', max_iter=100, n_init='auto', random_state=0)
    kmeans.fit(weights)
  return kmeans.cluster_centers_ , kmeans.labels_

def weight_clustering_decompress(model,codebook, labels,cluster_numebr, B=1):
    model.to('cpu')
    cluster_centers=torch.from_numpy(codebook)
    cluster_list=[]
    start_index=0
    with torch.no_grad():
        for name, params in model.named_parameters():
                    param_shape=list(params.size())
                    w=params.reshape(-1,B)
    #                 print(name, w.shape)
                    for i in range(len(w)):
                            ww=np.array(cluster_centers[labels[start_index]])
                            w[i]=torch.from_numpy(ww)
                            start_index+=1
                    cluster_list=w
                    reshape_size_tuple=tuple(param_shape)
                    cluster_list=torch.tensor(cluster_list,dtype=torch.float)
                    cluster_list=cluster_list.reshape(reshape_size_tuple)
                    params.data=cluster_list.data.cuda()
        return model

def pack_bits(arr, num_values):
    """
    Pack an array of integers in range [0, num_values-1] into minimal bits.
    
    arr         : 1D array-like of integers
    num_values  : number of unique values (max label + 1)
    """
    arr = np.array(arr, dtype=np.uint64)
    if arr.max() >= num_values:
        raise ValueError("Array contains values >= num_values")

    bit_width = math.ceil(math.log2(num_values))
    bitstring = 0
    bits_in_buffer = 0
    packed_bytes = []

    for val in arr:
        bitstring |= (int(val) << bits_in_buffer)
        bits_in_buffer += bit_width
        while bits_in_buffer >= 8:
            packed_bytes.append(bitstring & 0xFF)
            bitstring >>= 8
            bits_in_buffer -= 8

    if bits_in_buffer > 0:
        packed_bytes.append(bitstring & 0xFF)

    return np.array(packed_bytes, dtype=np.uint8), bit_width

def unpack_bits(packed, length, num_values):
    """
    Unpack bit-packed data into original integers.
    
    packed      : packed bytes (np.uint8 array)
    length      : number of integers to unpack
    num_values  : number of unique values (max label + 1)
    """
    bit_width = math.ceil(math.log2(num_values))
    values = []
    bitstring = 0
    bits_in_buffer = 0
    packed_iter = iter(packed)

    for _ in range(length):
        while bits_in_buffer < bit_width:
            try:
                byte = next(packed_iter)
            except StopIteration:
                raise ValueError("Not enough data to unpack")
            bitstring |= (byte << bits_in_buffer)
            bits_in_buffer += 8

        val = bitstring & ((1 << bit_width) - 1)
        bitstring >>= bit_width
        bits_in_buffer -= bit_width
        values.append(val)

    return np.array(values, dtype=np.uint64)

###############################################################################################################################################


def block_weight_clustering_layer_transfomer(model,clusters,blocks,num_codebooks, transformer_block_name=".0.",MiniKMeans=False ):
  model.to('cpu')
  total_uncompressed_size=0
  total_compressed_size=0
  
  # for testing
  # model.cuda()
  # return model, (1, 1, 1/1)

  with torch.no_grad():
    count=-1

    # clusters=[8,16,16] # [0] biase, [1] key value, [2] dense
    # blocks=[1,2,2]
    # num_codebooks=[1,40,40]


    for name, params in model.named_parameters():
      count+=1

      # layer norm and biases
      K=clusters[0]
      B=blocks[0]    
      N=num_codebooks[0]
    
      if transformer_block_name not in name:
            continue 

      # multi head self attention
      if  ".attention.self.query.weight" in name or  ".attention.self.key.weight" in name or  ".attention.self.value.weight" in name or ".attention.output.dense.weight" in name:
           K=clusters[1]
           B=blocks[1]
           N=num_codebooks[1]

      # feed forward
      if "intermediate.dense.weight" in name or  ".output.dense.weight" in name:
           K=clusters[2]
           B=blocks[2]
           N=num_codebooks[2]

      param_shape=list(params.size()) 
      flat_weights = params.reshape(-1)
      
      bit_width = math.ceil(math.log2(K))
      uncompressed_size= (flat_weights.shape[0] *32)/ (8*1024 *1024)
      compressed_size=  ((flat_weights.shape[0] *bit_width)/B+ K * B * N *32 )/ (8*1024 *1024 )

    #   print(f"{count}, name={name.ljust(70)} K={K}  B={B}  N={N}  param_shape={param_shape}  uncompressed size={round(uncompressed_size,4)} \
    #         compressed size={round(compressed_size,4)}")
      
      
      if flat_weights.shape[0] < K or flat_weights.numel()%B !=0 or flat_weights.numel() < B:
            total_uncompressed_size+=uncompressed_size
            total_compressed_size+=uncompressed_size
            print(f' Error: name={name.ljust(20)} =',flat_weights.shape[0], K, B, N, end=', ')
            continue

  
      
      if N>1:
          weights = params.reshape(-1, B)
          total_len = weights.shape[0]
          chunk_len = total_len // N


          if chunk_len < K :
               print(f' Error: name={name.ljust(20)} =',chunk_len, K, B, N, end=', ')
               total_uncompressed_size+=uncompressed_size
               total_compressed_size+=uncompressed_size
               continue

          clustered_chunks = []

          for i in range(N):
              start = i * chunk_len
              end = (i + 1) * chunk_len if i < N - 1 else total_len
              chunk = weights[start:end]

              if chunk.shape[0] < K or chunk.numel()%B !=0 or chunk.numel() < B:
                print(' Error 2: size=',chunk.shape[0], K, B, N, end=', ')
                clustered_chunks.append(chunk)
                continue
              
              if MiniKMeans:
                   kmeans = MiniBatchKMeans(n_clusters=K, random_state=i, batch_size=1024, max_iter=100, n_init=3).fit(chunk)
              else:
                    kmeans = KMeans(n_clusters=K, random_state=i).fit(chunk)
              
              
              centers = torch.from_numpy(kmeans.cluster_centers_).float()
              clustered_chunk = torch.stack([centers[label] for label in kmeans.labels_])
              clustered_chunks.append(clustered_chunk)

              # Print number of members in each cluster
              # counts = np.bincount(kmeans.labels_)
              # for cluster_id, count in enumerate(counts):
              #     print(f"                              Cluster {cluster_id}: {count} members", centers[cluster_id] )

          clustered_weights = torch.cat(clustered_chunks, dim=0)

          cluster_list = clustered_weights
          reshape_size_tuple = tuple(param_shape)
          cluster_list = cluster_list.reshape(reshape_size_tuple)
          params.data = cluster_list.data.float().cuda() 
      
      else:
          weights=params.reshape(-1,B)
          
          if MiniKMeans:
            kmeans = MiniBatchKMeans(n_clusters=K, random_state=0, batch_size=1024, max_iter=100, n_init=3).fit(weights)
          else:
            kmeans = KMeans(n_clusters=K, random_state=0).fit(weights)

          cluster_centers=torch.from_numpy(kmeans.cluster_centers_)
          cluster_list=[]
        #   for i in range(0,len(kmeans.labels_)):
        #        cluster_list.append(cluster_centers[kmeans.labels_[i]].view(1) )

          start_index=0
          for i in range(len(weights)):
                ww=np.array(cluster_centers[kmeans.labels_[start_index]])
                weights[i]=torch.from_numpy(ww)
                start_index+=1

          cluster_list=weights
          reshape_size_tuple=tuple(param_shape)
          cluster_list=torch.tensor(cluster_list,dtype=torch.float)
          cluster_list=cluster_list.reshape(reshape_size_tuple)
          params.data=cluster_list.data.cuda()

      total_uncompressed_size+=uncompressed_size
      total_compressed_size+=compressed_size

  model.cuda()
  # print(transformer_block_name, 'Total uncompressed size: ', round(total_uncompressed_size,4), 'Total compressed size: ', round(total_compressed_size,4) , 'Compression ratio: ', round(total_uncompressed_size/total_compressed_size,4) )
  return model, (total_uncompressed_size, total_compressed_size, total_uncompressed_size/total_compressed_size)



###########################################################################################################################################



def layers_weight_parameters(model):
  all_weights=[]
  with torch.no_grad():
    for name, params in model.named_parameters():
      all_weights.append([params.reshape(-1).cpu().numpy().shape[0],list(params.size()),name ])
  return all_weights

def layer_block_weight_clustering_numbers(model,n_clusters,B=1,layer_numbers=[],exclude_layers=[]):
  model.to('cpu')
  bit_width = math.ceil(math.log2(n_clusters))
  print('layer wise weight clustering',n_clusters )
  with torch.no_grad():
    count=-1
    
    for name, params in model.named_parameters():
      count+=1
      
      if len(exclude_layers)>0:
        if count in exclude_layers:
                continue
      
      if len(layer_numbers)>0:
        if count not in layer_numbers:
                continue

      param_shape=list(params.size()) 
      flat_weights = params.reshape(-1)
      print(count, name.ljust(70), param_shape,  'uncompressed size: ', round( (flat_weights.shape[0] *32)/ (8*1024 *1024),3) ,'compressed size: ',round( ((flat_weights.shape[0] *bit_width)/B)/ (8*1024 *1024 ),3))
      if flat_weights.shape[0] < n_clusters or flat_weights.numel()%B !=0 or flat_weights.numel() < B:
            print('can not be clustered: the layer size is very small')
            continue
      
      
      
      weights=params.reshape(-1,B)
      # print(weights.shape)
      # kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init='auto', random_state=0).fit(weights)
      kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(weights)
      cluster_centers=torch.from_numpy(kmeans.cluster_centers_)
      cluster_list=[]
    #   for i in range(0,len(kmeans.labels_)):
    #        cluster_list.append(cluster_centers[kmeans.labels_[i]].view(1) )

      start_index=0
      for i in range(len(weights)):
            ww=np.array(cluster_centers[kmeans.labels_[start_index]])
            weights[i]=torch.from_numpy(ww)
            start_index+=1

      cluster_list=weights
      reshape_size_tuple=tuple(param_shape)
      cluster_list=torch.tensor(cluster_list,dtype=torch.float)
      cluster_list=cluster_list.reshape(reshape_size_tuple)
      params.data=cluster_list.data.cuda()
    model.cuda()
  return model


def layer_block_weight_clustering_names(model,n_clusters,B=1):
  model.to('cpu')

  print('layer wise weight clustering',n_clusters )
  with torch.no_grad():
    count=-1
    
    for name, params in model.named_parameters():
      count+=1

      n_clusters=8
      B=1
      num_codebooks=0

      if "embeddings.word_embeddings.weight" in name or "embeddings.position_embeddings.weight" in name or "embeddings.token_type_embeddings.weight" in name:
           n_clusters=8
           num_codebooks=20
           B=1
           continue
      
      if "intermediate.dense.weight" in name or  ".output.dense.weight" in name:
           n_clusters=32
           num_codebooks=40
           B=2
      
      if  ".attention.self.query.weight" in name or  ".attention.self.key.weight" in name or  ".attention.self.value.weight" in name or ".attention.output.dense.weight" in name:
           n_clusters=32
           num_codebooks=40
           B=2
      
      param_shape=list(params.size()) 
      flat_weights = params.reshape(-1)
      
      bit_width = math.ceil(math.log2(n_clusters))
      print(count, name.ljust(70),n_clusters , param_shape,  'uncompressed size: ', round( (flat_weights.shape[0] *32)/ (8*1024 *1024),3) ,'compressed size: ',round( ((flat_weights.shape[0] *bit_width)/B)/ (8*1024 *1024 ),3))
      

      if round( (flat_weights.shape[0] *32)/ (8*1024 *1024),3) <1:
            print('---------> the layer size is very small')
            continue
      
      if flat_weights.shape[0] < n_clusters or flat_weights.numel()%B !=0 or flat_weights.numel() < B:
            print('can not be clustered: the layer size is very small')
            continue
      
      if num_codebooks>0:
          print('using multiple codebooks', num_codebooks)
          weights = params.reshape(-1, B)
          total_len = weights.shape[0]
          chunk_len = total_len // num_codebooks

          clustered_chunks = []

          for i in range(num_codebooks):
              start = i * chunk_len
              end = (i + 1) * chunk_len if i < num_codebooks - 1 else total_len
              chunk = weights[start:end]

              kmeans = KMeans(n_clusters=n_clusters, random_state=i).fit(chunk)
              centers = torch.from_numpy(kmeans.cluster_centers_).float()
              clustered_chunk = torch.stack([centers[label] for label in kmeans.labels_])
              clustered_chunks.append(clustered_chunk)

              # # Print number of members in each cluster
              # counts = np.bincount(kmeans.labels_)
              # for cluster_id, count in enumerate(counts):
              #     print(f"                              Cluster {cluster_id}: {count} members", centers[cluster_id] )

          clustered_weights = torch.cat(clustered_chunks, dim=0)

          cluster_list = clustered_weights
          reshape_size_tuple = tuple(param_shape)
          cluster_list = cluster_list.reshape(reshape_size_tuple)
          params.data = cluster_list.data.float().cuda() 
      
      else:
          weights=params.reshape(-1,B)
          kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(weights)
          cluster_centers=torch.from_numpy(kmeans.cluster_centers_)
          cluster_list=[]
        #   for i in range(0,len(kmeans.labels_)):
        #        cluster_list.append(cluster_centers[kmeans.labels_[i]].view(1) )

          start_index=0
          for i in range(len(weights)):
                ww=np.array(cluster_centers[kmeans.labels_[start_index]])
                weights[i]=torch.from_numpy(ww)
                start_index+=1

          cluster_list=weights
          reshape_size_tuple=tuple(param_shape)
          cluster_list=torch.tensor(cluster_list,dtype=torch.float)
          cluster_list=cluster_list.reshape(reshape_size_tuple)
          params.data=cluster_list.data.cuda()

  model.cuda()
  return model


def layer_block_weight_clustering_layer_name2(model, n_clusters=[], B=[]):
    model.to('cpu')
    n_clusters=[32,32,32,32,32,32,32,32,32,32,32,32]
    B=[1,1,1,1,1,1,1,1,1,1,1,1]
    N=[20,20,20,20,20,20,20,20,20,20,20,20]
    all_centers = []
    all_labels = []

    def cluster_and_replace(layer, layer_idx, n_cluster, B_val):
        print(f"\n🔍 Clustering weights for Layer {layer_idx}")
        weights = np.array([])

        with torch.no_grad():
            for name, param in layer.named_parameters():
                if param.numel() % B_val != 0 or param.numel() < B_val:
                    continue
                w = param.reshape(-1, B_val).numpy()
                weights = w if len(weights) == 0 else np.concatenate((weights, w))

        
        weights = weights.astype('double')
        kmeans = KMeans(n_clusters=n_clusters[0], random_state=0).fit(weights)

        print(f"Layer {layer_idx} weight shape: {weights.shape}", n_cluster, B_val)
        
        cluster_centers = torch.from_numpy(kmeans.cluster_centers_)
        labels = kmeans.labels_

        start_index = 0
        with torch.no_grad():
            for name, param in layer.named_parameters():
                if param.numel() % B_val != 0 or param.numel() < B_val:
                    continue
                param_shape = list(param.size())
                w = param.reshape(-1, B_val)
                for i in range(len(w)):
                    center = cluster_centers[labels[start_index]]
                    w[i] = center
                    start_index += 1
                clustered_tensor = w.reshape(param_shape)
                param.data = clustered_tensor.float().cuda()

        return cluster_centers, labels

    # Parallel execution across layers
    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = []
        for layer_idx, layer in enumerate(model.bert.encoder.layer):
            futures.append(executor.submit(
                cluster_and_replace,
                layer, layer_idx, n_clusters[layer_idx], B[layer_idx]
            ))

        for future in futures:
            centers, labels = future.result()
            all_centers.append(centers)
            all_labels.append(labels)

    model.cuda()
    return model


def layer_block_weight_clustering_layer_name(model, n_clusters=None, B=None, num_codebooks=None):
    model.to("cpu")
    if n_clusters is None:
        n_clusters = [32] * 12
    if B is None:
        B = [1] * 12
    if num_codebooks is None:
        num_codebooks = [10] * 12  # default: no multiple codebooks

    all_centers = []
    all_labels = []

    def cluster_and_replace(layer, layer_idx, n_cluster, B_val, n_codebooks):
        print(f"\n🔍 Clustering weights for Layer {layer_idx} | n_clusters={n_cluster}, codebooks={n_codebooks}, Block={B_val}")
        weights = np.array([])

        with torch.no_grad():
            for name, param in layer.named_parameters():
                if param.numel() % B_val != 0 or param.numel() < B_val:
                    continue
                w = param.reshape(-1, B_val).numpy()
                weights = w if len(weights) == 0 else np.concatenate((weights, w))

        print(f"  ➡️ Layer {layer_idx} weight shape: {weights.shape}")

        if n_codebooks > 1:
            # Split weights into codebooks
            total_len = weights.shape[0]
            chunk_len = total_len // n_codebooks
            clustered_chunks = []
            all_labels_local = []
            all_centers_local = []

            for i in range(n_codebooks):
                start = i * chunk_len
                end = (i + 1) * chunk_len if i < n_codebooks - 1 else total_len
                chunk = weights[start:end]

                kmeans = KMeans(n_clusters=n_cluster, random_state=i).fit(chunk)
                centers = torch.from_numpy(kmeans.cluster_centers_).float()
                labels = kmeans.labels_

                clustered_chunk = torch.stack([centers[label] for label in labels])
                clustered_chunks.append(clustered_chunk)
                all_centers_local.append(centers)
                all_labels_local.append(labels)

            clustered_weights = torch.cat(clustered_chunks, dim=0)
            cluster_centers = all_centers_local
            labels = all_labels_local

        else:
            # Single codebook (regular KMeans)
            weights = weights.astype("double")
            kmeans = KMeans(n_clusters=n_cluster, random_state=0).fit(weights)
            cluster_centers = torch.from_numpy(kmeans.cluster_centers_)
            labels = kmeans.labels_

            clustered_weights = torch.stack([cluster_centers[label] for label in labels])

        # Re-assign weights back to model
        start_index = 0
        with torch.no_grad():
            for name, param in layer.named_parameters():
                if param.numel() % B_val != 0 or param.numel() < B_val:
                    continue
                param_shape = list(param.size())
                w = clustered_weights[start_index:start_index + np.prod(param_shape) // B_val]
                w = w.reshape(param_shape)
                param.data = w.float().cuda()
                start_index += np.prod(param_shape) // B_val

        return cluster_centers, labels

    # Parallel execution across layers
    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = []
        for layer_idx, layer in enumerate(model.bert.encoder.layer):
            futures.append(executor.submit(
                cluster_and_replace,
                layer,
                layer_idx,
                n_clusters[layer_idx],
                B[layer_idx],
                num_codebooks[layer_idx]
            ))

        for future in futures:
            centers, labels = future.result()
            all_centers.append(centers)
            all_labels.append(labels)

    model.cuda()
    return model


def layer_weight_clustering(model,n_clusters,B=1):
  model.to('cpu')
  bit_width = math.ceil(math.log2(n_clusters))
  print('layer wise weight clustering',n_clusters )
  with torch.no_grad():
    count=0
    for name, params in model.named_parameters():
      
      param_shape=list(params.size()) 
      weights=params.reshape(-1,B)
      print(count, name.ljust(80), param_shape,  'uncompressed size: ',  (weights.shape[0] *32)/ (8*1024 *1024) ,'compressed size: ', (weights.shape[0] *bit_width)/ (8*1024 *1024 ))
      if weights.shape[0] < n_clusters:
            print('can not be clustered: the layer size is very small')
            continue
      kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(weights)
      cluster_centers=torch.from_numpy(kmeans.cluster_centers_)
      count+=1
      cluster_list=[]
      for i in range(0,len(kmeans.labels_)):
           cluster_list.append(cluster_centers[kmeans.labels_[i]].view(1) )

      reshape_size_tuple=tuple(param_shape)
      cluster_list=torch.tensor(cluster_list,dtype=torch.float)
      cluster_list=cluster_list.reshape(reshape_size_tuple)
      params.data=cluster_list.data.cuda()
    model.cuda()
  return model


def layer_weight_clustering2(model,n_clusters,B=1):

  model.to('cpu')
  bit_width = math.ceil(math.log2(n_clusters))
  print('layer wise weight clustering',n_clusters )
  with torch.no_grad():
    count=0
    for name, params in model.named_parameters():
      
      if "word_embeddings.weight" in name: 
            # Special case for embedding layer
            vocab_size, hidden_dim = params.shape  # [30522, 768]
            print(f"Clustering embeddings: vocab={vocab_size}, dim={hidden_dim}")
            # Each token embedding is a point in hidden_dim space
            weights = params.detach().cpu().numpy()  # [30522, 768]
            # weights=params.reshape(-1,2)
            # Cluster across tokens
            kmeans = KMeans(n_clusters=8, random_state=0).fit(weights)
            # Replace each token embedding with its cluster center
            clustered_embeddings = torch.from_numpy(kmeans.cluster_centers_[kmeans.labels_])
            params.data = clustered_embeddings.to(params.device).type_as(params.data)
            continue

      param_shape=list(params.size()) 
      weights=params.reshape(-1,B)
      print(count, name, param_shape, weights.shape, 'uncompressed size: ',  (weights.shape[0] *32)/ (8*1024 *1024) ,'compressed size: ', (weights.shape[0] *bit_width)/ (8*1024 *1024 ))
      if weights.shape[0] < n_clusters:
            print('can not be clustered: the layer size is very small')
            continue
      kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(weights)
      cluster_centers=torch.from_numpy(kmeans.cluster_centers_)
      count+=1
      cluster_list=[]
      for i in range(0,len(kmeans.labels_)):
           cluster_list.append(cluster_centers[kmeans.labels_[i]].view(1) )

      reshape_size_tuple=tuple(param_shape)
      cluster_list=torch.tensor(cluster_list,dtype=torch.float)
      cluster_list=cluster_list.reshape(reshape_size_tuple)
      params.data=cluster_list.data.cuda()
    model.cuda()
  return model