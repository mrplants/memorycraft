import tensorflow as tf

def kmeans(dataset: tf.data.Dataset,
              k:int,
              batch_size:int,
              max_iterations:int=300,
              min_inertia:float=0.0001,
              init_centroids:tf.Tensor=None,
              mean:float=0,
              std:float=1) -> tf.Tensor:
    """ Fits the data using kmeans clustering with Tensorflow.
    
    Args:
      data: Data iterator compatible with Tensorflow (e.g., tf.data.Dataset)
        Produces tf.Tensors with shape (n_features,).
      k: The number of clusters.
      batch_size: The number of samples per training batch.
      max_iterations: maximum number of iterations to converge.
      min_inertis: minimum change in inertia required to stop iterating.
      init_centroids: If not provided, the first k points will be chosen as the
        starting centroids.
      mean: used for normalizing the samples
      std: used for normalizing the samples
      
    Returns:  The fitted centroids
    """
    # Initialize centroids with the first few points
    if init_centroids == None:
        centroids = tf.cast(tf.stack([f for f in dataset.take(k)]), tf.float32)
        while centroids.shape[0] < k:
            more = tf.cast(tf.stack([f for f in dataset.take(k-centroids.shape[0])]), tf.float32)
            centroids = tf.concat([centroids, more], axis=0)
        centroids = tf.Variable((centroids - mean) / std)
    else:
        centroids = tf.Variable(init_centroids)
    next_round_centroids = tf.Variable(centroids)
    # Keep track of inertia to determine convergence
    last_inertia = None
    # Keep track of total iterations in case max_iterations has been provided
    num_iterations = 0
    while True:
        # Keep track of counts of points assigned to each centroid
        # (for calculating online average)
        centroid_counts = tf.zeros(k, dtype=tf.float32)
        inertia = 0
        for n_batch, batch in enumerate(dataset.batch(batch_size)):
            batch = tf.cast(batch, tf.float32)
            batch = (batch - mean) / std
            # Assign samples each to its closest centroid
            diffs = centroids - tf.expand_dims(batch, axis=1)
            # diffs axis 0 is the sample, axis 1 is the centroid
            norms = tf.norm(diffs, axis=2)
            # Assign centroids to each sample and save the total inertia
            assigned_centroids = tf.math.argmin(norms, axis=1)
            batch_inertia = tf.gather_nd(norms, tf.reshape(assigned_centroids, (-1,1)), batch_dims=1)
            inertia += tf.reduce_sum(batch_inertia / batch.shape[0]).numpy().item()
            # Update counts for each centroid
            batch_centroid_counts = tf.math.bincount(assigned_centroids, minlength=k, axis=0, dtype=tf.float32)
            new_centroid_counts = centroid_counts + batch_centroid_counts
            # Update the next round centroids as a running mean
            # Start by calculating the mean of this batch's assigned points
            c_update_mean = []
            for c_index in range(k):
                c_assigned_indexes = tf.where(assigned_centroids == c_index)
                if c_assigned_indexes.shape[0] != 0:
                    c_update_mean.append(tf.reduce_mean(tf.gather_nd(batch, c_assigned_indexes), axis=0))
                else:
                    c_update_mean.append(centroids[c_index])
            c_update_mean = tf.stack(c_update_mean)
            update_fractions = tf.math.divide_no_nan(centroid_counts, new_centroid_counts)
            next_round_centroids.assign(next_round_centroids * update_fractions[:,tf.newaxis] + c_update_mean * (1-update_fractions)[:,tf.newaxis])
            # Update running counts for each centroid
            centroid_counts = new_centroid_counts
        # Determine convergence criteria.  Break if converged.
        inertia /= (n_batch+1)
        if last_inertia != None and tf.math.abs(last_inertia - inertia) < min_inertia:
            break
        last_inertia = inertia
        num_iterations += 1
        if num_iterations >= max_iterations:
            break
        print(inertia)
        # Update centroids from next_round_centroids
        centroids.assign(next_round_centroids)
    print(centroid_counts.numpy().astype(int).tolist())
    return tf.identity(centroids) * std + mean