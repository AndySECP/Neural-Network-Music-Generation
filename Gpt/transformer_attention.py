from typing import Optional, Callable, Tuple

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer

from transformer_layers import WeightNormDense as Dense, LayerNorm, ApplyAttentionMask

class AttentionQKV(Model):
    """
    Computes attention based on provided similarity metric.
    """

    def __init__(self) -> None:
        super().__init__()
        self.apply_mask = ApplyAttentionMask()

    def call(self, queries, keys, values, mask=None):
        """Fast scaled dot product attention.

            :param queries: Tensor with shape [batch_size, heads (optional), n_queries, depth_k]
            :param keys:    Tensor with shape [batch_size, heads (optional), n_keyval, depth_k]
            :param values:  Tensor with shape [batch_size, heads (optional), n_keyval, depth_v]
            :param mask:    Tensor with shape [batch_size, n_queries, n_queries]

            :return: output: Tensor with shape [batch_size, heads (optional), n_queries, depth_v]
        """
        # n_queries corresponds to the sequence length on the query side
        # n_keyval corresponds to the sequence length on the key side (and value, as they are one and the same)
        # depth_k is the size of the projection that the key / query comparison is performed on.
        # depth_v is the size of the projection of the value projection. In a setting with one head, it is usually the dimension (dim) of the Transformer.
        # heads corresponds to the number of heads the attention is performed on.
         
        ###
        # The output is computed as a weighted sum of the values, where the weight assigned to each value is 
        # computed by a compatibility function of the query with the corresponding key.
        ###
        
        # PART 1: Implementation of Attention QKV
        
        Q = queries
        K = keys
        V = values

        # As defined is the Attention is all you need paper: https://arxiv.org/pdf/1706.03762.pdf
        key_dim = tf.cast(tf.shape(keys)[-1], tf.float32)
        similarity = tf.matmul(Q, K, transpose_b=True) / tf.sqrt(key_dim) # Compute the similarity according to the QKV formula

        masked_similarity = self.apply_mask(similarity, mask=mask) # We give you the mask to apply so that it is correct, you do not need to modify this.
        weights = tf.nn.softmax(masked_similarity) # Turn the similarity into a normalized output (softmax)
        output = tf.matmul(weights, V) # Obtain the output

        return output, weights


class MultiHeadProjection(Model):

    def __init__(self, n_heads) -> None:
        """Map the multi-headed attention across the map

        Arguments:
            similarity_metric {[type]} -- The metric that should be used for the similarity
            n_heads {int} -- The number of heads in the attention map

        """

        super().__init__()
        self.attention_map = AttentionQKV()
        self.n_heads = n_heads

    def build(self, input_shape):
        for shape in input_shape:
            assert shape[-1] % self.n_heads == 0, 'Shape of feature input must be divisible by n_heads'

    def call(self, inputs, mask=None):
        """Fast multi-head attention.

        :param queries: Tensor with shape [batch_size, n_queries, depth_k]
        :param keys:    Tensor with shape [batch_size, n_keyval, depth_k]
        :param values:  Tensor with shape [batch_size, n_keyval, depth_v]

        :return: output: Tensor with shape [batch_size, n_queries, depth_v]
        """
        queries, keys, values = inputs

        # Split each of the projection into its heads, by adding a new dimension
        # You must implement _split_heads, and _combine_heads
        queries_split = self._split_heads(queries)
        keys_split = self._split_heads(keys)
        values_split = self._split_heads(values)

        # Apply the attention map
        attention_output_split, _ = self.attention_map(queries_split, keys_split, values_split, mask=mask)

        # Re-combine the heads together, and return the output.
        output = self._combine_heads(attention_output_split)
        return output

    def _split_heads(self, tensor):
        tensor.shape.assert_has_rank(3)
        # PART 2: Implementation of the Multi-head attention.
       
        batch_size, tensorlen = tf.shape(tensor)[0], tf.shape(tensor)[1]
        feature_size = tensor.shape.as_list()[2]

        new_feature_size = int(feature_size / self.n_heads) # Compute what the feature size per head is.
        t1 = []
        for i in range(self.n_heads):
            t1.append(tf.slice(tensor, begin = [0,0, i*new_feature_size], size = [batch_size, tensorlen, new_feature_size]))
        tensor = tf.stack(t1)
        # n_heads, batch_size, tensorlen, new_feature_size
        
        # Transpose the matrix so the outer-dimensions are the batch-size and the number of heads
        tensor = tf.transpose(tensor, perm=[1, 0, 2, 3])
        # batch_size, n_heads, tensorlen, feature_size
        
        return tensor


    def _combine_heads(self, tensor):
        tensor.shape.assert_has_rank(4)

        # PART 2: Implementation of the Multi-head attention.
        # we are given the output from all the heads, and you must combine them back into 1 rank-3 matrix

        # Transpose back compared to the split, so that the outer dimensions are batch_size and sequence_length again
        tensor = tf.transpose(tensor, perm=[0, 2, 1, 3])

        batch_size, tensorlen = tf.shape(tensor)[0], tf.shape(tensor)[1]
        feature_size = tensor.shape.as_list()[-1]

        new_feature_size = feature_size * self.n_heads # What is the new feature size, if we combine all the heads
        
        tensor = tf.reshape(tensor,[batch_size, tensorlen, new_feature_size]) # Reshape the Tensor to remove the heads dimension and come back to a Rank-3 tensor
        
        return tensor


class MultiHeadAttention(Model):
    """
    Fast multi-head attention. Based on the Attention is All You Need paper.

    https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(self, n_heads) -> None:
        super().__init__()

        self.n_heads = n_heads
        self.attention_layer = MultiHeadProjection(n_heads)

    def build(self, input_shapes):
        query_antecedent_shape, memory_antecedent_shape = input_shapes
        self.qa_channels = query_antecedent_shape[-1]
        self.ma_channels = memory_antecedent_shape[-1]
        assert self.qa_channels % self.n_heads == 0 and self.ma_channels % self.n_heads == 0, \
            'Feature size must be divisible by n_heads'
        assert self.qa_channels == self.ma_channels, 'Cannot combine tensors with different shapes'

        self.query_layer = Dense(self.qa_channels, use_bias=False)
        self.key_layer = Dense(self.qa_channels, use_bias=False)
        self.value_layer = Dense(self.ma_channels, use_bias=False)

        self.output_layer = Dense(self.qa_channels, use_bias=False)


    def call(self, inputs, mask=None):
        """Fast multi-head self attention.

            :param inputs: tuple of (query_antecedent, memory_antecedent)
                query_antecedent -> tensor w/ shape [batch_size, n_queries, channels]
                memory_antecedent -> tensor w/ shape [batch_size, n_keyval, channels]
        """
        assert isinstance(inputs, tuple) or isinstance(inputs, list) and len(inputs) == 2, \
            'Must pass query and memory'
        query_antecedent, memory_antecedent = inputs
        q = self.query_layer(query_antecedent)
        k = self.key_layer(memory_antecedent)
        v = self.value_layer(memory_antecedent)

        attention_output = self.attention_layer((q, k, v), mask=mask)
        output = self.output_layer(attention_output)
        return output