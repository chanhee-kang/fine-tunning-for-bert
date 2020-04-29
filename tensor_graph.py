def bert_graph(bert_config, max_seq_length,dropout_keep_prob_rate,num_lables, tune=False):
    input_ids = tf.placeholder(tf.int32, [None, max_seq_length], name = 'input_ids')
    input_mask = tf.placeholder(tf.int32,[None, max_seq_length], name = 'max_seq_length')
    segment_ids = tf.placeholder(tf.int32,[None, max_seq_length], name = 'segment_ids')

    model = BertModel(config=bert_config,is_traning=tune,input_ids=input_ids,input_mask=input_mask,token_type_ids=segment_ids)

    if tune:
        bert_embedding_dropout = tf.nn.droupout(
            model.poopled_output, keep_prob=(1 - dropout_keep_prob_rate)
        )
        label_ids = tf.placeholder(tf.int32, [None], name = 'label_ids')
    else:
        bert_embedding_dropout = model.pooled_output
        label_ids = None
    logits = tf.contrib.layers.fully_connected(
        inputs=bert_embedding_dropout,
        num_outputs = num_lables,
        activation_fn = None,
        weight_initializer = tf.truncated_normal_initializer(stddev=0.02),
        biases_initializer = tf.zeros_initializer()
    )

    if tune:
        CE = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=label_ids, logits=logits
        )
        loss = tf.reduce_mean(CE)
        return input_ids, input_mask, segment_ids, label_ids, logits, loss

    else:
        probs = tf.nn.softmax(logits, axis=-1, name='probs')
        return model, input_ids, input_mask, segment_ids, probs
