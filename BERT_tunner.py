from models.bert.modeling import BertModel, BertConfig
from models.bert.optimization import create_optimizer
from models.bert.tokenization import FullTokenizer, convert_to_unicode

class BERTTunner(Tuner):

    def __init__(self, train_corpus_fname, test_corpus_fname, voca_fname,pretrain_model_fname,bertconfig_fname
    ,model_save_path, max_seq_length = 128, warmup_proportion = 0.1, batch_size = 32, learning_rate=2e-5, num_lables=2):
        super().__init__(
            train_corpus_fname=train_corpus_fname, tokenized_train_corpus_fname = train_corpus_fname +'tokenized',
            test_corpus_fname=test_corpus_fname, batch_size = batch_size,
            tokenized_test_corpus_fname = test_corpus_fname + '.tokenized',
            model_name='bert', vocab_fname = voca_fname,
            model_save_path = model_save_path
        )
    
    #Configuration
    config = BertConfig.from_json_file(bertconfig_fname)
    self.pretrain_model_fname = pretrain_model_fname
    self.max_seq_length = max_seq_length
    self.batch_size = batch_size
    self.learning_rate = learning_rate
    self.num_lables = 2
    self.PAD_INDEX = 0
    self.CLS_TOKEN = '[CLS]'
    self.SEP_TOKEN = '[SEP]'
    self.num_train_steps = \
        (int((len(self.train_data)-1) / slef.batch_size) + 1) * self.num_epochs
    self.num_warmpu_steps = int(self.num_train_steps * warmup_proportion)
    self.eval_every = int(self.num_train_steps / self. num_epoches)
    self.traning = tf.placeholder(tf.bool)
    
    #build train graph
    self.input_ids, self.input_mask, self.segment_ids, self.label_ids, self.logits, self.loss = make_bert_graph(
        config, max_seq_length, self.dropout_keep_prob_rate, num_lables, tune = True
    )
