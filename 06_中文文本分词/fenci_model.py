import tensorflow as tf
import numpy as np

class Model():
    """
    分词模型中使用双向RNN模型进行处理
    """
    def __init__(self, batch_size=32, n_layer=2, hidden_size=64, is_training=True, n_words=6000, learning_rate=0.001):
        """
        初始化类
        """
        self.batch_size = batch_size 
        self.is_training = is_training
        self.n_layer=n_layer
        self.hidden_size=hidden_size
        self.n_words = n_words
        self.learning_rate =  learning_rate
        self.build_model() 
        #self.init_sess() 
    def build_model(self):
        """
        构建计算图
        """
        self.graph = tf.Graph() 
        cell_fn = tf.nn.rnn_cell.BasicLSTMCell
        with self.graph.as_default():
            # 输入文本序列
            self.inputs = tf.placeholder(tf.int32, 
                                         [self.batch_size, None],
                                         name="inputs") 
            # 文本标注序列，用0-3数字表示
            self.target = tf.placeholder(tf.int32, 
                                         [self.batch_size, None],
                                         name="target") 
            # 序列不等长在计算loss函数部分需要置0，也就是乘以mask
            self.mask = tf.placeholder(tf.float32, 
                                         [self.batch_size, None],
                                         name="mask")   
            # 序列长度，在解码部分可能用到
            self.seqlen = tf.placeholder(tf.int32, 
                                         [self.batch_size],
                                         name="seqlen")
            # 定义双向RNN单元
            fw_cell_list = [cell_fn(self.hidden_size) for itr in range(self.n_layer)]
            bw_cell_list = [cell_fn(self.hidden_size) for itr in range(self.n_layer)]
            fw_cell = tf.nn.rnn_cell.MultiRNNCell(fw_cell_list)
            bw_cell = tf.nn.rnn_cell.MultiRNNCell(bw_cell_list)
            # Embedding
            emb_w = tf.get_variable("emb_w", [self.n_words, self.hidden_size])
            emb_input = tf.nn.embedding_lookup(emb_w, self.inputs)
            # 双向RNN，self.seqlen用于不同长度序列解码
            (fw_output, bw_output), _ = tf.nn.bidirectional_dynamic_rnn(
                fw_cell,
                bw_cell,
                emb_input,
                self.seqlen, 
                dtype=tf.float32
            )
            # 将两个网络输出进行连接
            outputs = tf.concat([fw_output, bw_output], 2) 
            self.logit = tf.layers.dense(outputs, 4) 
            self.pred_id = tf.argmax(self.logit, 2)
            # 计算loss函数
            self.loss = tf.contrib.seq2seq.sequence_loss(
                self.logit,
                self.target, 
                self.mask
            )
            # 计算准确率
            self.correct_prediction = tf.equal(tf.argmax(self.pred_id,1), tf.argmax(self.target,1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
            #self.loss = tf.reduce_mean(self.loss)
            # 优化
            self.step = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            self.all_var = tf.global_variables() 
            self.init = tf.global_variables_initializer() 
            self.saver = tf.train.Saver() 
    def init_sess(self, model="model"):
        """
        初始化会话
        """
        self.sess = tf.Session(graph=self.graph)
        self.sess.run(self.init)
        restore = tf.train.latest_checkpoint(model)
        if restore != None:
            self.saver.restore(self.sess, restore)
    def save(self, model="model/a", itr=0):
        self.saver.save(self.sess, model, itr) 
    def train(self, a1, a2, a3, a4):
        #print(a1.shape, a2.shape, a3.shape, a4.shape)
        ls, _, acc = self.sess.run([self.loss, self.step, self.accuracy], feed_dict={
            self.inputs:a1, 
            self.target:a2, 
            self.mask:a3, 
            self.seqlen:a4
        })
        return ls, acc 
    def valid(self, a1):
        data = np.array([a1])
        idx = self.sess.run(self.pred_id, feed_dict={
            self.inputs:data,
            self.seqlen:np.ones([1])*len(a1)
        })
        return idx[0]