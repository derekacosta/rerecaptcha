import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import time
class NN_config(object):
    def __init__(self,num_seqs=1000,num_steps=10,num_units=128,num_classes = 8,\
                num_layers = 1,embedding_size=100,vocab_size = 10000,\
                use_embeddings=False,embedding_init=None):
        self.num_seqs   = num_seqs
        self.num_steps  = num_steps
        self.num_units  = num_units
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.use_embeddings = use_embeddings
        self.embedding_init = embedding_init

class CALC_config(object):
    def __init__(self,batch_size=64,num_epoches = 20,learning_rate = 1.0e-3, \
                 keep_prob=0.5,show_every_steps = 10,save_every_steps=100):
        self.batch_size     = batch_size
        self.num_epoches    = num_epoches
        self.learning_rate  = learning_rate
        self.keep_prob      = keep_prob
        self.show_every_steps = show_every_steps
        self.save_every_steps = save_every_steps

class ClassifierRNN(object):
    def __init__(self, nn_config, calc_config):
        # assign revalent parameters
        self.num_seqs   = nn_config.num_seqs
        self.num_steps  = nn_config.num_steps
        self.num_units  = nn_config.num_units
        self.num_layers = nn_config.num_layers
        self.num_classes    = nn_config.num_classes
        self.embedding_size = nn_config.embedding_size
        self.vocab_size     = nn_config.vocab_size
        self.use_embeddings = nn_config.use_embeddings
        self.embedding_init = nn_config.embedding_init
        # assign calc ravalant values
        self.batch_size     = calc_config.batch_size
        self.num_epoches    = calc_config.num_epoches
        self.learning_rate  = calc_config.learning_rate
        self.train_keep_prob= calc_config.keep_prob
        self.show_every_steps = calc_config.show_every_steps
        self.save_every_steps = calc_config.save_every_steps
        # create networks models
        tf.reset_default_graph()
        self.build_inputs()
        self.build_rnns()
        self.build_loss()
        self.build_optimizer()
        self.saver = tf.train.Saver()

    def build_inputs(self):
        with tf.name_scope('inputs'):
            self.inputs = tf.placeholder(tf.int32, shape=[None,self.num_seqs],\
                                                                name='inputs')
            self.targets = tf.placeholder(tf.int32, shape=[None, self.num_classes],\
                                                                name='classes')
            self.keep_prob = tf.placeholder(tf.float32,name='keep_prob')
            self.embedding_ph = tf.placeholder(tf.float32, name='embedding_ph')

            if self.use_embeddings == False:
                self.embeddings = tf.Variable(tf.random_uniform([self.vocab_size,\
                                self.embedding_size],-0.1,0.1),name='embedding_flase')
                self.rnn_inputs = tf.nn.embedding_lookup(self.embeddings,self.inputs)
            else:
                embeddings = tf.Variable(tf.constant(0.0,shape=[self.vocab_size,self.embedding_size]),\
                                                            trainable=False,name='embeddings_true')
                self.embeddings = embeddings.assign(self.embedding_ph)
                self.rnn_inputs = tf.nn.embedding_lookup(self.embeddings,self.inputs)
                print('self.rnn_inputs.shape:',self.rnn_inputs.shape)

    def build_rnns(self):
        def get_a_cell(num_units,keep_prob):
            rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=num_units)
            drop = tf.contrib.rnn.DropoutWrapper(rnn_cell, output_keep_prob=keep_prob)
            return drop
        with tf.name_scope('rnns'):
            self.cell = tf.contrib.rnn.MultiRNNCell([get_a_cell(self.num_units,self.keep_prob) for _ in range(self.num_layers)])
            self.initial_state = self.cell.zero_state(self.batch_size,tf.float32)
            self.outputs, self.final_state = tf.nn.dynamic_rnn(self.cell,tf.cast(self.rnn_inputs,tf.float32),\
              initial_state = self.initial_state  )
            print('rnn_outputs',self.outputs.shape)

    def build_loss(self):
        with tf.name_scope('loss'):
            self.logits = tf.contrib.layers.fully_connected(inputs = tf.reduce_mean(self.outputs, axis=1), \
                                            num_outputs = self.num_classes, activation_fn = None)
            print('self.logits.shape:',self.logits.shape)
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits,\
                                        labels = self.targets))
            print('self.cost.shape',self.cost.shape)
            self.predictions = self.logits
            self.correct_predictions = tf.equal(tf.argmax(self.predictions, axis=1), tf.argmax(self.targets, axis=1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_predictions,tf.float32))
            print(self.cost.shape)
            print(self.correct_predictions.shape)

    def build_optimizer(self):
        with tf.name_scope('optimizer'):
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)

    def random_batches(self,data,shuffle=True):
        data = np.array(data)
        data_size = len(data)
        num_batches_per_epoch = int(data_size/self.batch_size)

        #del data
        for epoch in range(self.num_epoches):
            if shuffle :
                shuffle_index = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_index]
            else:
                shuffled_data = data
            for batch_num in range(num_batches_per_epoch):
                start = batch_num * self.batch_size
                end   = min(start + self.batch_size,data_size)
                yield shuffled_data[start:end]

    def fit(self,data,restart=False):
        if restart :
            self.load_model()
        else:
            self.session = tf.Session()
            self.session.run(tf.global_variables_initializer())
        with self.session as sess:
            step = 0
            accuracy_list = []
            # model saving
            save_path = os.path.abspath(os.path.join(os.path.curdir, 'models'))
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            plt.ion()
            #new_state = sess.run(self.initial_state)
            new_state = sess.run(self.initial_state)
            batches = self.random_batches(data)
            for batch in batches:
                x,y = zip(*batch)
                x = np.array(x)
                y = np.array(y)
                print(len(x),len(y),step)
                step += 1
                start = time.time()
                if self.use_embeddings == False:
                    feed = {self.inputs :x,
                        self.targets:y,
                        self.keep_prob : self.train_keep_prob,
                        self.initial_state: new_state}
                else:
                    feed = {self.inputs :x,
                        self.targets:y,
                        self.keep_prob : self.train_keep_prob,
                        self.initial_state: new_state,
                        self.embedding_ph: self.embedding_init}
                batch_loss, new_state, batch_accuracy , _ = sess.run([self.cost,self.final_state,\
                                                            self.accuracy, self.optimizer],feed_dict = feed)
                end = time.time()
                accuracy_list.append(batch_accuracy)
                # control the print lines
                if step%self.show_every_steps == 0:
                    print('steps/epoch:{}/{}...'.format(step,self.num_epoches),
                        'loss:{:.4f}...'.format(batch_loss),
                        '{:.4f} sec/batch'.format((end - start)),
                        'batch_Accuracy:{:.4f}...'.format(batch_accuracy)
                        )
                    plt.plot(accuracy_list)
                    plt.pause(0.5)
                if step%self.save_every_steps == 0:
                    self.saver.save(sess,os.path.join(save_path, 'model') ,global_step = step)
            self.saver.save(sess, os.path.join(save_path, 'model'), global_step=step)

    def load_model(self, start_path=None):
        if start_path == None:
            model_path = os.path.abspath(os.path.join(os.path.curdir,"models"))
            ckpt = tf.train.get_checkpoint_state(model_path)
            path = ckpt.model_checkpoint_path
            print("this is the start path of model:",path)
            self.session = tf.Session()
            self.saver.restore(self.session, path)
            print("Restored model parameters is complete!")

        else:
            self.session = tf.Session()
            self.saver.restore(self.session,start_path)
            print("Restored model parameters is complete!")

    def predict_accuracy(self,data,test=True):
        # loading_model
        self.load_model()
        sess = self.session
        iterations = 0
        accuracy_list = []
        predictions = []
        epoch_temp = self.num_epoches
        self.num_epoches = 1
        batches = self.random_batches(data,shuffle=False)
        for batch in batches:
            iterations += 1
            x_inputs, y_inputs = zip(*batch)
            x_inputs = np.array(x_inputs)
            y_inputs = np.array(y_inputs)
            if self.use_embeddings == False:
                feed = {self.inputs: x_inputs,
                        self.targets: y_inputs,
                        self.keep_prob: 1.0}
            else:
                feed = {self.inputs: x_inputs,
                        self.targets: y_inputs,
                        self.keep_prob: 1.0,
                        self.embedding_ph: self.embedding_init}
            to_train = [self.cost, self.final_state, self.predictions,self.accuracy]
            batch_loss,new_state,batch_pred,batch_accuracy = sess.run(to_train, feed_dict = feed)
            accuracy_list.append(np.mean(batch_accuracy))
            predictions.append(batch_pred)
            print('The trainning step is {0}'.format(iterations),\
                 'trainning_accuracy: {:.3f}'.format(accuracy_list[-1]))

        accuracy = np.mean(accuracy_list)
        predictions = [list(pred) for pred in predictions]
        predictions = [p for pred in predictions for p in pred]
        predictions = np.array(predictions)
        self.num_epoches = epoch_temp
        if test :
            return predictions, accuracy
        else:
            return accuracy

    def predict(self, data):
        #print(type(data))
        # load_model
        self.load_model()
        sess = self.session
        iterations = 0
        predictionss = []
        epoch_temp = self.num_epoches
        self.num_epoches = 1
        batches = self.random_batches(data)
        for batch in batches:
            x_inputs = batch
            if self.use_embeddings == False:
                feed = {self.inputs : x_inputs,
                        self.keep_prob:1.0}
            else:
                feed = {self.inputs : x_inputs,
                        self.keep_prob:1.0,
                        self.embedding_ph: self.embedding_init}
            batch_pred = sess.run([self.predictions],feed_dict=feed)
            predictionss.append(batch_pred)

        predictions = [list(pred) for pred in predictionss]
        predictions = [p for pred in predictions for p in pred]
        predictions = np.array(predictions)
        return predictions
