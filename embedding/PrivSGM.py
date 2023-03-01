import tensorflow as tf
import numpy as np
import argparse
import networkx as nx
from utils import graph_util
import math
from sklearn import metrics
from sklearn.externals import joblib

parser = argparse.ArgumentParser()
parser.add_argument('--embedding_dim', default=128)
parser.add_argument('--d_batch_size', default=128)
parser.add_argument('--proximity', default='second-order', help='first-order or second-order')
parser.add_argument('--edge_sampling', default='numpy', help='numpy or atlas or uniform')
parser.add_argument('--node_sampling', default='numpy', help='numpy or atlas or uniform')
parser.add_argument('--d_epoch', default=15)  # done
parser.add_argument('--g_epoch', default=5)  # done
parser.add_argument('--g_batch_size', default=128)  # done
parser.add_argument('--lr_gen', default=0.001)  # done
parser.add_argument('--lr_dis', default=0.001)  # done
parser.add_argument('--g_extraAddLayer', default=False)  # done
parser.add_argument('--leak', default=0.2, help='parameter of activation func leaky_relu')  # done
parser.add_argument('--d_weight', default=1)  # done
parser.add_argument('--d_alpha', default=0.00001)  # done
parser.add_argument('--K', default=5)
parser.add_argument('--low_bound', default=0, help='low bound for exp func')
parser.add_argument('--upper_bound', default=5, help='upper bound for exp func')
parser.add_argument('--sig', default=1.0)
parser.add_argument('--graclip_thres', default=0.5)

# args = parser.parse_args()

class discriminator:
    def __init__(self, args, edge_distribution):
        self.edge_distribution = edge_distribution
        self.u_i = tf.placeholder(name='u_i', dtype=tf.int32, shape=[None])
        self.u_j = tf.placeholder(name='u_j', dtype=tf.int32, shape=[None])
        self.label = tf.placeholder(name='label', dtype=tf.float32, shape=[None])
        # new addition
        self.edgeSamProb = tf.placeholder(name='edge_sampleProb', dtype=tf.float32,
                                          shape=[args.d_batch_size * (args.K + 1)])
        self.noise_embedding = tf.placeholder(name='node_fake_embedding', dtype=tf.float32,
                                              shape=[None, args.embedding_dim])
        self.embedding = tf.get_variable('target_embedding', [args.num_of_nodes, args.embedding_dim],
                                         initializer=tf.random_uniform_initializer(minval=-1., maxval=1.))
        self.u_i_embedding = tf.matmul(tf.one_hot(self.u_i, depth=args.num_of_nodes), self.embedding)
        if args.proximity == 'first-order':
            self.u_j_embedding = tf.matmul(tf.one_hot(self.u_j, depth=args.num_of_nodes), self.embedding)
        elif args.proximity == 'second-order':
            self.context_embedding = tf.get_variable('context_embedding', [args.num_of_nodes, args.embedding_dim],
                                                     initializer=tf.random_uniform_initializer(minval=-1., maxval=1.))
            self.u_j_embedding = tf.matmul(tf.one_hot(self.u_j, depth=args.num_of_nodes), self.context_embedding)
        self.inner_product = tf.reduce_sum(self.u_i_embedding * self.u_j_embedding, axis=1)
        # --------------------------------------------------------------------
        if args.is_ProbFuse:
            self.edgeProb_term = args.d_alpha * tf.pow(tf.sqrt(self.edgeSamProb), self.label - 1)

            if args.is_Normalize_ExpClip:
                # ---------- Adv based objective perturbation ---------
                print('performing PrivSGM algorithm')
                self.test_expclip = self.expclip(-self.label * self.inner_product,
                                                                  args.low_bound, args.upper_bound)
                log_sigmoid = tf.log(tf.div(1.0, 1 + self.expclip(-self.label * self.inner_product,
                                                                  args.low_bound, args.upper_bound)))
                self.sgm_loss = -tf.reduce_mean(self.edgeProb_term * log_sigmoid)
            else:
                self.sgm_loss = -tf.reduce_mean(self.edgeProb_term * tf.log_sigmoid(self.label * self.inner_product))

            if args.is_NewSGM_ObjPer:
                # --------------- Objective Perturbation -------------
                print('performing NewSGM_ObjPer algorithm')
                self.Laplace_noise_for_NewSGM_ObjPer = \
                    tf.placeholder(name='Noise_for_NewSGM_ObjPer', dtype=tf.float32, shape=[None, args.embedding_dim])
                # noise_term = tf.reduce_sum(noise_embedding * self.u_i_embedding, axis=1)
                noise_term = tf.reduce_sum(self.Laplace_noise_for_NewSGM_ObjPer * self.u_i_embedding, axis=1)
                self.sgm_loss = -tf.reduce_mean(self.edgeProb_term * tf.log_sigmoid(self.label * self.inner_product)
                                                + noise_term)
            else:
                self.sgm_loss = -tf.reduce_mean(self.edgeProb_term * tf.log_sigmoid(self.label * self.inner_product))
        else:
            self.sgm_loss = -tf.reduce_mean(tf.log_sigmoid(self.label * self.inner_product))

        # ----------- adv term ------------
        # self.u_i_embedding * self.noise_embedding note that it may do not work! check it again
        self.adv_inner_product = tf.reduce_sum(self.u_i_embedding * self.noise_embedding, axis=1)
        self.adv_loss = tf.reduce_mean(tf.log(1 - tf.log_sigmoid(self.adv_inner_product)))
        if args.is_advTerm:
            # with adv and weight
            self.loss = self.sgm_loss + args.d_weight * self.adv_loss
        else:
            self.loss = self.sgm_loss
        # ---------------------------------
        self.learning_rate = tf.placeholder(name='learning_rate', dtype=tf.float32)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=args.lr_dis)
        # self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=args.lr_dis)
        if args.is_GradientClip:
            self.d_vars = [self.embedding, self.context_embedding]
            self.grads_and_vars = self.optimizer.compute_gradients(self.loss, self.d_vars)
            # noisy_grads_and_vars = []  # noisy version
            # flag = 0
            for i, (g, v) in enumerate(self.grads_and_vars):  # for each pair
                # for gv in self.grads_and_vars:  # for each pair
                #     g = g[0]  # get the gradient, type in loop one: Tensor
                if g is not None and v is not None:
                    if "target_embedding:0" in v.name:
                        g = tf.clip_by_norm(g, args.graclip_thres)
                        g = g + self.noise(g)
                        self.grads_and_vars[i] = (g, v)
                    else:
                        self.grads_and_vars[i] = (g, v)
            self.train_op = self.optimizer.apply_gradients(self.grads_and_vars)
        else:
            if args.is_Normalize_ExpClip or args.is_NewSGM_ObjPer:
                print('performing batch normalization for PirvSGM or NewSGM_ObjPer')
                self.u_j_embedding = tf.layers.batch_normalization(self.u_j_embedding)
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    self.train_op = self.optimizer.minimize(self.loss)
            else:
                self.train_op = self.optimizer.minimize(self.loss)

    def expclip(self, x, a=None, b=None):
        # default scaling constants to match tanh corner shape
        self._c_tanh = 2 / (np.e * np.e + 1)  # == 1 - np.tanh(1) ~= 0.24
        self._c_softclip = np.log(2) / self._c_tanh
        self._c_expclip = 1 / (2 * self._c_tanh)
        """
        Exponential soft clipping, with parameterized corner sharpness.
        Simpler functional form but 3rd, 5th, ... and subequent odd derivatives are discontinuous at 0
        """
        self.c = self._c_expclip
        if a is not None and b is not None:
            self.c /= (b - a) / 2

        self.v = tf.clip_by_value(x, a, b)

        if a is not None:
            self.v = self.v + tf.exp(-self.c * np.abs(x - a)) / (2 * self.c)
        if b is not None:
            self.v = self.v - tf.exp(-self.c * np.abs(x - b)) / (2 * self.c)
        # print('执行了')
        return self.v

    def noise(self, tensor):
        '''add noise to tensor'''
        '''
        Deep Learning with Differential Privacy
        \sigma \geq c_2 \frac{q \sqrt{T \log (1 / \delta)}}{\varepsilon}
        '''
        c_2 = 1
        q = np.max(self.edge_distribution)  # take max edge probability
        delta = 0.00001  # 10^-5
        sigma = c_2 * q * tf.sqrt(args.d_epoch * tf.log(1/delta))/args.epsilon
        s = tensor.get_shape().as_list()  # get shape of the tensor
        rt = tf.random_normal(s, mean=0.0, stddev=sigma * args.graclip_thres)
        t = tf.add(tensor, rt)
        return t

class generator:
    def __init__(self, n_node, node_emd_init):
        self.n_node = n_node
        self.emd_dim = args.embedding_dim
        self.node_emd_init = node_emd_init

        # with tf.variable_scope('generator'):
        if node_emd_init:
            self.node_embedding_matrix = tf.get_variable(name='gen_node_embedding',
                                                         shape=self.node_emd_init.shape,
                                                         initializer=tf.constant_initializer(self.node_emd_init),
                                                         trainable=True)
        else:
            self.node_embedding_matrix = tf.get_variable(name='gen_node_embedding',
                                                         shape=[self.n_node, self.emd_dim],
                                                         initializer=tf.contrib.layers.xavier_initializer(
                                                             uniform=False),
                                                         trainable=True)

        self.gen_w_1 = tf.get_variable(name='gen_w',
                                       shape=[self.emd_dim, self.emd_dim],
                                       initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                       trainable=True)
        self.gen_b_1 = tf.get_variable(name='gen_b',
                                       shape=[self.emd_dim],
                                       initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                       trainable=True)
        self.gen_w_2 = tf.get_variable(name='gen_w_2',
                                       shape=[self.emd_dim, self.emd_dim],
                                       initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                       trainable=True)
        self.gen_b_2 = tf.get_variable(name='gen_b_2',
                                       shape=[self.emd_dim],
                                       initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                       trainable=True)
        # self.bias_vector = tf.Variable(tf.zeros([self.n_node]))

        self.node_ids = tf.placeholder(tf.int32, shape=[None])

        self.noise_embedding = tf.placeholder(tf.float32, shape=[None, self.emd_dim])  # laplace noise matrix

        self.pos_node_embedding = tf.placeholder(tf.float32, shape=[None, self.emd_dim])  # pos samples from dis

        self.node_embedding = tf.nn.embedding_lookup(self.node_embedding_matrix, self.node_ids)

        self.node_fake_embedding = self.generate_node(self.node_embedding, self.noise_embedding)

        self.score = tf.reduce_sum(tf.multiply(self.pos_node_embedding, self.node_fake_embedding), axis=1)

        # self.neg_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits( \
        #     labels=tf.ones_like(self.score) * (1.0 - config.label_smooth), logits=self.score)) \
        #                 + config.lambda_gen * (tf.nn.l2_loss(self.node_embedding) + tf.nn.l2_loss(self.gen_w_1))
        self.loss = tf.reduce_mean(tf.log(1 - tf.log_sigmoid(self.score)))
        # self.neg_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.score),
        #                                                                        logits=self.score))

        # self.loss = self.neg_loss

        optimizer = tf.train.AdamOptimizer(args.lr_gen)
        # optimizer = tf.train.GradientDescentOptimizer(args.lr_gen)
        self.g_updates = optimizer.minimize(self.loss)

    def generate_node(self, node_embedding, noise_embedding):
        if args.g_extraAddLayer:
            output_1 = tf.nn.leaky_relu(tf.matmul(noise_embedding, self.gen_w_1) + self.gen_b_1)
            output_2 = tf.reshape(node_embedding, [-1, self.emd_dim])
            output = output_1 + output_2
        else:
            output = tf.nn.leaky_relu(tf.matmul(noise_embedding, self.gen_w_1) + self.gen_b_1)

        return output

class AliasSampling:
    # Reference: https://en.wikipedia.org/wiki/Alias_method
    def __init__(self, prob):
        self.n = len(prob)
        self.U = np.array(prob) * self.n
        self.K = [i for i in range(len(prob))]
        overfull, underfull = [], []
        for i, U_i in enumerate(self.U):
            if U_i > 1:
                overfull.append(i)
            elif U_i < 1:
                underfull.append(i)
        while len(overfull) and len(underfull):
            i, j = overfull.pop(), underfull.pop()
            self.K[j] = i
            self.U[i] = self.U[i] - (1 - self.U[j])
            if self.U[i] > 1:
                overfull.append(i)
            elif self.U[i] < 1:
                underfull.append(i)

    def sampling(self, n=1):
        x = np.random.rand(n)
        i = np.floor(self.n * x)
        y = self.n * x - i
        i = i.astype(np.int32)
        res = [i[k] if y[k] < self.U[i[k]] else self.K[i[k]] for k in range(n)]
        if n == 1:
            return res[0]
        else:
            return res

class prepare_data:
    def __init__(self, graph_file=None):
        self.g = graph_file
        self.num_of_nodes = len(self.g.nodes())
        self.num_of_edges = len(self.g.edges())
        self.edges_raw = self.g.edges(data=True)
        self.nodes_raw = self.g.nodes(data=True)

        # self.edge_distribution = np.array([attr['weight'] for _, _, attr in self.edges_raw], dtype=np.float32)
        self.edge_distribution = []
        for node_a, node_b, attr in self.edges_raw:
            node_a_outDegree = self.g.out_degree(node_a)
            self.edge_distribution.append(node_a_outDegree)
        self.edge_distribution /= np.sum(self.edge_distribution)
        self.edge_sampling = AliasSampling(prob=self.edge_distribution)
        # self.node_negative_distribution = np.power(
        #     np.array([self.g.degree(node, weight='weight') for node, _ in self.nodes_raw], dtype=np.float32),
        #     1 / self.num_of_nodes)
        self.node_negative_distribution = \
            np.array([1 / self.num_of_nodes for node, _ in self.nodes_raw], dtype=np.float32)
        self.node_negative_distribution /= np.sum(self.node_negative_distribution)
        self.node_sampling = AliasSampling(prob=self.node_negative_distribution)

        self.node_index = {}
        self.node_index_reversed = {}
        for index, (node, _) in enumerate(self.nodes_raw):
            self.node_index[node] = index  # 对节点进行重新编号
            self.node_index_reversed[index] = node
        self.edges = [(self.node_index[u], self.node_index[v]) for u, v, _ in self.edges_raw]
        # print('success')

    def prepare_data_for_dis(self, sess, generator):
        # print('error')
        if args.edge_sampling == 'numpy':
            edge_batch_index = np.random.choice(self.num_of_edges, size=args.d_batch_size, p=self.edge_distribution)
        elif args.edge_sampling == 'atlas':
            edge_batch_index = self.edge_sampling.sampling(args.d_batch_size)
        elif args.edge_sampling == 'uniform':
            edge_batch_index = np.random.randint(0, self.num_of_edges, size=args.d_batch_size)
        u_i = []
        u_j = []
        label = []
        edgeProb = []  # new addition
        for edge_index in edge_batch_index:
            edge = self.edges[edge_index]
            if self.g.__class__ == nx.Graph:
                if np.random.rand() > 0.5:  # important: second-order proximity is for directed edge
                    edge = (edge[1], edge[0])
            u_i.append(edge[0])
            u_j.append(edge[1])
            label.append(1)
            edgeProb.append(self.edge_distribution[edge_index])  # new addition
            for i in range(args.K):
                while True:
                    if args.node_sampling == 'numpy':
                        negative_node = np.random.choice(self.num_of_nodes, p=self.node_negative_distribution)
                    elif args.node_sampling == 'atlas':
                        negative_node = self.node_sampling.sampling()
                        # print('执行了这一步')
                    elif args.node_sampling == 'uniform':
                        negative_node = np.random.randint(0, self.num_of_nodes)
                    if not self.g.has_edge(self.node_index_reversed[negative_node], self.node_index_reversed[edge[0]]):
                        break
                u_i.append(edge[0])
                u_j.append(negative_node)
                label.append(-1)
                edgeProb.append(self.edge_distribution[edge_index])  # new addition
        # ------------------- generate fake node ---------------------
        noise_embedding = np.random.normal(0.0, args.sig, (args.d_batch_size * (args.K + 1), args.embedding_dim))
        # node_fake_embedding = noise_embedding
        if args.is_advTerm:
            node_fake_embedding = sess.run(generator.node_fake_embedding,
                                           feed_dict={generator.node_ids: u_i,
                                                      generator.noise_embedding: noise_embedding})
        else:
            node_fake_embedding = noise_embedding


        return u_i, u_j, label, edgeProb, node_fake_embedding


    def prepare_data_for_gen(self, sess, model, index, node_list):
        node_ids = []
        if isinstance(node_list, list) is False:
            node_list = list(node_list)
        # test = node_list[index * args.g_batch_size: (index + 1) * args.g_batch_size]
        # print(test)
        for node_id in node_list[index * args.g_batch_size: (index + 1) * args.g_batch_size]:
            # n_sample = min(self.graph[node_id][1], config.n_sample_max)
            #             n_sample = config.n_sample_max
            #             for i in range(n_sample):
            node_ids.append(node_id)

        # noise_embedding = np.random.normal(0.0, args.sig, (len(node_ids), args.embedding_dim))
        # noise_embedding = Laplace_mechanism(length, L1_sensitivity, args.epsilon)
        if args.is_Normalize_ExpClip:  # if privacy is true, distribution is laplace.
            L2_sensitivity = args.d_alpha * args.K * (1+args.upper_bound) / (args.leak * args.d_weight * self.num_of_nodes)
            noise_embedding = np.random.laplace(loc=0, scale=L2_sensitivity / args.epsilon,
                                                size=(len(node_ids), args.embedding_dim))
            # noise_embedding = np.random.normal(0.0, L2_sensitivity / args.epsilon,
            #                                    (len(node_ids), args.embedding_dim))
        else:  # if no privacy, distribution is laplace.
            noise_embedding = np.random.normal(0.0, args.sig, (len(node_ids), args.embedding_dim))

        # tf.reset_default_graph()
        # dis = discriminator(args)
        pos_node_embedding = sess.run(model.u_i_embedding,
                                      feed_dict={model.u_i: node_ids})
        # with tf.Session() as sess:
        #     sess.run(tf.global_variables_initializer())
        #     dis_node_embedding = sess.run(dis.u_i_embedding,
        #                                    feed_dict={dis.u_i: np.array(node_ids)})

        return node_ids, noise_embedding, pos_node_embedding

class trainModel:
    def __init__(self, inf_display, graph, test_pos=None, test_neg=None, node_label=None):
        self.inf_display = inf_display
        self.node_label = node_label
        self.test_pos = test_pos
        self.test_neg = test_neg
        self.graph = graph
        self.data_loader = prepare_data(self.graph)
        args.num_of_nodes = self.data_loader.num_of_nodes
        self.model = discriminator(args, self.data_loader.edge_distribution)
        self.generator = generator(len(self.graph.nodes()), None)

    def train_dis(self, test_task=None, test_ratios=None, output_filename=None):
        best_auc = []
        # test_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0., 0.9]
        with tf.Session() as sess:
            print(args)
            print('batches\tloss\tsampling time\ttraining_time\tdatetime')
            # if d_epoch == 0:
            for indep_run_time in range(args.indep_run_times):
                sess.run(tf.global_variables_initializer())  # note that this initilization's location
                flag_auc = 0
                flag_micro = [0] * len(test_ratios)
                flag_macro = [0] * len(test_ratios)
                for epoch in range(args.n_epoch):
                    # initial_embedding = sess.run(self.discriminator.embedding)
                    for d_epoch in range(args.d_epoch):  # just a test
                        # learning_rate = args.learning_rate
                        sampling_time, training_time = 0, 0
                        for b in range(math.floor(args.num_of_nodes / args.d_batch_size)):
                            u_i, u_j, label, edgeProb, node_fake_embedding = \
                                self.data_loader.prepare_data_for_dis(sess, self.generator)

                            if args.is_NewSGM_ObjPer:
                                # print('performing NewSGM_ObjPer sessrun')
                                Laplace_Noise_matrix = Laplace_noise_for_NewSGM_ObjPer()
                                feed_dict = {self.model.u_i: u_i, self.model.u_j: u_j, self.model.label: label,
                                             self.model.edgeSamProb: edgeProb,
                                             self.model.Laplace_noise_for_NewSGM_ObjPer: Laplace_Noise_matrix}
                                _, loss = sess.run([self.model.train_op, self.model.loss], feed_dict=feed_dict)
                            else:
                                feed_dict = {self.model.u_i: u_i, self.model.u_j: u_j, self.model.label: label,
                                             self.model.edgeSamProb: edgeProb, self.model.noise_embedding: node_fake_embedding}
                                _, loss = sess.run([self.model.train_op, self.model.loss], feed_dict=feed_dict)

                            # print('loss value is', loss)

                        embedding = sess.run(self.model.embedding)
                        context_embedding = sess.run(self.model.context_embedding)
                        # print(self.inf_display[0], self.inf_display[1], loss, indep_run_time, epoch, d_epoch)
                        # --------------------------------------------------------------------------------
                        if test_task == 'lp':
                            embedding_mat = np.dot(embedding, embedding.T)
                            auc = CalcAUC(embedding_mat, self.test_pos, self.test_neg)
                            if auc > flag_auc:
                                flag_auc = auc
                            # if epoch == args.n_epoch - 1:
                            #     best_auc.append(flag_auc)
                            print(indep_run_time, epoch, d_epoch, auc)

                        gen_loss = 0.0
                        gen_neg_loss = 0.0
                        gen_cnt = 0
                    if args.is_advTerm:
                        for g_epoch in range(args.g_epoch):
                            gen_loss, gen_neg_loss, gen_cnt = self.train_gen(sess, gen_loss, gen_neg_loss, gen_cnt)

                if test_task == 'lp':
                    best_auc.append(flag_auc)

    def train_gen(self, sess, gen_loss, neg_loss, gen_cnt):
        # print('generator is running')
        node_list = self.graph.nodes()
        # np.random.shuffle(node_list)
        # sess = tf.Session()
        # sess.run(tf.global_variables_initializer())
        info = ''
        for index in range(math.floor(len(node_list) / args.g_batch_size)):
            node_ids, noise_embedding, pos_node_embedding = self.data_loader.prepare_data_for_gen(sess, self.model,
                                                                                                  index, node_list)
            _loss, _neg_loss = sess.run(
                [self.generator.g_updates, self.generator.loss],
                feed_dict={self.generator.node_ids: np.array(node_ids),
                           self.generator.noise_embedding: np.array(noise_embedding),
                           self.generator.pos_node_embedding: np.array(pos_node_embedding)})

        return (gen_loss, neg_loss, gen_cnt)

def Laplace_noise_for_NewSGM_ObjPer():
    L2_sensitivity = args.d_alpha * args.K / args.num_of_nodes
    noise_embedding = np.random.laplace(loc=0, scale=L2_sensitivity / args.epsilon,
                                        size=(args.d_batch_size * (args.K + 1), args.embedding_dim))
    # noise_embedding = tf.convert_to_tensor(noise_embedding, dtype=tf.float32)
    return noise_embedding

def Laplace_mechanism(length, L1_sensitivity, epsilon):
    noise = np.random.laplace(scale=L1_sensitivity / epsilon, size=length)
    return noise

def CalcAUC(sim, test_pos, test_neg):
    pos_scores = np.asarray(sim[test_pos[0], test_pos[1]]).squeeze()
    neg_scores = np.asarray(sim[test_neg[0], test_neg[1]]).squeeze()
    scores = np.concatenate([pos_scores, neg_scores])
    labels = np.hstack([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    fpr, tpr, _ = metrics.roc_curve(labels, scores, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    return auc

if __name__ == '__main__':
    test_task = 'lp'
    set_algo_name = 'PrivSGM'

    if set_algo_name == 'PrivSGM':
        parser.add_argument('--n_epoch', default=30)
        parser.add_argument('--indep_run_times', default=2)
        parser.add_argument('--epsilon', default=0.025)
        parser.add_argument('--is_ProbFuse', default=True)
        parser.add_argument('--is_advTerm', default=True)
        parser.add_argument('--is_NewSGM_ObjPer', default=False)
        parser.add_argument('--is_Normalize_ExpClip', default=True)
        parser.add_argument('--is_GradientClip', default=False)
        print('performing PrivSGM algorithm')

    args = parser.parse_args()

    if test_task == 'lp':
        set_dataset_names = ['lp_cora']
        set_split_name = 'train0.8_test0.2'

        epsilon_values = [0.025]
        for epsilon_value in epsilon_values:
            args.epsilon = epsilon_value
            set_epsilon_str = 'epsilon' + str(args.epsilon)
            set_learning_rate = 'step' + str(args.lr_dis)

            for set_dataset_name in set_dataset_names:
                tf.reset_default_graph()  # note that 函数用于清除默认图形堆栈并重置全局默认图形
                set_nepoch_name = 'nepoch' + str(args.n_epoch)
                # ------------------------------------------------------
                oriGraph_filename = '../data/' + set_dataset_name +'/train_1'
                train_filename = '../data/' + set_dataset_name + '/' + set_split_name + '/'
                output_filename = set_algo_name + '_' + set_dataset_name + '_' + set_split_name + '_' \
                                  + set_nepoch_name + '_' + set_epsilon_str + '_' + set_learning_rate
                isDirected = False
                # Load graph
                trainGraph = graph_util.loadGraphFromEdgeListTxt(oriGraph_filename, directed=isDirected)
                trainGraph = nx.adjacency_matrix(trainGraph)
                # ------------------------------------------------------
                train_pos = joblib.load(train_filename + 'train_pos.pkl')
                train_neg = joblib.load(train_filename + 'train_neg.pkl')
                test_pos = joblib.load(train_filename + 'test_pos.pkl')
                test_neg = joblib.load(train_filename + 'test_neg.pkl')
                # train_pos, train_neg, test_pos, test_neg = sample_neg(trainGraph, test_ratio=0.2, max_train_num=100000)

                trainGraph = trainGraph.copy()  # the observed network
                trainGraph[test_pos[0], test_pos[1]] = 0  # mask test links
                trainGraph[test_pos[1], test_pos[0]] = 0  # mask test links
                trainGraph.eliminate_zeros()  # make sure the links are masked when using the sparse matrix in scipy-1.3.x

                row, col = train_neg
                trainGraph = trainGraph.copy()
                trainGraph[row, col] = 1  # inject negative train
                trainGraph[col, row] = 1  # inject negative train
                trainGraph = nx.from_scipy_sparse_matrix(trainGraph)
                trainGraph = trainGraph.to_directed()  # convert to directed graph, as PageRank is used.
                # ------------------------------------------------------
                # testGraph = graph_util.loadGraphFromEdgeListTxt(test_filename, directed=isDirected)
                # trainGraph = trainGraph.to_directed()
                print('Num nodes: %d, num edges: %d' % (trainGraph.number_of_nodes(), trainGraph.number_of_edges()))
                inf_display = [test_task, set_dataset_name]
                tm = trainModel(inf_display, trainGraph, test_pos=test_pos, test_neg=test_neg)
                tm.train_dis(test_task=test_task, test_ratios=[0], output_filename=output_filename)


