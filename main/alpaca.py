import gym
import tensorflow as tf
import numpy as np
import yaml
import time
from copy import deepcopy
import matplotlib.pyplot as plt
from tensorflow.contrib import rnn
from tensorflow.python import debug as tf_debug

class ALPaCA:
    def __init__(self,config):
        # print('CONFIG', config)
        self.config = deepcopy(config)
        self.lr=config['lr']
        self.formulation = config['formulation']
        if self.formulation == 'time':
            self.x_dim = config['x_dim_time']
            self.y_dim = config['y_dim_time']
        elif self.formulation == 'space':
            self.x_dim = config['x_dim_space']
            self.y_dim = config['y_dim_space']
        else:
            self.x_dim = config['x_dim']
            self.y_dim = config['y_dim']
        self.sigma_scalar = self.config['sigma_eps']
        self.updates_so_far = 0
        arch_string = [str(config_layer) for config_layer in self.config['nn_layers']]
        arch_string = '_'.join(arch_string)
        self.model_name = self.formulation+'_'+str(time.time())+'_'+'action='+self.config['action']+'_nn_layers='+arch_string+'_lr='+str(self.lr)+'_sigma_eps='+str(self.sigma_scalar)+'_num_class_samples='+str(self.config['num_class_samples'])
        
    def construct_model(self,sess,graph=None):
        if not graph:
            graph = tf.get_default_graph()
            
        with graph.as_default():
            # num_updates = self.config['num_updates']
            last_layer = self.config['nn_layers'][-1]


            sigma_scalar = self.sigma_scalar #move to config
            if sigma_scalar is list:
                self.SigEps = tf.diag( np.array(sigma_scalar) )
            else:    
                self.SigEps = sigma_scalar*tf.eye(self.y_dim)

            self.K = tf.get_variable('K_init',shape=[last_layer,self.y_dim])

            self.L_asym = tf.get_variable('L_asym',shape=[last_layer,last_layer])
            self.L = self.L_asym @ tf.transpose(self.L_asym)

            self.update_x = tf.placeholder(tf.float32, shape=[None,None,self.x_dim])
            self.update_y = tf.placeholder(tf.float32, shape=[None,None,self.y_dim])

            self.num_models = tf.shape(self.update_x)[0]
            self.max_num_updates = tf.shape(self.update_x)[1]*tf.ones((self.num_models,), dtype=tf.int32)
            self.num_updates = tf.placeholder_with_default(self.max_num_updates, shape=(None,))

            self.x = tf.placeholder(tf.float32, shape=[None,None,self.x_dim])
            self.y = tf.placeholder(tf.float32, shape=[None,None,self.y_dim])

            with tf.variable_scope('model',reuse=None) as training_scope:
                # just for debug / peeking under the hood
                self.phi = tf.map_fn( lambda x: self.basis_lstm(x),
                                 elems=self.x,
                                 dtype=tf.float32)

                # the actual model 
                # tf.Assert(self.x.shape[1] == 1, [self.x])
                pred_fn = lambda inp: self.pred_f(*inp)
                self.y_pred, self.spread_fac, self.Sig_pred = tf.map_fn( pred_fn,
                                                        elems=(self.update_x, self.update_y, self.num_updates, self.x),
                                                        dtype = (tf.float32, tf.float32, tf.float32) )



                loss_fn = lambda inp: self.loss_f(*inp)
                losses = tf.map_fn( loss_fn,
                                    elems=(self.y_pred, self.spread_fac, self.Sig_pred, self.y),
                                    dtype = (tf.float32) )


                self.total_loss = tf.reduce_mean(losses)


                self.rmse = tf.reduce_mean( tf.sqrt( tf.reduce_sum( tf.square(self.y_pred - self.y), axis=-1 ) ) )
                self.mpv = tf.reduce_mean( tf.matrix_determinant(self.Sig_pred) )

                self.optimizer = tf.train.AdamOptimizer(self.lr)

                global_step = tf.Variable(0, trainable=False, name='global_step')
                self.train_op = self.optimizer.minimize(self.total_loss,global_step=global_step)

                #summaries
                config_tensor = [tf.convert_to_tensor([k, str(v)]) for k, v in self.config.items()]
                tf.summary.scalar('total_loss', self.total_loss)
                tf.summary.scalar('RMSE', self.rmse)
                tf.summary.scalar('MPV', self.mpv)
                tf.summary.tensor_summary('K', self.K)
                tf.summary.tensor_summary('Lambda', self.L)
                # tf.summary.text('config', tf.stack(config_tensor))
                self.train_writer = tf.summary.FileWriter('summaries/train_'+self.model_name, sess.graph, flush_secs=10)
                self.val_writer = tf.summary.FileWriter('summaries/val_'+self.model_name, sess.graph, flush_secs=10)
                self.merged = tf.summary.merge_all()

            sess.run(tf.global_variables_initializer())
    
    
    def model(self,K,L_inv,x):
        Phi = self.basis_lstm(x)
        mean = batch_matmul(tf.transpose(K),Phi)
        spread_fac = 1 + batch_quadform(L_inv, Phi)
        Sig = tf.expand_dims(spread_fac, axis=-1)*tf.expand_dims(self.SigEps, axis=0)
        return (mean, spread_fac, Sig)
    
    # uses online training examples ux, uy to update posterior and output posterior over y | x, ux, uy
    def pred_f(self, ux,uy,Nu,x):
        # assert(x.shape[0] == 1), "shape is wrong!"
        print('shape of ux', ux.shape)
        print('made it to pred_f')
        ux = ux[:Nu,:]
        uy = uy[:Nu,:]

        print('ux dim pred_f', ux.shape)
        uPhi = self.basis_lstm(ux)
        Kn,Ln_inv = self.batch_update(self.K, self.L, uPhi, uy)
        return self.model(Kn,Ln_inv,x)
    
    def loss_f(self,y_pred,spread_fac,Sig_pred,y_test):
        print('made it to loss_f')
        #logdet = tf.linalg.logdet(Sig_pred) # (N_test, 1)
        logdet = self.y_dim*tf.log(spread_fac) + tf.linalg.logdet(self.SigEps)
        Sig_pred_inv = tf.linalg.inv(Sig_pred)
        quadf = batch_quadform(Sig_pred_inv, (y_test - y_pred)) #(N_test, 1)
        loss = tf.reduce_mean(logdet + quadf)
        
        return loss
            
    def batch_update(self,K,L,X,Y):
        """ 
        Computes Kn and the inverse of Ln. 
        Can replace this with an update based on woodbury identity for large last layer problems,
        which reduces the complexity from O(last layer size squared) to O(batch size squared)
        """
        
        Ln_inv = tf.matrix_inverse(tf.transpose(X) @ X + L)
        Kn = Ln_inv @ (tf.transpose(X) @ Y + L @ K)
        return  Kn,Ln_inv
    
    def batch_update_np(self,K,L,X,Y):
        Ln_inv = np.linalg.inv( X.T @ X + L )
        Kn = Ln_inv @ (X.T @ Y + L @ K)
        return Kn, Ln_inv
    
    # x_up, y_up, x_test are all [N, n]
    # returns y_pred, Sigma_pred
    def test(self, sess, ux, uy, x):
        feed_dict = {
            self.update_y: uy,
            self.update_x: ux,
            self.x: x
        }
        y_pred, Sig_pred = sess.run([self.y_pred, self.Sig_pred], feed_dict)
        return y_pred, Sig_pred

    def test_rmse(self, sess, x_test_full, y_test_full):
        num_rows, row_length, _ = x_test_full.shape
        rmse_array = np.zeros((num_rows, row_length - 1))
        for row in range(num_rows):
            x_test = x_test_full[row, :, :]
            y_test = y_test_full[row, :, :]

            for i in range(1, row_length):
                ux = x_test[:i, :]
                ux = np.expand_dims(ux, axis=0)
                uy = y_test[:i, :]
                uy = np.expand_dims(uy, axis=0)
                x = x_test[i, :]
                x = np.expand_dims(np.expand_dims(x, axis=0), axis=0)
                y = y_test[i, :]
                y = np.expand_dims(np.expand_dims(y, axis=0), axis=0)

                y_pred, Sig_pred = self.test(sess, ux, uy, x)

                feed_dict = {
                    self.y_pred: y_pred, 
                    self.y: y
                }
                rmse = sess.run(self.rmse, feed_dict)
                # print('rmse', rmse)
                rmse_array[row, i-1] = rmse

        avg_rmse = np.mean(rmse_array)
        return avg_rmse

    # convenience function to use just the encoder on numpy input
    def encode(self, sess, x):
        feed_dict = {
            self.x: x
        }
        return sess.run(self.phi, feed_dict)
        
    def train(self, sess, y, x, y_val, x_val, num_train_updates, plot_loss=False, save_model=True):
        num_samples = self.config['num_class_samples']
        horizon = self.config['data_horizon']
        test_horizon = self.config['test_horizon']
        loss_array = np.zeros(num_train_updates)
        val_loss_array = np.zeros(num_train_updates)
        saver = tf.train.Saver()

        #minimize loss
        for i in range(num_train_updates):
            # Check val stats
            feed_dict_val = self.gen_variable_horizon_data_ordered(x_val, y_val, i+1, num_samples, horizon, test_horizon)
            print('feed_dict_val ux shape', feed_dict_val[self.update_x].shape)
            print('feed_dict_val uy shape', feed_dict_val[self.update_y].shape)
            print('feed_dict_val x shape', feed_dict_val[self.x].shape)
            print('feed_dict_val y shape', feed_dict_val[self.y].shape)
            print('feed_dict_val num_updates shape', feed_dict_val[self.num_updates].shape)

            # aaa, bbb, ccc = self.pred_f(feed_dict_val[self.update_x][0,:,:], feed_dict_val[self.update_y][0,:,:], feed_dict_val[self.num_updates][0], feed_dict_val[self.x][0,:,:])
            # print('aaa', aaa)

            print('hello hello')

            val_summary, val_loss = sess.run([self.merged,self.total_loss],feed_dict_val)
            self.val_writer.add_summary(val_summary, self.updates_so_far)
            val_loss_array[i] = val_loss

            # Check train stats
            feed_dict = self.gen_variable_horizon_data_ordered(x, y, i+1, num_samples, horizon, test_horizon)
            summary,loss, _ = sess.run([self.merged,self.total_loss,self.train_op],feed_dict)
            self.train_writer.add_summary(summary, self.updates_so_far)
            loss_array[i] = loss

            if i % 50 == 0:
                print('loss:', loss)
                print('val_loss: ', val_loss)

            if i % 1000 == 0:
                # Append the index number to the checkpoint name:
                saver.save(sess, 'checkpoints/'+self.model_name, global_step=i)
            
            self.updates_so_far += 1

        saver.save(sess, 'checkpoints/'+self.model_name, global_step=i)

        if plot_loss:
            # plot loss_array
            plt.plot(np.arange(num_train_updates), loss_array, np.arange(num_train_updates), val_loss_array)
            plt.ylabel('Loss')
            plt.show()

    def gen_variable_horizon_data(self,x,y,num_samples,horizon,test_horizon):
        num_updates = np.random.randint(horizon+1, size=num_samples)
        
        M,N = x.shape[0:2]
        M_ind = np.random.choice(M, num_samples)
        batch_ind = np.random.choice(N, test_horizon+horizon)
        x = x[M_ind,:,:]
        y = y[M_ind,:,:]
        x = x[:,batch_ind,:]
        y = y[:,batch_ind,:]
        
        # splitting into update and train sets along number of samples; first index is task/param
        uy = y[:,:horizon,:]
        ux = x[:,:horizon,:]

        y = y[:,horizon:,:]
        x = x[:,horizon:,:]

        feed_dict = {
                self.update_y: uy,
                self.update_x: ux,
                self.y: y,
                self.x: x,
                self.num_updates: num_updates
                }
        
        return feed_dict

    def gen_variable_horizon_data_ordered(self,x,y,time_step,num_samples,horizon,test_horizon):
        num_updates = np.random.randint(horizon+1, size=num_samples)
        print('num_updates', num_updates)
        
        M,N = x.shape[0:2]
        M_ind = np.random.choice(M, num_samples)
        x = x[M_ind,:,:]
        y = y[M_ind,:,:]
        
        # splitting into update and train sets along number of samples; first index is task/param
        uy = y[:,:time_step+1,:]
        ux = x[:,:time_step+1,:]

        y = y[:,time_step:time_step+1,:]
        x = x[:,time_step:time_step+1,:]

        # y = np.expand_dims(y[:,time_step,:], axis=1)
        # x = np.expand_dims(x[:,time_step,:], axis=1)

        print('time_step', time_step)
        print('y', y.shape)
        print('x', x.shape)
        print('ux dimensions here', ux.shape)

        feed_dict = {
                self.update_y: np.float32(uy),
                self.update_x: np.float32(ux),
                self.y: np.float32(y),
                self.x: np.float32(x),
                self.num_updates: num_updates,
                }
        
        return feed_dict

    # def gen_variable_horizon_data_time(self,x,y,num_samples,horizon,test_horizon):
    #     num_updates = np.random.randint(horizon+1, size=num_samples)

    #     print('hola', [item.shape for item in y])
    #     M = len(x)
    #     M_ind = np.random.choice(M, num_samples)
    #     x = [x[i] for i in M_ind]
    #     y = [y[i] for i in M_ind]

    #     ux = [arr[:horizon,:] for arr in x]
    #     uy = [arr[:horizon,:] for arr in y]
    #     x = [arr[horizon:horizon+test_horizon,:] for arr in x]
    #     y = [arr[horizon:horizon+test_horizon,:] for arr in y]
    #     print('hey hey', [item.shape for item in y])
        
    #     # feed_dict = {
    #     #         self.update_y: uy,
    #     #         self.update_x: ux,
    #     #         self.y: np.expand_dims(y, axis=0),
    #     #         self.x: np.expand_dims(x, axis=0),
    #     #         self.num_updates: num_updates
    #     # }

    #     feed_dict = {
    #             self.update_y: np.stack(uy),
    #             self.update_x: np.stack(ux),
    #             self.y: np.stack(y),
    #             self.x: np.stack(x),
    #             self.num_updates: num_updates
    #     }

    #     return feed_dict

    def basis(self,x,name='basis'):
        layer_sizes = self.config['nn_layers']
        activations = {
            'relu': tf.nn.relu,
            'tanh': tf.nn.tanh,
            'sigmoid': tf.nn.sigmoid
        }
        activation = activations[ self.config['activation'] ]

        inp = x
        print('x basis', x)
        with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
            # inp = tf.stack([x],axis=1)
            for units in layer_sizes:
                inp = tf.layers.dense(inputs=inp, units=units,activation=activation)
            
        return inp

    def basis_lstm(self, x, name='basis_lstm'):
        print('made it to basis_lstm')
        layer_sizes = self.config['nn_layers']
        activations = {
            'relu': tf.nn.relu,
            'tanh': tf.nn.tanh,
            'sigmoid': tf.nn.sigmoid
        }
        activation = activations[ self.config['activation'] ]

        print('updates_so_far', self.updates_so_far)
        inp = tf.unstack(x, num=self.updates_so_far+1)
        inp = [tf.expand_dims(x, axis=0) for x in inp]
        print('inp', inp)
        n_hidden = 256

        with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
            lstm_layer = rnn.BasicLSTMCell(n_hidden)
            outputs, _ = rnn.static_rnn(lstm_layer, inp, dtype=tf.float32)

        return outputs[0]

    
def batch_matmul(mat, batch_v, name='batch_matmul'):
    with tf.name_scope(name):
        return tf.transpose(tf.matmul(mat,tf.transpose(batch_v)))
        
def tf_quadform(A,b):
    print('A shape', A.get_shape())
    print('b shape', b.get_shape())
    q1 = A @ b
    return tf.transpose(b) @ q1

# works for A = [n,n] or [N,n,n]
# assumes b = [N,n]
# returns  [N,1]
def batch_quadform(A, b):
    A_dims = A.get_shape().ndims
    b_vec = tf.expand_dims(b, axis=-1)
    if A_dims == 3:
        return tf.squeeze( tf.matrix_transpose(b_vec) @ A @ b_vec, axis=-1)
    elif A_dims == 2:
        Ab = tf.expand_dims( tf.transpose( A @ tf.transpose(b) ), axis=-1) # N x n x 1
        return tf.squeeze( tf.matrix_transpose(b_vec) @ Ab, axis = -1)
    else:
        raise ValueError('Matrix size of %d is not supported.'%(A_dims))

        
def sampleMN(K, L_inv, Sig):
    mean = np.reshape(K.T, [-1])
    cov = np.kron(Sig, L_inv)
    K_vec = np.random.multivariate_normal(mean,cov)
    return np.reshape(K_vec, K.T.shape).T
        
class AdaptiveDynamics(ALPaCA):
    def __init__(self, config):
        super(AdaptiveDynamics, self).__init__(config)
        
    def train(self,sess,y,x,num_train_updates):
        super(AdaptiveDynamics, self).train(sess,y,x,num_train_updates)
        
        # obtain key things as numpy
        self.K0,self.L0,self.sigeps = sess.run( (self.K, self.L, self.SigEps) )
        self.reset_to_prior()
    
    def sample_rollout(self, sess, x0, actions):
        T, a_dim = actions.shape
        mult_sample = False
        if x0.ndim == 1:
            N_samples = 1
            x_dim = x0.shape[0]
            
            x0 = np.expand_dims(x0, axis=1)
        elif x0.ndim == 2:
            mult_sample = True
            N_samples = x0.shape[0]
            x_dim = x0.shape[1]
            
        actions = np.tile(np.expand_dims(actions,axis=0), (N_samples, 1, 1))
            
        x_pred = np.zeros( (N_samples, T+1, x_dim) )
        x_pred[:,0,:] = x0
        
        Kn = self.Ln_inv @ self.Qn
        Ks = [ sampleMN(Kn, self.Ln_inv, self.sigeps) for _ in range(N_samples) ]
        for t in range(0, T):
            x_inp = np.concatenate( (x_pred[:,t:t+1,:], actions[:,t:t+1,:]), axis=2 )
            phi = self.encode(sess, x_inp)
            for j in range(N_samples):
                x_pred[j,t+1,:] = x_pred[j,t,:] + np.squeeze( phi[j,:,:] @ Ks[j] )
        
        if mult_sample:
            return x_pred[:,1:,:]
        else:
            return x_pred[0,1:,:]
    
    def reset_to_prior(self):
        self.Ln_inv = np.linalg.inv(self.L0)
        self.Qn = self.L0 @ self.K0
        
    def incorporate_transition(self,sess,x,u,xp):
        # perform RLS update to Kn, Ln
        x_inp = np.reshape( np.concatenate( (x,u), axis=0 ), (1,1,-1) )
        phi = self.encode(sess, x_inp)[0,:,:].T
        y = np.reshape(xp - x, (-1,1))
        
        phiLamphi = (phi.T @ self.Ln_inv @ phi)[0,0]
        LninvPhi = self.Ln_inv @ phi
        self.Ln_inv = self.Ln_inv - 1./(1. + phiLamphi) * LninvPhi @ LninvPhi.T
        self.Qn = phi @ y.T + self.Qn
        
    def incorporate_batch(self, sess, X, Y):
        Phi = self.encode(sess,X)
        Kn,Ln_inv = self.batch_update_np(self.K0,self.L0,Phi[0,:,:],Y[0,:,:])
        self.Ln_inv = Ln_inv
        self.Kn = Kn
        self.Qn = np.linalg.inv(self.Ln_inv) @ self.Kn
