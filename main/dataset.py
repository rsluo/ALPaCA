import numpy as np
import gym
import tqdm
import os

class Dataset:
    def __init__(self):
        pass

    # draw n_sample (x,y) pairs drawn from n_func functions
    # returns (x,y) where each has size [n_func, n_samples, x/y_dim]
    def sample(self, n_funcs, n_samples):
        raise NotImplementedError

class HandTrajectoryDataset(Dataset):
    def __init__(self, root_dir, action_label, basis, num_input_points, num_hand_points=21, input_dim=3, row_length=30, shuffle=False):
        self.root_dir = root_dir
        self.action_label = action_label
        self.basis = basis
        self.num_input_points = num_input_points
        self.num_hand_points = num_hand_points
        self.input_dim = input_dim
        self.row_length = row_length
        self.shuffle = shuffle
        # for a, b, c in os.walk(self.root_dir):
        #     print('a', a)
        #     print('b', b)
        #     print('c', c)
        # Get all filepaths, excluding empty or small files
        if self.action_label == 'train-set':
            self.all_filepaths = [a for (a, b, c) in os.walk(self.root_dir) if (len(b) == 0) and (os.stat(os.path.join(a,c[0])).st_size > 6000)]
        else:
            self.all_filepaths = [a for (a, b, c) in os.walk(self.root_dir) if (len(b) == 0) and (a.split('/')[-2] == self.action_label) and (os.stat(os.path.join(a,c[0])).st_size > 15000)]

    def get_item(self, filepath_idx):
        filepath = os.path.join(self.all_filepaths[filepath_idx], "skeleton.txt")

        with open(filepath) as file:
            file_contents = file.readlines()
            traj_length = len(file_contents)

            assert (traj_length >= self.num_input_points + 1), "Trajectory is too short!"

            num_rows = (traj_length - self.num_input_points - 1) // self.row_length
            x_array = np.zeros((num_rows, self.row_length, (self.num_input_points-1)*self.input_dim*self.num_hand_points))
            y_array = np.zeros((num_rows, self.row_length, self.input_dim*self.num_hand_points))
            init_array = np.zeros((num_rows, self.row_length, self.input_dim*self.num_hand_points))
            
            for i in range(self.row_length * num_rows):
                current_row = i // self.row_length
                for j in range(1, self.num_input_points):
                    prev_file_line = file_contents[i + j - 1].split()
                    file_line = file_contents[i + j].split()
                    for k in range(len(file_line) - 1):
                        x_array[current_row, (i%self.row_length), (j-1)*self.input_dim*self.num_hand_points + k] = float(file_line[k+1]) - float(prev_file_line[k+1])

                temp_y = np.asarray(file_contents[i+j+1].split()).astype(np.float)
                temp_y_prev = np.asarray(file_contents[i+j].split()).astype(np.float)
                y_array[current_row, (i%self.row_length), :] = temp_y[1:] - temp_y_prev[1:]
                temp_init = np.asarray(file_contents[i].split()).astype(np.float)
                init_array[current_row, i%self.row_length, :] = temp_init[1:]

        return x_array, y_array, init_array

    def sample(self, n_funcs, n_samples, return_init=False):
        sample_ids = np.random.choice(len(self.all_filepaths), n_funcs)

        # If the basis is an lstm. Not really using n_samples in this case, just getting all points in the trajectory 
        if self.basis == 'lstm':        
            x_matrix = np.zeros((0, self.row_length, (self.num_input_points-1)*self.input_dim*self.num_hand_points))
            y_matrix = np.zeros((0, self.row_length, self.input_dim*self.num_hand_points))
            init_matrix = np.zeros((0, self.row_length, self.input_dim*self.num_hand_points))     

            print('n_funcs:', n_funcs)
            for i in range(n_funcs):
                print('i:', i)
                idx = sample_ids[i]
                x_array, y_array, init_array = self.get_item(idx)
                print('x_array shape', x_array.shape)
                y_matrix = np.vstack((y_matrix, y_array))
                x_matrix = np.vstack((x_matrix, x_array))
                init_matrix = np.vstack((init_matrix, init_array))
                print('x_matrix shape', x_matrix.shape)
            print()

        # If the basis is an mlp
        else:
            x_matrix = np.zeros((0, n_samples, (self.num_input_points-1)*self.input_dim*self.num_hand_points))
            y_matrix = np.zeros((0, n_samples, self.input_dim*self.num_hand_points))
            init_matrix = np.zeros((0, n_samples, self.input_dim*self.num_hand_points))  

            for i in range(n_funcs):
                idx = sample_ids[i]
                x_array, y_array, init_array = self.get_item(idx)
                samples_to_keep = np.random.choice(self.row_length, n_samples)
                y_matrix = np.vstack((y_matrix, y_array[:,samples_to_keep,:]))
                x_matrix = np.vstack((x_matrix, x_array[:,samples_to_keep,:]))
                init_matrix = np.vstack((init_matrix, init_array[:,samples_to_keep,:]))

        if return_init:
            return x_matrix, y_matrix, init_matrix

        return x_matrix, y_matrix


class PresampledDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.x_dim = X.shape[-1]
        self.y_dim = Y.shape[-1]
        self.N = X.shape[0]
        self.T = X.shape[1]
    
    def sample(self, n_funcs, n_samples):
        x = np.zeros((n_funcs, n_samples, self.x_dim))
        y = np.zeros((n_funcs, n_samples, self.y_dim))
        
        for i in range(n_funcs):
            j = np.random.randint(self.N)
            if n_samples > self.T:
                raise ValueError('You are requesting more samples than are in the dataset.')
 
            inds_to_keep = np.random.choice(self.T, n_samples)
            x[i,:,:] = self.X[j,inds_to_keep,:]
            y[i,:,:] = self.Y[j,inds_to_keep,:]
        
        return x,y
        
class PresampledTrajectoryDataset(Dataset):
    def __init__(self, trajs, controls):
        self.trajs = trajs
        self.controls = controls
        self.o_dim = trajs[0].shape[-1]
        self.u_dim = controls[0].shape[-1]
        self.N = len(trajs)
    
    def sample(self, n_funcs, n_samples):
        o_dim = self.o_dim
        u_dim = self.u_dim
        x_dim = o_dim + u_dim
        y_dim = o_dim
        x = np.zeros((n_funcs, n_samples, x_dim))
        y = np.zeros((n_funcs, n_samples, y_dim))
        
        for i in range(n_funcs):
            j = np.random.randint(self.N)
            T = self.controls[j].shape[0]
            if n_samples > T:
                raise ValueError('You are requesting more samples than are in this trajectory.')
            start_ind = 0
            if T > n_samples:
                start_ind = np.random.randint(T-n_samples)
            inds_to_keep = np.arange(start_ind, start_ind+n_samples)
            x[i,:,:self.o_dim] = self.trajs[j][inds_to_keep]
            x[i,:,self.o_dim:] = self.controls[j][inds_to_keep]
            y[i,:,:] = self.trajs[j][inds_to_keep+1] #- self.trajs[j][inds_to_keep]
        
        return x,y

class SinusoidDataset(Dataset):
    def __init__(self, config, noise_var=None, rng=None):
        self.amp_range = config['amp_range']
        self.phase_range = config['phase_range']
        self.freq_range = config['freq_range']
        self.x_range = config['x_range']
        if noise_var is None:
            self.noise_std = np.sqrt( config['sigma_eps'] )
        else:
            self.noise_std = np.sqrt( noise_var )
            
        self.np_random = rng
        if rng is None:
            self.np_random = np.random

    def sample(self, n_funcs, n_samples, return_lists=False):
        x_dim = 1
        y_dim = 1
        x = np.zeros((n_funcs, n_samples, x_dim))
        y = np.zeros((n_funcs, n_samples, y_dim))

        amp_list = self.amp_range[0] + self.np_random.rand(n_funcs)*(self.amp_range[1] - self.amp_range[0])
        phase_list = self.phase_range[0] + self.np_random.rand(n_funcs)*(self.phase_range[1] - self.phase_range[0])
        freq_list = self.freq_range[0] + self.np_random.rand(n_funcs)*(self.freq_range[1] - self.freq_range[0])
        for i in range(n_funcs):
            x_samp = self.x_range[0] + self.np_random.rand(n_samples)*(self.x_range[1] - self.x_range[0])
            y_samp = amp_list[i]*np.sin(freq_list[i]*x_samp + phase_list[i]) + self.noise_std*self.np_random.randn(n_samples)

            x[i,:,0] = x_samp
            y[i,:,0] = y_samp

        if return_lists:
            return x,y,freq_list,amp_list,phase_list

        return x,y
    
class MultistepDataset(Dataset):
    def __init__(self, config, noise_var=None, rng=None):
        self.step_min = config['step_min']
        self.step_max = config['step_max']
        self.num_steps = config['num_steps']
        self.x_range = config['x_range']
        if noise_var is None:
            self.noise_std = np.sqrt( config['sigma_eps'] )
        else:
            self.noise_std = np.sqrt( noise_var )
            
        self.np_random = rng
        if rng is None:
            self.np_random = np.random
            
    def sample(self, n_funcs, n_samples, return_lists=False):
        x_dim = 1
        y_dim = 1
        x = np.zeros((n_funcs, n_samples, x_dim))
        y = np.zeros((n_funcs, n_samples, y_dim))
        
        step_mat = np.zeros((n_funcs, self.num_steps))
        
        for i in range(n_funcs):
            step_pts = self.step_min + self.np_random.rand(self.num_steps)* (self.step_max - self.step_min)
            step_mat[i,:] = step_pts
            
            x_samp = self.x_range[0] + self.np_random.rand(n_samples)*(self.x_range[1] - self.x_range[0])
            y_samp = self.multistep(x_samp, step_pts)

            x[i,:,0] = x_samp
            y[i,:,0] = y_samp

        if return_lists:
            return x,y,step_mat

        return x,y
    
    def multistep(self, x, step_pts):
        x = x.reshape([1,-1])
        step_pts = step_pts.reshape([-1,1])
        y = 2.*np.logical_xor.reduce( x > step_pts, axis=0) - 1.
        y += self.noise_std*self.np_random.randn(x.shape[1])
        return y

# Assumes env has a forward_dynamics(x,u) function
class GymUniformSampleDataset(Dataset):
    def __init__(self, env):
        self.env = env
        self.o_dim = env.observation_space.shape[-1]
        self.u_dim = env.action_space.shape[-1]

    def sample(self, n_funcs, n_samples):
        o_dim = self.o_dim
        u_dim = self.u_dim
        x_dim = o_dim + u_dim
        y_dim = o_dim
        x = np.zeros((n_funcs, n_samples, x_dim))
        y = np.zeros((n_funcs, n_samples, y_dim))

        for i in range(n_funcs):
            self.env.reset()
            for j in range(n_samples):
                s = self.env.get_ob_sample()
                a = self.env.action_space.sample()#get_ac_sample()
                sp = self.env.forward_dynamics(s,a)
                
                x[i,j,:o_dim] = s
                x[i,j,o_dim:] = a
                y[i,j,:] = sp

        return x,y

# wraps a gym env + policy as a dataset
# assumes that the gym env samples parameters from the prior upon reset
class GymDataset(Dataset):
    def __init__(self, env, policy, state_dim=None):
        self.env = gym.wrappers.TimeLimit(env, max_episode_steps=100)
        self.policy = policy
        self.use_state = False
        self.o_dim = env.observation_space.shape[-1]
        if state_dim is not None:
            self.use_state = True
            self.o_dim = state_dim
        self.u_dim = env.action_space.shape[-1]

    def sample(self, n_funcs, n_samples, shuffle=False, verbose=False):
        o_dim = self.o_dim
        u_dim = self.u_dim
        x_dim = o_dim + u_dim
        y_dim = o_dim
        x = np.zeros((n_funcs, n_samples, x_dim))
        y = np.zeros((n_funcs, n_samples, y_dim))

        
        pbar = tqdm.tqdm(disable=(not verbose), total=n_funcs)
        for i in range(n_funcs):
            # sim a trajectory
            x_traj = []
            u_traj = []
            xp_traj = []

            ob = self.env.reset()
            if self.use_state:
                s = self.env.unwrapped.state
            done = False
            while not done:
                ac = self.policy(ob)
                obp, _, done, _ = self.env.step(ac)
                
                if self.use_state:
                    sp = self.env.unwrapped.state
                    x_traj.append(s)
                    u_traj.append(ac)
                    xp_traj.append(sp)
                else:
                    x_traj.append(ob)
                    u_traj.append(ac)
                    xp_traj.append(obp)

                ob = obp
                if self.use_state:
                    s = sp

            T = len(x_traj)
            if T < n_samples:
                print('episode did not last long enough')
                #n_samples = T-1
                i -= 1
                continue

            if shuffle:
                inds_to_keep = np.random.choice(T, n_samples)
            else:
                start_ind = 0 #np.random.randint(T-n_samples)
                inds_to_keep = range(start_ind, start_ind+n_samples)
            x[i,:,:o_dim] = np.array(x_traj)[inds_to_keep,:]
            x[i,:,o_dim:] = np.array(u_traj)[inds_to_keep,:]
            y[i,:,:] = np.array(xp_traj)[inds_to_keep,:]
            
            pbar.update(1)

        pbar.close()
        return x,y

    
class Randomizer(gym.Wrapper):
    def __init__(self, env, prereset_fn):
        super(Randomizer, self).__init__(env)
        self.prereset_fn = prereset_fn
    
    def reset(self):
        self.prereset_fn(self.unwrapped)
        return self.env.reset()
        
    
