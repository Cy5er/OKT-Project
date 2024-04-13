#!/usr/bin/env python
# coding: utf-8

# In[4]:


import torch
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import seaborn as sns
sns.set_theme()
import random
from tqdm import tqdm
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')


# In[2]:


# os.environ['CUDA_VISIBLE_DEVICES']='7'


# In[5]:


## set random seed
def set_random_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ### load model

# In[6]:


path = 'results/all_submissions'
lstm_init = 'rand'
print(os)

## Create a LSTM with input dimension [968] (prompt=[768], ASTNN=[200]) and output dimension [768]
set_random_seed(123)
lstm = torch.load(os.path.join(path, 'lstmm'))
lstm.cuda();
lstm.eval();


# ### load dataset

# In[5]:


## load data, optionally use first attempt only
dataset = pd.read_pickle('../data/dataset.pkl')
students = dataset['SubjectID'].unique()
p_ids = dataset['ProblemID'].unique()
print(f'Available question IDs: \n  {p_ids}')


# ### Load dataset and all student knowledge states

# In[6]:


def get_knowledge_states_for_generator(lstm, lstm_inputs, students):
    '''
    get a student's knowledge state
    '''
    # get lstm inputs
    lstm_ins = [lstm_inputs[s] for s in students]
    max_len = max(len(i) for i in lstm_ins)
    padded_lstm_ins = [i + [torch.zeros(i[0].shape[0])]*(max_len - len(i)) for i in lstm_ins]
    padded_lstm_ins = torch.stack([torch.stack(x, dim=0) for x in padded_lstm_ins], dim=1).float() # dim=T*B*D

    # get knowledge states
    if lstm_init == 'rand':
        hidden_h, hidden_c = torch.rand(1, padded_lstm_ins.shape[1], 768).cuda(), torch.rand(1, padded_lstm_ins.shape[1], 768).cuda()
    elif lstm_init == 'zero':
        hidden_h, hidden_c = torch.zeros(1, padded_lstm_ins.shape[1], 768).cuda(), torch.zeros(1, padded_lstm_ins.shape[1], 768).cuda()

    out, hidden = lstm(padded_lstm_ins.cuda(), (hidden_h, hidden_c)) # shape = T*B*D
        
    return out


# In[7]:


## results: a dictionary with key as student ID, values are a list of dictionary, one for a student's attempt 
def get_results(students, dataset, lstm):
    results = {}
    i_prev = 0
    for idx in tqdm(range(len(students)), desc='students'):

        s = dataset[dataset['SubjectID'] == students[idx]]
        lstm_input = {students[idx]:s.input.tolist()} 
        
        with torch.no_grad():
            out = get_knowledge_states_for_generator(lstm, lstm_input, [students[idx]])
        out = out.cpu().detach()
        
        assert(out.shape[0] == len(s))

        student_results = []
        for i, row in s.iterrows():
            real_idx = i - i_prev
            student_results.append({'p_id': s.iloc[real_idx]['ProblemID'],
                                    'ks': out[real_idx][0].cpu().numpy(),
                                    't': real_idx,
                                    'score': s.iloc[real_idx]['Score_y']})
        results[students[idx]] = student_results
        i_prev = i+1 

    return results


# In[8]:


results = get_results(students, dataset, lstm)


# ### Extract knowledge state for a selecte question

# In[9]:


def p_data_formulation(selected_id):    
    kcs_plot = []
    students_ids = []
    timesteps_plot = []
    scores_plot = []
    for r in results:
        for s in results[r]:
            if s['p_id'] == selected_id:
                students_ids.append(r)
                kcs_plot.append(s['ks'])
                timesteps_plot.append(s['t'])
                scores_plot.append(s['score'])

    return students_ids, kcs_plot, timesteps_plot, scores_plot


# ### Visualization of latent student knowledge states (each color corresponds to one student)

# In[12]:


def plot_ks(selected_question, p_ids):
    if selected_question not in p_ids:
        print('Not a valid question, please choose another question')
    else:
        print(f'Plotting Knowledge States for Question {selected_question}')

    students_ids, kcs_plot, _, _ = p_data_formulation(selected_question)
    
    # perform TSNE dimension reduction to plot knowledge states in latent space
    kcs_tsne = TSNE(n_components=2).fit_transform(np.vstack(kcs_plot))

    fig = plt.figure(figsize=(8,8))

    ## color by each student
    color = cm.rainbow(np.linspace(0, 1, len(set(students_ids))))
    student_color_dict = {list(set(students_ids))[i]: color[i] for i in range(len(color))}
    plt.scatter(x=kcs_tsne[:,0], y=kcs_tsne[:,1], 
                c=[student_color_dict[p] for p in students_ids], alpha=0.7)

    xt = plt.xticks(color='w')
    yt = plt.yticks(color='w')

    plt.show()


# In[13]:


## question id = 5
plot_ks(5, p_ids)


# ### Plot knowledge state trajectory of selected student and question

# In[25]:


def plot_trajectory(selected_question, selected_student, p_ids, students):
    if selected_question not in p_ids:
        print('Not a valid question, please choose another question')
    elif selected_student > 246:
        print('Not a valid student, please choose another student')
    else:
        print(f'Plotting Knowledge States for Question {selected_question} and student number {selected_student}')

    selected_student = students[selected_student]
    
    students_ids, kcs_plot, timesteps_plot, scores_plot = p_data_formulation(selected_question)
    
    set_random_seed(123)
    # perform TSNE dimension reduction to plot knowledge states trajectory
    kcs_tsne = TSNE(n_components=2).fit_transform(np.vstack(kcs_plot))

    # select student to plot trajectory
    student_selected_idx = np.array([1 if item == selected_student else 0 for item in students_ids]).astype(np.bool)
    kcs_tsne_j = kcs_tsne[student_selected_idx,:]
    scores_plot_j = np.array(scores_plot)[student_selected_idx]
    timesteps_plot_j = np.array(timesteps_plot)[student_selected_idx]

    # color the wrong answer as red, partially correct answe as yellow and correct answer as green
    color = ['red', 'yellow', 'green']
    labels = ['wrong', 'partially correct', 'correct']
    score_color_dict = {list(set(scores_plot))[i]: color[i] for i in range(len(set(scores_plot)))}
    score_label_dict = {list(set(scores_plot_j))[idx]: labels[idx] for idx in range(len(set(scores_plot_j)))}

    fig = plt.figure(figsize=(8,8))
#     fig = plt.scatter(x=kcs_tsne_j[:,0], y=kcs_tsne_j[:,1],
#                       c=[score_color_dict[p] for p in scores_plot_j])
    
    # plot points for each attempt 
    for ss in set(scores_plot_j):
        tmp = np.array([1 if item==ss else 0 for item in scores_plot_j]).astype(np.bool)
        fig = plt.scatter(x=kcs_tsne_j[:,0][tmp], y=kcs_tsne_j[:,1][tmp],
                          c=score_color_dict[ss], s=150,
                          label=score_label_dict[ss])
    
    # plot timestep for each attempt (starting at zero)
    for idx, time in enumerate(timesteps_plot_j):
        plt.annotate(time - timesteps_plot_j[0], 
                     (kcs_tsne_j[:,0][idx], kcs_tsne_j[:,1][idx]), 
                     fontsize=20, color='black')
    
    # plot trajectory lines 
    for idx in range(len(kcs_tsne_j[:,0])-1):
        plt.annotate("", xy=(kcs_tsne_j[:,0][idx+1], kcs_tsne_j[:,1][idx+1]), 
                     xytext=(kcs_tsne_j[:,0][idx], kcs_tsne_j[:,1][idx]),
                     arrowprops=dict(arrowstyle="->,head_length=0.8,head_width=0.5", color='blue', lw=1.5))

    plt.legend(fontsize=15, loc='upper right')
    xt = plt.xticks(color='w')
    yt = plt.yticks(color='w')

    plt.show()


# In[26]:


## question id = 38, student number = 129
plot_trajectory(38, 129, p_ids, students)


# In[ ]:




