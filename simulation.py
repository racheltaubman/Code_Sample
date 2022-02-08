from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
from datetime import date
import itertools
import os
import math
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import matplotlib._color_data as mcd


#############################
start_time = time.time()


def Hopfield_with_Lrules(N, E_indices, I_indices, stimuli, EE_rows, EE_cols, 
        EI_rows, EI_cols, IE_rows, IE_cols, J, T, eta, H, eps, alpha):
    """ Hopfield network with STDP and IP learning rules. Outputs energy and
    MSE values and arrays keeping track of the sum of connection weights,
    average firing, sum of excitatory firing thresholds, and hyperparameters
    epsilon and alpha values.
    """

    # FL = "firing list," records neuron firing for each neuron in response to 
    # each stimulus value in succession. The first 2 firing data points 
    # are randomly generated and won't be considered in evaluating the model
    FL = [list(np.random.choice([0, 1], N)), list(np.random.choice([0, 1], N))]
    # f represents the firing in response to one stimulus value
    f = np.array(FL[1])
    
    Jsums = []
    Jsums.append(np.sum(J0[EE_rows, EE_cols]))
    
    avT_ex = [np.sum(T0[E_indices])/len(T0[E_indices])]
    NE = len(E_indices)
    NI = len(I_indices)

    avrgfiring = [sum(FL[1])/N]
    
    epsvals = [eps]
    alphavals = [alpha]
    #firing at t corresponds to stimuli at t-1 and state at t-2
    for t in range(len(stimuli)):
        f_new = np.zeros(N)
        
        f_new[E_indices] = np.dot(
            f[E_indices], J[EE_cols, EE_rows].reshape(NE, NE) ) \
                - np.dot(f[I_indices], J[IE_cols, IE_rows].reshape(NI, NE) ) \
                    + stimuli[t] - T[E_indices]

        f_new[I_indices] = np.dot(
            f[E_indices], J[EI_cols, EI_rows].reshape(NE, NI) ) - T[I_indices]

        f_new[f_new >= T] = 1
        f_new[f_new < T] = 0
        f = f_new

        FL.append(f.tolist())
        FA = np.array(FL)

        avrgfiring.append(sum(f)/N)

        #update J
        if np.all(J[EE_rows, EE_cols] >= 0) and (np.sum(J[EE_rows, EE_cols]) <= NE):
            J[EE_rows, EE_cols] += (
                eps * (FA[t][E_indices].reshape(-1,1) * FA[t - 1][E_indices]) -  \
                alpha * (FA[t - 1][E_indices].reshape(-1,1) * FA[t][E_indices])
                            ).reshape(NE*NE,)

        elif np.sum(J[EE_rows, EE_cols]) > NE:
            eps = (eps - 0.01*eps)/(eps+alpha)
            alpha = (alpha + 0.01*alpha)/(eps+alpha)
            
        elif np.sum(J[EE_rows, EE_cols]) < 0:
            alpha = (alpha - 0.01*alpha)/(eps+alpha)
            eps = (eps + 0.01*eps)/(eps+alpha)

        epsvals.append(eps)
        alphavals.append(alpha)            
        Jsums.append(np.sum(J[EE_rows, EE_cols]))

        #update T
        T[E_indices] = T[E_indices] + eta * (FA[t][E_indices] - H[0:len(E_indices)])

        avT_ex.append(np.sum(T[E_indices])/len(T[E_indices]))


    #all firing values corresponding to last 1000 stimuli values
    FA_last = FA[len(FA) - 1001 : -1]
    #last 1000 stimuli values
    stimuli_last = stimuli[len(stimuli) - 1000:]

    reg_all = LinearRegression().fit(FA_last, stimuli_last)

    return (
        np.mean(FA_last), 1-reg_all.score(FA_last, stimuli_last),
        Jsums, avrgfiring, avT_ex,
        epsvals, alphavals
        )


def Hopfield_no_rules(N, E_indices, I_indices, stimuli, 
        EE_rows, EE_cols, EI_rows, EI_cols, IE_rows, IE_cols, J, T):
    """ Hopfield network no learning rules (connection weights don't change). 
      Outputs energy and MSE values and average firing.
    """

    #firing list
    FL = [list(np.random.choice([0, 1], N))]
    f = np.array(FL[0])
    avrgfiring = [sum(FL[0])/N]

    for t in range(len(stimuli)):
        f_new = np.zeros(N)

        f_new[E_indices] = np.dot(
            f[E_indices], J[EE_cols, EE_rows].reshape(NE, NE) )\
                - np.dot(f[I_indices], J[IE_cols, IE_rows].reshape(NI, NE) ) \
                    + stimuli[t] - T[E_indices]

        f_new[I_indices] = np.dot(
            f[E_indices], J[EI_cols, EI_rows].reshape(NE, NI) ) - T[I_indices]
        
        
        f_new[f_new >= T] = 1
        f_new[f_new < T] = 0
        f = f_new

        avrgfiring.append(sum(f)/N)

        FL.append(list(f))

    #firing array with values corresponding to last 1000 stimuli values
    FA = np.array(FL)
    FA_last = FA[len(FA) - 1001 : -1]
    #last 1000 stimuli values
    stimuli_last = stimuli[len(stimuli) - 1000 :]

    reg_all = LinearRegression().fit(FA_last, stimuli_last)

    return (
        np.mean(FA_last), 1-reg_all.score(FA_last, stimuli_last), avrgfiring
        )


def initialvalues(NE, NI):
    """Outputs initial values for connection weight matrix J and firing
    thresholds T based on the number of excitatory and inhibitory neurons 
    (NE and NI, respectively). Also outputs indices for matrix positions for
    pairwise connections, i.e. excitatory-excitatory (EE), 
    excitatory-inhibitory (EI), inhibitory-excitatory (IE), and 
    inhibitory-inhibitory (II). II connections and self-connections equal zero.
    """
    L = np.array(NE*[1]+NI*[0])
    np.random.shuffle(L)
    E_indices = np.where(L == 1)[0]
    I_indices = np.where(L == 0)[0]

    T0 = np.zeros(N)

    T0[E_indices] = np.random.uniform(0.3, 0.5, NE)
    T0[I_indices] = np.random.uniform(0, 1, NI)

    EI =  list(itertools.product(E_indices, I_indices))
    IE =  list(itertools.product(I_indices, E_indices))
    EE = list(itertools.product(E_indices, E_indices))
    II = list(itertools.product(I_indices, I_indices))

    EI_rows = tuple([EI[i][0] for i in range(len(EI))])
    EI_cols = tuple([EI[i][1] for i in range(len(EI))])

    IE_rows = tuple([IE[i][0] for i in range(len(IE))])
    IE_cols = tuple([IE[i][1] for i in range(len(IE))])

    EE_rows = tuple([EE[i][0] for i in range(len(EE))])
    EE_cols = tuple([EE[i][1] for i in range(len(EE))])

    II_rows = tuple([II[i][0] for i in range(len(II))])
    II_cols = tuple([II[i][1] for i in range(len(II))])

    #inhibitory = 0, excitatory = 1
    J0 = np.random.uniform(0, 1, (N, N))
    #no connections of each neuron with itself
    np.fill_diagonal(J0, 0)
    #no inhibitory/inhibitory connections:
    J0[II_rows, II_cols] = 0

    #EE connections sum to 1
    J0[EE_rows, EE_cols] = \
        J0[EE_rows, EE_cols]/np.sum(J0[EE_rows, EE_cols])

    #IE connections sum to 1
    J0[IE_rows, IE_cols] = \
        J0[IE_rows, IE_cols]/np.sum(J0[IE_rows, IE_cols])

    #EI connections sum to 1
    J0[EI_rows, EI_cols] = \
        J0[EI_rows, EI_cols]/np.sum(J0[IE_rows, IE_cols])

    return (
        E_indices, I_indices, EE_rows, EE_cols, EI_rows, EI_cols, 
        IE_rows, IE_cols, J0, T0
    )



def nextstep(state,output):
    """takes state and previous output (input for next step),
        and returns next state and ouput
    """
    if state == "A":
        output = np.random.choice([0,1], p = prob_array)
        
        if output == 0:
            newstate = "A"
        elif output == 1:
            newstate = "B"

    elif state == "B":
        output = 1
        newstate = "A"

    return newstate, output


def get_stimulivals(prob_array):
    """Output stimuli values for simulation using 2 state 
    hidden Markov Model with transition probabilities from A to A
    and A to B, respectively, from prob_array. A -> A transition outputs 0,
    A -> B transition outputs 1, and B -> A transition outputs 1. Outputs
    are the stimuli values.
    """
    state = np.random.choice(["A","B"])
    state_list = [state]
    A_indices = []
    B_indices = []
    zero_indices = []
    one_indices = []

    if state == "A":
        output = np.random.choice([0,1], p = prob_array)
        A_indices.append(0)
    elif state == "B":
        output = 1
        B_indices.append(0)

    stimuli = [output]

    if output == 0:
        zero_indices.append(0)
    elif output == 1:
        one_indices.append(0)

    #neuron firing at time 2 corresponds to stimuli at time 1 & state at time 0
    n = 1
    for t in range(Trials):
        state, output = nextstep(state, output)

        if state == "A":
            A_indices.append(n)
        else:
            B_indices.append(n)

        stimuli.append(output)
        state_list.append(state)

        if output == 0:
            zero_indices.append(n)
        elif output == 1:
            one_indices.append(n)
            
        n += 1

    

    return(A_indices, B_indices, zero_indices, one_indices, stimuli)


#############################################################################
# Set up simulation info

#NE is the number of excitatory neurons, min(NE) = 5
NE = 10
#choose transition probabilities from
# state A to A and states A to B, respectively
prob_array = [1/2, 1/2]

#number of inhibitory neurons
NI = math.ceil(0.2*NE)
#total number of neurons
N = NE + NI


#run at least 1,000 trails
Trials = 1500
#run the Trials this many times (Runs_of_trials)
Runs_of_trials = 2
Subtrials = list(range(Runs_of_trials))


#file naming information
Filerun = 1
D = date.today()
today = D.strftime("%m%d%Y")

#all pairwise combinations of neurons, include each neuron paired with itself
#I didn't keep using this, but a useful command for evaluating pairwise relationship
neuron_pairs = list(itertools.combinations_with_replacement(range(N), 2))

# #H values to loop through
Harray = np.array([0.1, 0.25, 0.5, 0.75])
# #Epsilon and alpha values to loop through
Earray = np.array([0.1, 0.25, 0.5, 0.75])
# # restating that the alpha values are equal to the epsilon values 
# #(Aarray and Earray are interchangeable)
Aarray = Earray

#Abbreviation for dimensions
LH = len(Harray)
LE = len(Aarray)
LA = len(Aarray)

#CHANGE in MSE and energy (learning rule - no learning rule)
# axis 1 = H, axis 2 = eps/alpha, axis 3 = [DeltaEnergy, DeltaMSE],
# axis 5 = trial number
DeltaMSEEnergyarray = np.zeros([LE, LH, 2, Runs_of_trials])


#MSE and energy for each run of Trials
#axis 0 = eps & alpha initial value, axis 1 = H, axis 2 = [LR, No LR],
# axis 3 = [Energy, MSE], axis 4 = output from run of Trials
#note: NOT *change* in MSE and energy
MSEEnergy = np.zeros([LE, LH, 2, 2, Runs_of_trials])

#To store sums of the connection weights
#axes: eps & alpha initial value, H, [LR, No LR], 
# subtrial, sum of connection weights
Jsums_all = np.zeros([LE, LH, Runs_of_trials, Trials+2])

#To store average firing rate after each timestep
#axes: eps & alpha initial value, H, [LR, No LR], subtrial, 
# avrg firing at each timestep for all runs of trials
avrgfiring_all = np.zeros([LE, LH, 2, Runs_of_trials, Trials+2])

#Sum of the excitatory firing thresholds after each timestep for 
# all runs of trials (only applicable with learning rule)
# axes: eps & alpha initial value, H, subtrial, sum of excitatory thresholds
avrgT_ex = np.zeros([LE, LH, Runs_of_trials, Trials+2])

# Epsilon value at each timestep for all runs of trials
#(only applicable with learning rule)
# axes: eps & alpha initial value, H, subtrial, epsilon value
epsvals = np.zeros([LE, LH, Runs_of_trials, Trials + 2])

# Alpha value at each timestep for all runs of trials
#(only applicable with learning rule)
# axes: eps & alpha initial value, H, subtrial, alpha value
alphavals = np.zeros([LE, LH, Runs_of_trials, Trials + 2])

#keep track of parameter number
count = 0
total_params = LE*LH

#keep track of simulation run time 
start_time = time.time()


#Run simulation
a = 0
for alpha in Aarray:
    b = 0
    for H in Harray:
        eps = alpha
        eta = 0.001
        H0 = H*np.ones(N)

        for T in Subtrials:

            #I didn't keep using A/B or 0/1 indices in my analysis,
            #But left them here in case they're of use
            A_indices, B_indices, zero_indices, one_indices, stimuli = \
                 get_stimulivals(prob_array)

            IV = initialvalues(NE, NI)

            E_indices, I_indices, EE_rows, \
                EE_cols, EI_rows, EI_cols, IE_rows, IE_cols, J0, T0 = IV


            foo2 = Hopfield_no_rules(N, E_indices, I_indices,
                                    stimuli,EE_rows, EE_cols, EI_rows, EI_cols, 
                                    IE_rows, IE_cols, J0, T0)

            foo1 = Hopfield_with_Lrules(N, E_indices,
                                        I_indices, stimuli, 
                                        EE_rows, EE_cols,
                                        EI_rows, EI_cols,
                                        IE_rows, IE_cols,
                                        J0, T0, eta, H0, eps, alpha)

            #store outputs (can condense this code, but it's fully written
            # out to avoid mistakes):

            #Energy LR
            MSEEnergy[a, b, 0, 0, T] = foo1[0]
            #MSE LR
            MSEEnergy[a, b, 0, 1, T] = foo1[1]

            #Energy NoLR
            MSEEnergy[a, b, 1, 0, T] = foo2[0]
            #MSE NoLR
            MSEEnergy[a, b, 1, 1, T] = foo2[1]

            #DeltaEnergy for all states
            DeltaMSEEnergyarray[a, b, 0, T] = foo1[0] - foo2[0]
            #DeltaMSE for all states
            DeltaMSEEnergyarray[a, b, 1, T] = foo1[1] - foo2[1]

            Jsums_all[a, b, T, :] = foo1[2]

            avrgfiring_all[a, b, 0, T, :] = foo1[3]

            avrgfiring_all[a, b, 1, T, :] = foo2[2]

            avrgT_ex[a, b, T, :] = foo1[4]

            epsvals[a, b, T, :] = foo1[5]

            alphavals[a, b, T, :] = foo1[6]
            
           
        count+=1

        # uncomment the following to keep track of simulation time per parameter:
        # print('{}/{} parameters completed'.format(count, total_params))
        # print("--- %s seconds ---" % (time.time() - start_time))
    
        b += 1
    a += 1


#save the data

outputID = '_{4}N_{0}x{1}k_{2}_Run{3}'.format(
            Runs_of_trials,Trials/1000, today, Filerun, N
            )
new_folder = 'Ouput{1}'.format(
    round(prob_array[0],2), round(prob_array[1],2)
    ) + outputID
newpath = os.getcwd() + '\\'+ new_folder

#change filename to with new filerun number if the file already exists
if os.path.exists(newpath):
    while os.path.exists(newpath) and Filerun < 20:
        Filerun += 1
        outputID = '_{4}N_{0}x{1}k_{2}_Run{3}'.format(
            Runs_of_trials,Trials, today, Filerun, N
            )
        new_folder = 'Output' + outputID
        newpath = os.getcwd() + '\\'+ new_folder
        try:
            os.mkdir(newpath)
        except OSError:
            print ("Creation of the directory %s failed" % newpath)
        else:
            print ("Successfully created the directory %s " % newpath)
            break
else:
    try:
        os.mkdir(newpath)
    except OSError:
        print ("Creation of the directory %s failed" % newpath)
    else:
        print ("Successfully created the directory %s " % newpath)



#save data
np.save(r".\{0}\DeltaMSEEnergyarray{1}.npy".format(
    new_folder, outputID), DeltaMSEEnergyarray)

np.save(r".\{0}\MSEEnergy{1}.npy".format(new_folder, outputID), MSEEnergy)

np.save(r".\{0}\Jsums{1}.npy".format(new_folder, outputID), Jsums_all)

np.save(r".\{0}\avrgfiring{1}.npy".format(new_folder, outputID), avrgfiring_all)

np.save(r".\{0}\avrgT_ex{1}.npy".format(new_folder, outputID), avrgT_ex)


np.save(r".\{0}\epsvals{1}.npy".format(new_folder, outputID), epsvals)

np.save(r".\{0}\alphavals{1}.npy".format(new_folder, outputID), alphavals)





##############################################################################
###########################################################################
#plot data


#all outputs go to 'new_folder' (created earlier in simulation to store data); 
# this creates new subfolders (Hfolders) for each H value; then I stored the 
# plots that were for specific trials like Jsums, firing frequency, 
# firing thresholds, or anything else for specific trials
for a in range(LH):
    Hfolder = 'H{}'.format(Harray[a])
    #os.getcwd() is your current directory
    newpath2 = os.getcwd() + '\\'+ new_folder + '\\' + Hfolder
    os.mkdir(newpath2)


#to plot Delta MSE & Delta Energy with scatter plots for various epsilon & alpha values 
#that have the same H value:

#make labels for scatter plot points (epsilon, alpha)
labels = []
for eps in Earray:
    labels.append('({0},{0})'.format(round(eps,3)))


#loop through values & make scatter plots
for b in range(LH):
    X = []
    Y = []
    for a in range(LE):
        X.append(list(DeltaMSEEnergyarray[a, b, 1, :]))
        Y.append(list(DeltaMSEEnergyarray[a, b, 0, :]))

    #note: make sure you close figures (e.g. plt.close('all')) when making lots of figures or 
    #your computer might crash from having too many figures open at once
    plt.close('all')
    fig, ax = plt.subplots(1, 1, figsize = (10,8), sharex = True)

    for x,y,lab in zip(X,Y,labels):
        ax.scatter(x,y,label=lab, s = 60, alpha = 0.8)

    colormap = plt.cm.gist_ncar\
    #define that you want the colors to be evenly spread out
    #along colormap spectrum for your number of datapoints
    colorst = [colormap(i) for i in np.linspace(0, 0.9,len(ax.collections))]
    for t,j1 in enumerate(ax.collections):
        j1.set_color(colorst[t])

    #create legend and label axes and title
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width*0.6, box.height])
    plt.rcParams.update({'font.size':14})
    plt.rc('legend', fontsize=12)
    plt.rc('figure', titlesize=16)
    ax.legend(loc = 'center left', bbox_to_anchor=(1, 0.5), \
            ncol = 2, title = 'Parameters (\u03F5, \u03B1)')
    plt.xlabel('\u0394MSE', fontsize = 'small')
    plt.ylabel('\u0394Energy', fontsize = 'small')
    plt.title('\u0394MSE all states vs \u0394Energy all states for 12 neurons\n \
        last 1,000 timesteps from {0} runs of {1} timesteps each \n \
        for the parameter H = {2} and varying \u03F5 and \u03B1'.format(
            Runs_of_trials, Trials, round(Harray[b],3))
            )

    Hfolder = 'H{}'.format(Harray[b])

    plt.savefig(r".\{0}\{1}\DeltaMSEvsEnergy_H{2}{3}.png".format(
        new_folder, Hfolder, Harray[b], outputID))
    plt.close(fig)

##############################################################################

#plots for thesis Figures 3 & 4

#NOTE: you could easily combine all of the following "looping through 
# individual trials" plotting into one for loop, but it doesn't 
# take long to run, and I sometimes want to copy & paste 
#and then run just one loop in the terminal

#plot sum of J values
for a in range(LE):
    for b in range(LH):
        
        plt.close('all')

        Hfolder = 'H{}'.format(Harray[b])

        #using PdfPages to save all of the following plots within next for loop
        #within the same PDF file
        with PdfPages(r".\{0}\{1}\Jsums_({2},{2},{3}){4}.pdf".format(
            new_folder, Hfolder, Earray[a], Harray[b], outputID
        )) as pdf:
            #loop through the data/labels/title for each plot
            for T in Subtrials:

                fig = plt.figure(figsize=(9,9))
                x = list(range(Trials+2))
                y = list(Jsums_all[a, b, T, :])
                plt.scatter(x,y, s = 20)
                plt.xlabel('Timestep')
                plt.ylabel('Sum of neuron weights')
                plt.title('Evolution of sum of neuron weights over {0} timesteps,\n\
                    trial {1} for 12 neurons. Parameters \u03F5 = {2}, \n\
                    \u03B1 = {3}, H = {4}. \u0394MSE = {5}, \u0394Energy = {6}'.format(
                    Trials, T, round(Aarray[b],3), round(Aarray[b],3), round(Harray[b],3),
                    round(DeltaMSEEnergyarray[a, b, 1, T],3), 
                    round(DeltaMSEEnergyarray[a, b, 0, T],3))
                    )

                #save the figure and close it
                pdf.savefig(fig)
                plt.close()
                plt.close(fig)



#plot average firing
for a in range(LE):
    for b in range(LH):
        
        plt.close('all')

        Hfolder = 'H{}'.format(Harray[b])

        with PdfPages(r".\{0}\{1}\avrgfiring_({2},{3},{4}){5}.pdf".format(
            new_folder, Hfolder, Aarray[a], Aarray[a], Harray[b], outputID
        )) as pdf:
            #loop through the data/labels/title for each plot
            for T in Subtrials:
                plt.close('all')

                fig = plt.figure(figsize=(9,9))
                x = list(range(Trials+2))
                y = list(avrgfiring_all[a, b, 0, T, :])
                plt.scatter(x,y, s = 5)
                plt.xlabel('Timestep')
                plt.ylabel('Average firing')
                plt.title('Evolution of average neuron firing over {0} timesteps,\n\
                    trial {1}, for 12 neurons. Parameters \u03F5 = {2}, \n\
                    \u03B1 = {3}, H = {4}. \u0394MSE = {5}, \u0394Energy = {6}'.format(
                    Trials, T, round(Aarray[a],3), round(Aarray[a],3), round(Harray[b],3),
                    round(DeltaMSEEnergyarray[a, b, 1, T],3), 
                    round(DeltaMSEEnergyarray[a, b, 0, T],3))
                    )

                #save the figure and close it
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close('all')
                plt.close(fig)



#plot changing firing threshold
for a in range(LA):
    for b in range(LH):
        
        plt.close('all')

        Hfolder = 'H{}'.format(Harray[b])

        with PdfPages(r".\{0}\{1}\avrgT_excitatory_({2},{3},{4}){5}.pdf".format(
            new_folder, Hfolder, Aarray[a], Aarray[a], Harray[b], outputID
        )) as pdf:
            #loop through the data/labels/title for each plot
            for T in Subtrials:

                fig = plt.figure(figsize=(10,10))
                x = list(range(Trials+2))
                y = list(avrgT_ex[a, b, T, :])
                plt.scatter(x,y, s = 10)
                plt.xlabel('Timestep')
                plt.ylabel('Average excitatory firing threshold')
                plt.title('Evolution of average excitatory fring threshold\n\
                    over {0} timesteps,trial {1}, for 12 neurons. \n\
                    Parameters \u03F5 = {2}, \u03B1 = {3}, H = {4}\n\
                    \u0394MSE = {5}, \u0394Energy = {6}'.format(
                    Trials, T, round(Aarray[a],3), round(Aarray[a],3), round(Harray[b],3),
                    round(DeltaMSEEnergyarray[a, b, 1, T],3), 
                    round(DeltaMSEEnergyarray[a, b, 0, T],3))
                    )

                pdf.savefig(fig)
                plt.close('all')
                plt.close(fig)

  


#plot changing alpa and epsilon
for a in range(LE):
    for b in range(LH):
        plt.close('all')

        Hfolder = 'H{}'.format(Harray[b])

        with PdfPages(r".\{0}\{1}\epsilon_({2},{3},{4}){5}.pdf".format(
            new_folder, Hfolder, Aarray[a], Aarray[a], Harray[b], outputID
        )) as pdf:
            #loop through the data/labels/title for each plot
            for T in Subtrials:

                fig = plt.figure(figsize=(10,10))
                x = list(range(Trials+2))
                y = list(epsvals[a, b, T, :])
                plt.scatter(x,y, s = 10)
                plt.xlabel('Timestep')
                plt.ylabel('Epsilon')
                plt.title('Epsilon over {0} timesteps, trial {1}, 12 neurons. \n\
                    Parameters \u03F5 = {2}, \u03B1 = {3}, H = {4}\n\
                    \u0394MSE = {5}, \u0394Energy = {6}'.format(
                    Trials, T, round(Aarray[b],3), round(Aarray[b],3), round(Harray[b],3),
                    round(DeltaMSEEnergyarray[a, b, 1, T],3), 
                    round(DeltaMSEEnergyarray[a, b, 0, T],3))
                    )

                pdf.savefig(fig)
                plt.close('all')
                plt.close(fig)

        with PdfPages(r".\{0}\{1}\alpha_({2},{3},{4}){5}.pdf".format(
            new_folder, Hfolder, Aarray[a], Aarray[a], Harray[b], outputID
        )) as pdf:
            #loop through the data/labels/title for each plot
            for T in Subtrials:

                fig = plt.figure(figsize=(10,10))
                x = list(range(Trials+2))
                y = list(alphavals[a, b, T, :])
                plt.scatter(x,y, s = 10)
                plt.xlabel('Timestep')
                plt.ylabel('Alpha')
                plt.title('Alpha over {0} timesteps, trial {1}, 12 neurons. \n\
                    Parameters \u03F5 = {2}, \u03B1 = {3}, H = {4}\n\
                    \u0394MSE = {5}, \u0394Energy = {6}'.format(
                    Trials, T, round(Aarray[a],3), round(Aarray[a],3), round(Harray[b],3),
                    round(DeltaMSEEnergyarray[a, b, 1, T],3), 
                    round(DeltaMSEEnergyarray[a, b, 0, T],3))
                    )

                pdf.savefig(fig)
                plt.close('all')
                plt.close(fig)


##############################################################################
#make plot like Figure 2 in my thesis (color-coded points in 3D grid)

#lIused colors from an xkcd color palette: https://xkcd.com/color/rgb/
#This is just a list of the colors for my data points
colornames = [mcd.XKCD_COLORS['xkcd:green'],
            mcd.XKCD_COLORS['xkcd:clear blue'],
            mcd.XKCD_COLORS['xkcd:fire engine red'],
            mcd.XKCD_COLORS['xkcd:black']]
colors = {}

for a in range(LA):
    for b in range(LH):
        DEnergy = np.average(DeltaMSEEnergyarray[a, b, 0, :])
        DMSE = np.average(DeltaMSEEnergyarray[a, b, 1, :])
        
        #assign colors based on MSE/Energy criteria
        if DEnergy < 0 and DMSE < 0:
            colors['({},{},{})'.format(a,a,b)] = colornames[0]
        
        elif DEnergy >= 0 and DMSE < 0:
            colors['({},{},{})'.format(a,a,b)] = colornames[1]

        elif DEnergy < 0 and DMSE >= 0:
            colors['({},{},{})'.format(a,a,b)] = colornames[2]

        elif DEnergy >= 0 and DMSE >= 0:
            colors['({},{},{})'.format(a,a,b)] = colornames[3]

        
fig = plt.figure(figsize = (10, 7))
ax = fig.add_subplot(111, projection='3d')


#x axis = eps, y axis = alpha, z axis = H
for a in range(LE):
    for b in range(LH):
        x = Earray[a]
        y = Aarray[a]
        z = Harray[b]
        col = colors['({},{},{})'.format(a, a, b)]
        ax.scatter(x, y, z, c = col, marker = 'o')

#'\u0394' = unicode delta symbol
k1 = '\u0394Energy'
k2 = '\u0394MSE'

#create legend labels that correspond to correct colors
#the unicode symbols are for equals to or greater than or equals to or less than
legend_elements = [Line2D([0], [0], marker='o', color='w', 
                        label='{0} < 0, {1} < 0'.format(k1, k2),
                          markerfacecolor= colornames[0], markersize=10),
                    Line2D([0], [0], marker='o', color='w', 
                        label='{0} \u2265 0, {1} < 0'.format(k1, k2),
                          markerfacecolor= colornames[1], markersize=10),
                    Line2D([0], [0], marker='o', color='w', 
                        label='{0} < 0, {1} \u2265 0'.format(k1, k2),
                          markerfacecolor= colornames[2], markersize=10),
                    Line2D([0], [0], marker='o', color='w', 
                        label='{0} \u2265 0, {1} \u2265 0'.format(k1, k2),
                          markerfacecolor= colornames[3], markersize=10)]

#position legend relative to plot
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width*0.7, box.height])
ax.legend(handles=legend_elements,
        loc = 'center left', bbox_to_anchor=(1, 0.5))
plt.title('Average \u0394MSE and \u0394Energy for different\n \
     hyperparameter values for 12 neurons', pad = 15)
plt.rcParams.update({'font.size':15})
plt.rc('legend', fontsize=14)   
plt.rc('figure', titlesize=16)
plt.rc('axes', labelsize=20)

ax.set_zlabel('H', labelpad=15)
plt.xlabel('\u03F5',fontsize=20, labelpad=15)
plt.ylabel('\u03B1', fontsize=20, labelpad=15)


#############################################################################################################


#create plot like thesis Figure 5 (very similar to code for figure 2)

#plot FINAL eps and alpha

colornames = [mcd.XKCD_COLORS['xkcd:green'],
            mcd.XKCD_COLORS['xkcd:clear blue'],
            mcd.XKCD_COLORS['xkcd:fire engine red'],
            mcd.XKCD_COLORS['xkcd:black']]
colors = {}

for a in range(LA):
    for b in range(LH):
        for T in range(Runs_of_trials):
            DEnergy = DeltaMSEEnergyarray[a, b, 0, T]
            DMSE = DeltaMSEEnergyarray[a, b, 1, T]
            
            if DEnergy < 0 and DMSE < 0:
                colors['({},{},{})'.format(a, b, T)] = colornames[0]
            
            elif DEnergy >= 0 and DMSE < 0:
                colors['({},{},{})'.format(a, b, T)] = colornames[1]

            elif DEnergy < 0 and DMSE >= 0:
                colors['({},{},{})'.format(a, b, T)] = colornames[2]

            elif DEnergy >= 0 and DMSE >= 0:
                colors['({},{},{})'.format(a, b, T)] = colornames[3]

        
fig = plt.figure(figsize = (10, 7))
ax = fig.add_subplot(111, projection='3d')


xvals = []
yvals = []
zvals = []
for a in range(LE):
    for b in range(LH):
        for T in range(Runs_of_trials):
            x = epsvals[a, b, T, -1]
            xvals.append(x)
            y = alphavals[a, b, T, -1]
            yvals.append(y)
            z = Harray[b]
            zvals.append(z)
            col = colors['({},{},{})'.format(a, b, T)]
            ax.scatter(x, y, z, c = col, marker = 'o')


k1 = '\u0394Energy'
k2 = '\u0394MSE'

legend_elements = [Line2D([0], [0], marker='o', color='w', 
                        label='{0} < 0, {1} < 0'.format(k1, k2),
                          markerfacecolor= colornames[0], markersize=10),
                    Line2D([0], [0], marker='o', color='w', 
                        label='{0} \u2265 0, {1} < 0'.format(k1, k2),
                          markerfacecolor= colornames[1], markersize=10),
                    Line2D([0], [0], marker='o', color='w', 
                        label='{0} < 0, {1} \u2265 0'.format(k1, k2),
                          markerfacecolor= colornames[2], markersize=10),
                    Line2D([0], [0], marker='o', color='w', 
                        label='{0} \u2265 0, {1} \u2265 0'.format(k1, k2),
                          markerfacecolor= colornames[3], markersize=10)]


box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width*0.7, box.height])
ax.legend(handles=legend_elements,
        loc = 'center left', bbox_to_anchor=(1, 0.5))
plt.title('Average \u0394MSE and \u0394Energy for different\n \
     hyperparameter values for 12 neurons, 25 subtrials', pad = 15)
plt.rcParams.update({'font.size':15})
plt.rc('legend', fontsize=14)   
plt.rc('figure', titlesize=16)
plt.rc('axes', labelsize=20)

ax.set_zlabel('H', labelpad=15)
plt.xlabel('\u03F5 final',fontsize=20, labelpad=15)
plt.ylabel('\u03B1 final', fontsize=20, labelpad=15)



###############################################

#code to make your plot a rotating animation and save it as an mp4 file! 
#You can use it to animate any 3d plots
#To do this you have to dowload (pip install works) and then 
# import the following packages/functions:
    #from matplotlib.animation import FuncAnimation
    #import matplotlib.animation as animation
#I also had to download the "ffmpeg" package (video processing tool that 
# encodes the animation) & explicitly set the path to it 
# (plt.rcParams['animation.ffmpeg_path']) and specify that I was 
# using it as the video "writer"

#the code:

from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

# set intial view angle
ax.view_init(30, 0)

def update(angle):
   ax.view_init(30, angle + 1)
   return ax

#                                         #put your own path here to location of that file
plt.rcParams['animation.ffmpeg_path'] = 'C:\\Users\\taubm\\Anaconda3\\Library\\bin\\ffmpeg.exe'

# view angle rotates 360 degrees w/ update(angle) function
anim = FuncAnimation(fig, update, frames = np.arange(0, 360),
interval = 80)


mywriter = animation.FFMpegWriter()
anim.save('DeltaMSEEnergy_eps_alpha_final{}.mp4'.format(outputID),writer=mywriter)


