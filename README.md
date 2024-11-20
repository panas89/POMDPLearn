# POMDPLearn
A library to learn MDP and POMDP models using a single dataset

More information on what are [POMDPs](http://www.pomdp.org/) and [MDPs](https://mpatacchiola.github.io/blog/2016/12/09/dissecting-reinforcement-learning.html) follow the links.


# Installation & Setup

- Pull repo
- Preferably use linux os.
- Install octave from ubuntuSoftware
- Intall oct2py python module using pip


# use ubuntu 18.04

1. use docker 
```
sudo apt update
sudo apt install docker.io
docker run -it --name my-ubuntu-container ubuntu:18.04 bash
```
--name my-ubuntu-container: Assigns a name to the container (my-ubuntu-container) for easier management later.
No --rm: Ensures the container persists after you exit.

2. Inside the container, install Octave 4.2.2:

```
apt update
apt install octave=4.2.2-1ubuntu1
octave --version
```

3. Use miniconda

```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda3_portable
rm Miniconda3-latest-Linux-x86_64.sh
~/miniconda3/bin/conda init
source ~/.bashrc
conda activate
```

4. Create conda env

```
conda create --name POMDPLearn python=3.7.3
```

5. BNT library uses mex (source wikipedia:A MEX file is a type of computer file that provides an interface between MATLAB or Octave and functions written in C, C++ or Fortran. It stands for "MATLAB executable".)

- On bash select files to mex certain .c files

- run following commands.

```
sudo apt install liboctave-dev
cd ~/POMDPLearn/BNT/BNT/potentials/Tables/
mkoctfile -mex marg_table.c
mkoctfile -mex mult_by_table.c
mkoctfile -mex divide_by_table.c
```

## Dependencies



```
pip install -r requirements.txt
```

# Usage 


## Learning an MDP or POMDP model


### Steps:

#### Data preprocessing

- The state of the dataset to be used by the POMDPLearn library must satisfy the following criteria
    - States,action, and observations must be separate colummns with the keyword "state_", "action_", "obs_" followed by the number of the epoch in the horizon.
    
#### Dataset and Model definition

##### Dataset

- The preprocessed dataset will be input to the MDP or POMDP dataset object
- Using this dataset object we define an MDP or POMDP object

##### MDP or POMDP model definition

- By calling the MDP class and passing the pandas dataframe of our MDP dataset we automatically instantiate an MDP dataset,similarly for the POMDP class.

```
mdpModel = MDP(df=dfMDP_dataset)
```

#### Training and solving the MDP or POMDP model

##### MDP

- Using the trainMDP() method of the MDP class we train the MDP model

- Using the MDPsolve() method we solve the MDP model using value iteration

##### POMDP

- Using the trainPOMDP() method of the MDP class we train the MDP model

- Using the POMDPsolve() method we solve the MDP model using value iteration

#### Policy execution

##### MDP

- The policy obtained using the MDPsolve() method can be executed for any initial state using the policyExecution() method

##### POMDP

- Define an initial beleif model to generate initial beliefs using baseline features
- Using the getRecActions() method we get the policy that we should follow over the horizon.


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.