
Hippocampus memory model with forgetting mechanism
==================================================

This sections shows a hippocampus memory model with forgetting mechanism (Memory class in hippocampus_with_forgetting).

Theoretical model
-----------------

This memory model comes from the paper entitled: "A bio-inspired implementation of a sparse-learning spike-based hippocampus memory model", which can be found `here <https://arxiv.org/abs/2206.04924>`_.

To refer to this particular model:

**APA**: Casanueva-Morato, D., Ayuso-Martinez, A., Dominguez-Morales, J. P., Jimenez-Fernandez, A., & Jimenez-Moreno, G. (2024). A bio-inspired implementation of a sparse-learning spike-based hippocampus memory model. IEEE Transactions on Emerging Topics in Computing.

**ISO 690**: CASANUEVA-MORATO, Daniel, et al. A bio-inspired implementation of a sparse-learning spike-based hippocampus memory model. IEEE Transactions on Emerging Topics in Computing, 2024.

**MLA**: Casanueva-Morato, Daniel, et al. "A bio-inspired implementation of a sparse-learning spike-based hippocampus memory model." IEEE Transactions on Emerging Topics in Computing (2024).

**BIBTEX**: @article{casanueva2024bio, title={A bio-inspired implementation of a sparse-learning spike-based hippocampus memory model}, author={Casanueva-Morato, Daniel and Ayuso-Martinez, Alvaro and Dominguez-Morales, Juan P and Jimenez-Fernandez, Angel and Jimenez-Moreno, Gabriel}, journal={IEEE Transactions on Emerging Topics in Computing}, year={2024}, publisher={IEEE}}


Class information
-----------------

.. automodule:: sPyMem.hippocampus_with_forgetting.hippocampus_with_forgetting
   :members:
   :undoc-members:
  
How to use the model
---------------------

To integrate the memory model in your own network, just import the model class and instantiate it:

.. code-block::
	
	from sPyMem.hippocampus_with_forgetting import hippocampus_with_forgetting
	
	memory = hippocampus_with_forgetting.Memory(cueSize, contSize, sim, ILayer, OLayer)

The full example can be found at `sPyMem Github <https://github.com/dancasmor/sPyMem>`_, and for other examples see `Test and applications <Test.html>`_ section.

This model of memory can perform 3 basic operations: learning memories, reacalling learned memories and forgetting them. Forgetting will take place automatically when an attempt is made to learn a memory with the same cue as a previously stored memory.

In order to carry out learning and recall operations in this model, it is necessary to consider the following. For learning operations, spikes need to be held for 3 time units at the input of the memory and no further operation can be performed until 7 time units later. In the case of recall operations, spikes must be displayed for a single time unit and 6 time units must be waited until the next operation.

When performing a learning operation, the network stores a memory and, 8 time units after having started the operation, the memory returns the learned memory to its output. In the case of a recall operation, after 6 time units the cue used to start the operation will appear at the memory output and one time unit later the rest of the memory.

For more information on this temporality, principles of operation, internal functioning, ... read the paper.

Custom config files
-------------------

When the memory model is instantiated, if no value is passed to the configFilePath variable, it will take the default configuration file. This can be found in the `sPyMem Github <https://github.com/dancasmor/sPyMem>`_ repository.

This file contains all the parameters that define the neuron models and the initial state of the neurons of each population, as well as all the parameters necessary to create the synapses between populations. Among these synapses, there are also those connecting the input and output populations of the model. 

Although changes to any internal network parameters are discouraged as the results of the network could be unpredictable, the user is allowed to define a different configuration file. This could be done by downloading the default configuration file, changing the desired parameters and passing the complete path to it as the network creation parameter. In this way, the network will take the internal parameters indicated in the new configuration file. 

The file format is a json file with 3 main fields:

* **neuronParameters**: parameters of the LIF neuron model used for each population.

* **initNeuronParameters**: initial conditions at the level of membrane potential of the neurons in each population.

* **synParameters**: internal parameters of the synapse models used for each set of connections between populations.


