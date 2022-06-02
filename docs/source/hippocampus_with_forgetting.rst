
Hippocampus memory model with forgetting mechanism
==================================================

This sections shows the hippocampus memory model with forgetting mechanism (Memory class in hippocampus_with_forgetting).

Theoretical model
-----------------

This memory model comes from the paper entitled: "A bio-inspired implementation of a sparse-learning spike-based hippocampus memory model", which can be found here (pending acceptance and publication). 

To refer to this particular model:
(pending acceptance and publication)

Class information
-----------------

.. automodule:: sPyMem.hippocampus_with_forgetting
   :members:
   :undoc-members:
  
How to use the model
---------------------

To integrate the memory model in your own network, just import the model class and instantiate it:

.. code-block::
	
	from sPyMem import hippocampus_with_forgetting
	
	memory = hippocampus_with_forgetting.Memory(cueSize, contSize, sim, ILayer, OLayer)

The full example can be found at `sPyMem Github <https://github.com/dancasmor/sPyMem>`_, and for other examples see `Test and applications <Test.html>`_ section.

Custom config files
-------------------

When the memory model is instantiated, if no value is passed to the configFilePath variable, it will take the default configuration file. This can be found in the `sPyMem Github <https://github.com/dancasmor/sPyMem>`_ repository.

This file contains all the parameters that define the neuron models and the initial state of the neurons of each population, as well as all the parameters necessary to create the synapses between populations. Among these synapses, there are also those connecting the input and output populations of the model. 

Although changes to any internal network parameters are discouraged as the results of the network could be unpredictable, the user is allowed to define a different configuration file. This could be done by downloading the default configuration file, changing the desired parameters and passing the complete path to it as the network creation parameter. In this way, the network will take the internal parameters indicated in the new configuration file. 

The file format is a json file with 3 main fields:

* **neuronParameters**: parameters of the LIF neuron model used for each population.

* **initNeuronParameters**: initial conditions at the level of membrane potential of the neurons in each population.

* **synParameters**: internal parameters of the synapse models used for each set of connections between populations.


