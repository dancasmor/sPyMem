
CA3 content addressable memory model with forgetting mechanism
==============================================================

This sections shows a hippocampus memory model with forgetting mechanism and addressable by address and content (Memory class in ca3_content_addressable).

Theoretical model
-----------------

This hippocampal memory model builds on the foundations of the `CA3 <hippocampus_with_forgetting.html>`_ model to address content from the addresses to which they are associated and extends it to address addresses from content. In other words, from a binary content input (each bit/neuron of the content is made up of 0's or no pulse and 1's or neuron activation), the memory returns in which addresses it has as content at least one 1 (or active neuron) in one of those positions.

(In progress to publish this new model ...)


Class information
-----------------

.. automodule:: sPyMem.CA3_content_addressable.CA3_content_addressable
   :members:
   :undoc-members:
  
How to use the model
---------------------

To integrate the memory model in your own network, just import the model class and instantiate it:

.. code-block::
	
	from sPyMem.CA3_content_addressable import CA3_content_addressable
	
	memory = CA3_content_addressable.Memory(cueSize, contSize, sim, ILayer, OLayer)

The full example can be found at `sPyMem Github <https://github.com/dancasmor/sPyMem>`_, and for other examples see `Test and applications <Test.html>`_ section.

This model of memory can perform the 3 basic operations: learning memories, reacalling learned memories and forgetting them. Forgetting will take place automatically when an attempt is made to learn a memory with the same cue as a previously stored memory. The main difference with CA3 model is that it has two types of recall operations. The first is to recall the content from a cue, for which a cue or address is passed and the memory returns the rest of the content associated with that address. The second is to recall which addresses are associated with a piece of content. To do this, a piece of content is passed and the memory returns those addresses that match at least one 1 of the content with which they are associated.

In order to carry out learning and recall operations in this model, it is necessary to consider the following. For learning operations, spikes need to be held for 3 time units at the input of the memory and no further operation can be performed until 7 time units later. In the case of recall operations, spikes must be displayed for a single time unit and 6 time units must be waited until the next operation.

When performing a learning operation, the network stores a memory and, 5 (and 7) time units after having started the operation, the memory returns the learned memory to its output. In the case of a recall operation, after 5 time units the cue used to start the operation will appear at the memory output and one time unit later the rest of the memory.

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


