
Hippocampus memory model with forgetting mechanism bioinspired DG and CA1 layers
================================================================================

This sections shows a hippocampus memory model with forgetting mechanism (Memory class in hippocampus_bioinspired_dg_ca1).


Theoretical model
-----------------

This memory is identical to `hippocampus_with_forgetting <hippocampus_with_forgetting.html>`_ but with a more bio-inspired design and implementation of the DG and CA1 layers that does not depend on the `sPyBlocks <https://github.com/alvayus/spyblocks>`_ library.


Class information
-----------------

.. automodule:: sPyMem.hippocampus_bioinspired_dg_ca1.hippocampus_bioinspired_dg_ca1
   :members:
   :undoc-members:
  
How to use the model
---------------------

To integrate the memory model in your own network, just import the model class and instantiate it:

.. code-block::
	
	from sPyMem.hippocampus_bioinspired_dg_ca1 import hippocampus_bioinspired_dg_ca1

	memory = hippocampus_bioinspired_dg_ca1.Memory(cueSize, contSize, sim)
	memory.connect_in(ILayer)
	memory.connect_out(OLayer)

The full example can be found at `sPyMem Github <https://github.com/dancasmor/sPyMem>`_, and for other examples see `Test and applications <Test.html>`_ section.

This memory model is identical in functionality and use to the `hippocampus_with_forgetting <hippocampus_with_forgetting.html>`_ model. The main difference is that architecturally it is a more computationally efficient design, requiring fewer resources (neurons, synapses, etc.) and presents a slight temporal improvement. 

At the time level, after performing a learning operation, the waiting time to perform the next operation is still 7 time units, however, after performing a recall operation, the time to the next operation is 5 time units (1 time unit less than model hippocampus_with_forgetting).

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



