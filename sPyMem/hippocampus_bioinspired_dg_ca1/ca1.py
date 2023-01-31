import math


class CA1:
    """Spike-based CA1 binary encoding model

       :ivar inSize: number of input neurons to the CA1 model, i.e., number of one-hot digits
       :vartype inSize: int
       :param sim: object in charge of handling the simulation
       :type sim: simulation object (spynnaker8 for spynnaker)
       :param neuronParameters: all neuron parameters of each population (for more information see `Custom config files`_)
       :type neuronParameters: dict
       :param initNeuronParameters: init membrane potential of each population (for more information see `Custom config files`_)
       :type initNeuronParameters: dict

       :ivar size: number of neuron of CA1Layer, i.e., number of binary digits
       :vartype size: int
       :ivar inSize: number of input neurons to the CA1 model, i.e., number of one-hot digits
       :vartype inSize: int
       :ivar sim: object in charge of handling the simulation, initial value: sim
       :vartype sim: simulation object (spynnaker8 for spynnaker)
       :ivar CA1Layer: CA1 population of the model, initial value: CA1Layer
       :vartype CA1Layer: population
       :ivar popNeurons: dict that contains the number of neuron of each population, at the input interface level - {"ILayer": ilInputSize, "DGLayer": dgInputSize, "CA3cueLayer": self.cueSize, "CA3contLayer": self.contSize, "CA1Layer": self.cueSize, "OLayer": ilInputSize}
       :vartype popNeurons: dict
       :ivar neuronParameters: all neuron parameters of each population (for more information see `Custom config files`_)
       :vartype neuronParameters: dict
       :ivar initNeuronParameters: init membrane potential of each population (for more information see `Custom config files`_)
       :vartype initNeuronParameters: dict
    """
    def __init__(self, inSize, sim, neuronParameters, initNeuronParameters):
        """Constructor method
        """
        self.size = int(math.ceil(math.log2(inSize+1)))
        self.inSize = int(inSize)
        self.sim = sim
        self.neuronParameters = neuronParameters
        self.initNeuronParameters = initNeuronParameters

        # Create the network
        self.create_population()

    def create_population(self):
        """Create all populations of the CA1 model

            :returns:
        """
        # CA1
        self.CA1Layer = self.sim.Population(self.size, self.sim.IF_curr_exp(**self.neuronParameters["CA1L"]), label="CA1Layer")
        self.CA1Layer.set(v=self.initNeuronParameters["CA1"]["vInit"])

    def connect(self, ILayer, OLayer, synInParameters, synOutParameters):
        """Create synapses that connect the CA1 model with an input and output layer

            :returns:
        """
        # in-CA1: (in_i-ca1_j) excitatory, i in binary indicate to which j is connected
        for inID in range(1, self.inSize + 1):
            # Get binary representation
            binaryID, ca1Neuron = [], []
            self.decimal_to_binary(inID, binaryID)
            # Assign 1's digits to ca1 neurons
            for i in range(len(binaryID)):
                if binaryID[i] == 1:
                    ca1Neuron.append(i)
            # Make input synapses
            for ca1ID in ca1Neuron:
                self.sim.Projection(self.sim.PopulationView(ILayer, [inID - 1]),
                                    self.sim.PopulationView(self.CA1Layer, [ca1ID]),
                                    self.sim.AllToAllConnector(allow_self_connections=True),
                                    synapse_type=self.sim.StaticSynapse(
                                        weight=synInParameters["initWeight"],
                                        delay=synInParameters["delay"]),
                                    receptor_type=synInParameters["receptor_type"])

        # CA1-out: 1 to 1 excitatory
        self.sim.Projection(self.CA1Layer, OLayer, self.sim.OneToOneConnector(),
                            synapse_type=self.sim.StaticSynapse(weight=synOutParameters["initWeight"], delay=synOutParameters["delay"]),
                            receptor_type=synOutParameters["receptor_type"])

    def decimal_to_binary(self, num, list):
        """Given a number, obtains its binary representation in a list of 0s and 1s

            :param num: number to get binary representation
            :type num: int
            :param list: list to store the binary representation of num
            :type list: list

            :returns:
        """
        if num > 1:
            self.decimal_to_binary(num // 2, list)
        list.insert(0, num % 2)