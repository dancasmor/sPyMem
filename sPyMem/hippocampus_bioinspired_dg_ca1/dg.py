import math


class DG:
    """Spike-based DG one-hot encoding model

       :param size: number of neuron of DGLayer, i.e., number of one-hot codes
       :type size: int
       :param sim: object in charge of handling the simulation
       :type sim: simulation object (spynnaker8 for spynnaker)
       :param neuronParameters: all neuron parameters of each population (for more information see `Custom config files`_)
       :type neuronParameters: dict
       :param initNeuronParameters: init membrane potential of each population (for more information see `Custom config files`_)
       :type initNeuronParameters: dict
       :param synParameters: all synapses parameters of each synapse group (for more information see `Custom config files`_)
       :type synParameters: dict

       :ivar size: number of neuron of DGLayer, i.e., number of one-hot codes
       :vartype size: int
       :ivar inSize: number of input neurons to the DG model, calculate based on the size of DGLayer, i.e., number of binary digits
       :vartype inSize: int
       :ivar sim: object in charge of handling the simulation, initial value: sim
       :vartype sim: simulation object (spynnaker8 for spynnaker)
       :ivar DGLayer: DG population of the model, initial value: DGLayer
       :vartype DGLayer: population
       :ivar popNeurons: dict that contains the number of neuron of each population, at the input interface level - {"ILayer": ilInputSize, "DGLayer": dgInputSize, "CA3cueLayer": self.cueSize, "CA3contLayer": self.contSize, "CA1Layer": self.cueSize, "OLayer": ilInputSize}
       :vartype popNeurons: dict
       :ivar neuronParameters: all neuron parameters of each population (for more information see `Custom config files`_)
       :vartype neuronParameters: dict
       :ivar initNeuronParameters: init membrane potential of each population (for more information see `Custom config files`_)
       :vartype initNeuronParameters: dict
       :ivar synParameters: all synapses parameters of each synapse group (for more information see `Custom config files`_)
       :vartype synParameters: dict
    """
    def __init__(self, size, sim, neuronParameters, initNeuronParameters, synParameters):
        """Constructor method
        """
        self.size = int(size)
        self.inSize = int(math.ceil(math.log2(size+1)))
        self.sim = sim
        self.neuronParameters = neuronParameters
        self.initNeuronParameters = initNeuronParameters
        self.synParameters = synParameters

        # Create populations
        self.create_population()
        # Create synapses
        self.create_synapses()

    def create_population(self):
        """Create all populations of the DG model

            :returns:
        """
        # DG
        self.DGLayer = self.sim.Population(self.size, self.sim.IF_curr_exp(**self.neuronParameters["DGL"]), label="DGLayer")
        self.DGLayer.set(v=self.initNeuronParameters["DG"]["vInit"])

    def create_synapses(self):
        """Create all synapses of the DG model

            :returns:
        """
        # DG-DG inhibitoy statis all to all (except with itself)
        self.sim.Projection(self.DGLayer, self.DGLayer, self.sim.AllToAllConnector(allow_self_connections=False),
                            synapse_type=self.sim.StaticSynapse(weight=self.synParameters["DGL-DGL"]["initWeight"],
                                                                delay=self.synParameters["DGL-DGL"]["delay"]),
                            receptor_type=self.synParameters["DGL-DGL"]["receptor_type"])

    def connect_in(self, ILayer, synInExcParameters, synInInhParameters):
        """Create synapses that connect the DG model with an input layer

            :param ILayer: input population to the DG model
            :type ILayer: population
            :param synInExcParameters: IL-DGL-exc synapses parameters (for more information see `Custom config files`_)
            :type synInExcParameters: dict
            :param synInInhParameters: IL-DGL-inh synapses parameters (for more information see `Custom config files`_)
            :type synInInhParameters: dict

            :returns:
        """
        # Calculate v diff between v threslhold and v rest + 0.5 (ensuring get to the threshold)
        vdiff = self.neuronParameters["DGL"]["v_thresh"] - self.neuronParameters["DGL"]["v_reset"] + 0.5
        # in-DG: (in_i-dg_j) i_exc are the binary digit of j equals to 1 and i_inh the digit equals to 0
        for dgID in range(1, self.size + 1):
            # Create synapses relation binary digits of dg neuron equals to 1 is exc syn and equals to 0 is inh syn
            #   - Get binary representation
            binaryID, inExcNeuron, inInhNeuron = [], [], []
            self.decimal_to_binary(dgID, binaryID)
            #   - Add 0's to left to get same length as number of inputs neurons
            if len(binaryID) < self.inSize:
                binaryID = binaryID + [0] * (self.inSize - len(binaryID))
            #   - Assign 0's digits to inh and 1's digits to exc
            for i in range(len(binaryID)):
                if binaryID[i] == 1:
                    inExcNeuron.append(i)
                else:
                    inInhNeuron.append(i)

            # + in-DGL-exc: excitatory synapses equals binary representation
            for inID in inExcNeuron:
                self.sim.Projection(self.sim.PopulationView(ILayer, [inID]),
                                    self.sim.PopulationView(self.DGLayer, [dgID - 1]),
                                    self.sim.AllToAllConnector(allow_self_connections=True),
                                    synapse_type=self.sim.StaticSynapse(weight=vdiff / len(inExcNeuron),
                                                                        delay=synInExcParameters["delay"]),
                                    receptor_type=synInExcParameters["receptor_type"])

            # + in-DGoL-inh: inhibitory synapses the rest of in neurons
            inInhNeuron = [id for id in range(self.inSize) if id not in inExcNeuron]
            for inID in inInhNeuron:
                self.sim.Projection(self.sim.PopulationView(ILayer, [inID]),
                                    self.sim.PopulationView(self.DGLayer, [dgID - 1]),
                                    self.sim.AllToAllConnector(allow_self_connections=True),
                                    synapse_type=self.sim.StaticSynapse(weight=vdiff, delay=synInInhParameters["delay"]),
                                    receptor_type=synInInhParameters["receptor_type"])

    def connect_out(self, OLayer, synOutParameters):
        """Create synapses that connect the DG model with an output layer

            :param OLayer: output population of the DG model
            :type OLayer: population
            :param synOutParameters: DGL-OL synapses parameters (for more information see `Custom config files`_)
            :type synOutParameters: dict

            :returns:
        """
        # DG-out: 1 to 1 excitatory
        self.sim.Projection(self.DGLayer, OLayer, self.sim.OneToOneConnector(),
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
