
import json
import os


"""
CA3 with one-hot cue

+ Population:
    + Input: memory input with cue part in one-hot
    + CA3cue: store direction/cue of memories in one-hot codification
    + CA3cont: store content of memories
    + Output: output of the network

+ Synapses: 
    + Input-CA3cue: 1 to 1 excitatory and static (first n bits: corresponding to the one-hot cue of memories)
    + Input-CA3cont: 1 to 1 excitatory and static (the rest of the bits)
    + CA3cue-CA3cont: all to all excitatory and dinamic (STDP).
    + CA3cue-Output: 1 to 1 excitatory and static
    + CA3cont-Output: 1 to 1 excitatory and static
"""


class Memory:
    """Spike-based bio-inspired hippocampal (CA3 only) memory model with forgetting

       :param cueSize: number of cues of the memory
       :type cueSize: int
       :param contSize: size of the content of the memory in bits/neuron
       :type contSize: int
       :param sim: object in charge of handling the simulation
       :type sim: simulation object (spynnaker8 for spynnaker)
       :param ILayer: input population to the memory model
       :type ILayer: population
       :param OLayer: output population of the memory model
       :type OLayer: population
       :param configFilePath: path + filename to the config file of internal model parameters
       :type configFilePath: int, optional
       :param initCA3W: list of initial weight to use in CA3 synapse (initial memory content); format of each element of the list: (source_neuron_id, destination_neuron_id, initial_weight, delay)
       :type initCA3W: list, optional

       :ivar cueSize: number of cues of the memory, initial value: cueSize
       :vartype cueSize: int
       :ivar contSize: size of the content of the memory in bits/neuron, initial value: contSize
       :vartype contSize: int
       :ivar sim: object in charge of handling the simulation, initial value: sim
       :vartype sim: simulation object (spynnaker8 for spynnaker)
       :ivar ILayer: input population to the memory model, initial value: ILayer
       :vartype ILayer: population
       :ivar CA3cueLayer: CA3cue population
       :vartype CA3cueLayer: population
       :ivar CA3contLayer: CA3cont population
       :vartype CA3contLayer: population
       :ivar OLayer: output population of the memory model, initial value: OLayer
       :vartype OLayer: population
       :ivar configFilePath: path + filename to the config file of internal model parameters, initial value: configFilePath or internal path to default config file
       :vartype configFilePath: str
       :ivar initCA3W: list of initial weight to use in CA3 synapse (initial memory content); format of each element of the list: (source_neuron_id, destination_neuron_id, initial_weight, delay), initial value: None or input class parameter
       :vartype initCA3W: list
       :ivar popNeurons: dict that contains the number of neuron of each population, at the input interface level - {"ILayer": ilInputSize, "DGLayer": dgInputSize, "CA3cueLayer": self.cueSize, "CA3contLayer": self.contSize, "CA1Layer": self.cueSize, "OLayer": ilInputSize}
       :vartype popNeurons: dict
       :ivar neuronParameters: all neuron parameters of each population (for more information see `Custom config files`_)
       :vartype neuronParameters: dict
       :ivar initNeuronParameters: init membrane potential of each population (for more information see `Custom config files`_)
       :vartype initNeuronParameters: dict
       :ivar synParameters: all synapses parameters of each synapse group (for more information see `Custom config files`_)
       :vartype synParameters: dict
       :ivar IL_CA3contL_conn: IL-CA3cont synapses
       :vartype IL_CA3contL_conn: synapse
       :ivar IL_CA3cueL_conn: IL-CA3cue synapses
       :vartype IL_CA3cueL_conn: synapse
       :ivar CA3cueL_CA3contL_conn: CA3cue-CA3cont synapses (STDP)
       :vartype CA3cueL_CA3contL_conn: synapse
       :ivar CA3cueL_OL_conn: CA3cue-OL synapses
       :vartype CA3cueL_OL_conn: synapse
       :ivar CA3contL_OL_conn: CA3cont-OL synapses
       :vartype CA3contL_OL_conn: synapse
    """
    def __init__(self, cueSize, contSize, sim, ILayer, OLayer, initCA3W=None, configFilePath=None):
        """Constructor method
        """
        # Storing parameters
        self.cueSize = cueSize
        self.contSize = contSize
        self.sim = sim
        self.ILayer = ILayer
        self.OLayer = OLayer

        if configFilePath == None:
            self.configFilePath = os.path.dirname(__file__) + "/config/network_config.json"
        else:
            self.configFilePath = os.getcwd() + "/" + configFilePath

        self.initCA3W = initCA3W

        # Open configurations files to get the parameters
        self.open_config_files()
        # Create the network
        self.create_population()
        self.create_synapses()

    def read_json(self):
        """Open json file

            :raises: :class:`NameError`: path to config file not found

            :returns: the json data as a dict
            :rtype: dict
        """
        try:
            file = open(self.configFilePath)
            return json.load(file)
        except FileNotFoundError:
            raise NameError(str(self.configFilePath) + " -  path to config file not found")

    def open_config_files(self):
        """Open configuration json file with all the internal parameters needed by the network and assign parameters to variables

            :returns:
        """
        # + Calculated memory parameters
        # Size of IN population
        inputSize = self.cueSize + self.contSize
        # Number of neurons for each population
        self.popNeurons = {"ILayer": inputSize, "CA3cueLayer": self.cueSize, "CA3contLayer": self.contSize, "OLayer": inputSize}

        # + Network components parameters
        network_config = self.read_json()
        # Neurons paramaters
        self.neuronParameters = network_config["neuronParameters"]
        # Initial neuron parameters
        self.initNeuronParameters = network_config["initNeuronParameters"]
        # Synapses parameters
        self.synParameters = network_config["synParameters"]

    def create_population(self):
        """Create all populations of the memory model

            :returns:
        """
        # CA3cue
        self.CA3cueLayer = self.sim.Population(self.popNeurons["CA3cueLayer"], self.sim.IF_curr_exp(**self.neuronParameters["CA3cueL"]),
                                               label="CA3cueLayer")
        self.CA3cueLayer.set(v=self.initNeuronParameters["CA3cueL"]["vInit"])
        # CA3cont
        self.CA3contLayer = self.sim.Population(self.popNeurons["CA3contLayer"], self.sim.IF_curr_exp(**self.neuronParameters["CA3contL"]),
                                                label="CA3contLayer")
        self.CA3contLayer.set(v=self.initNeuronParameters["CA3contL"]["vInit"])

    def create_synapses(self):
        """Create all synapses of the memory model

            :returns:
        """
        # IL-CA3cueL -> 1 to 1, excitatory and static (first cueSize bits/neurons)
        self.IL_CA3cueL_conn = self.sim.Projection(self.sim.PopulationView(self.ILayer, range(0, self.popNeurons["CA3cueLayer"])), self.CA3cueLayer,
                                                   self.sim.OneToOneConnector(), synapse_type=self.sim.StaticSynapse(
                                                        weight=self.synParameters["IL-CA3cueL"]["initWeight"],
                                                  delay=self.synParameters["IL-CA3cueL"]["delay"]),
                                              receptor_type=self.synParameters["IL-CA3cueL"]["receptor_type"])

        # IL-CA3cont -> 1 to 1, excitatory and static (last m neurons of IL: only the number of cues to use)
        self.IL_CA3contL_conn = self.sim.Projection(self.sim.PopulationView(self.ILayer, range(self.popNeurons["CA3cueLayer"], self.popNeurons["ILayer"], 1)),
                                              self.CA3contLayer,
                                              self.sim.OneToOneConnector(),
                                              synapse_type=self.sim.StaticSynapse(
                                                   weight=self.synParameters["IL-CA3contL"]["initWeight"],
                                                   delay=self.synParameters["IL-CA3contL"]["delay"]),
                                              receptor_type=self.synParameters["IL-CA3contL"]["receptor_type"])

        # CA3cue-CA3cont -> all to all STDP
        # + Time rule
        timing_rule = self.sim.SpikePairRule(tau_plus=self.synParameters["CA3cueL-CA3contL"]["tau_plus"],
                                        tau_minus=self.synParameters["CA3cueL-CA3contL"]["tau_minus"],
                                        A_plus=self.synParameters["CA3cueL-CA3contL"]["A_plus"],
                                        A_minus=self.synParameters["CA3cueL-CA3contL"]["A_minus"])
        # + Weight rule
        weight_rule = self.sim.AdditiveWeightDependence(w_max=self.synParameters["CA3cueL-CA3contL"]["w_max"],
                                                   w_min=self.synParameters["CA3cueL-CA3contL"]["w_min"])
        # + STDP model
        stdp_model = self.sim.STDPMechanism(timing_dependence=timing_rule, weight_dependence=weight_rule,
                                       weight=self.synParameters["CA3cueL-CA3contL"]["initWeight"],
                                       delay=self.synParameters["CA3cueL-CA3contL"]["delay"])
        # + Create the STDP synapses
        if self.initCA3W == None:
            self.CA3cueL_CA3contL_conn = self.sim.Projection(self.CA3cueLayer, self.CA3contLayer,
                                                             self.sim.AllToAllConnector(allow_self_connections=True),
                                                             synapse_type=stdp_model)
        else:
            self.CA3cueL_CA3contL_conn = self.sim.Projection(self.CA3cueLayer, self.CA3contLayer,
                                                             self.sim.FromListConnector(self.initCA3W),
                                                             synapse_type=stdp_model)

        # CA3cue-Output -> 1 to 1 excitatory and static
        self.CA3cueL_OL_conn = self.sim.Projection(self.CA3cueLayer, self.sim.PopulationView(self.OLayer, range(0, self.popNeurons["CA3cueLayer"])),
                                               self.sim.OneToOneConnector(),
                                               synapse_type=self.sim.StaticSynapse(
                                                   weight=self.synParameters["CA3cueL-OL"]["initWeight"],
                                                   delay=self.synParameters["CA3cueL-OL"]["delay"]),
                                               receptor_type=self.synParameters["CA3cueL-OL"]["receptor_type"])

        # CA3cont-Output -> 1 to 1 excitatory and static
        self.CA3contL_OL_conn = self.sim.Projection(self.CA3contLayer, self.sim.PopulationView(self.OLayer, range(self.popNeurons["CA3cueLayer"], self.popNeurons["OLayer"], 1)),
                                              self.sim.OneToOneConnector(),
                                              synapse_type=self.sim.StaticSynapse(
                                              weight=self.synParameters["CA3contL-OL"]["initWeight"],
                                              delay=self.synParameters["CA3contL-OL"]["delay"]),
                                              receptor_type=self.synParameters["CA3contL-OL"]["receptor_type"])
