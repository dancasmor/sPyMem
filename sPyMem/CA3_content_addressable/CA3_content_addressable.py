
import json
import os


"""
CA3 complex interconections one-hot cue content addressable

+ Population:
    + Input: memory input with cue part in one-hot
    + CA3cue: (CA3cueCueRecall and CA3cueContRecall) store direction/cue of memories in one-hot codification
    + CA3cont: (CA3contCueRecall and CA3contContRecall) store content of memories
    + CA3contCond: conditions the activity of CA3contCont as a function of the activity of CA3cueCue and CA3contCue
    + CA3contCondInt: recurrent colateral interneuron in CA3contCond that act as an excitatory feedback
    + CA3merge: (CA3mergeCue and CA3mergeCont) merge the activity of both cue and cont CA3 subpopulations
    + Output: output of the network

+ Synapses: 
    + Input-CA3cueCueRecall: 1 to 1 excitatory and static (first n bits: corresponding to the one-hot cue of memories)
    + Input-CA3contCueRecall: 1 to 1 excitatory and static (the rest of the bits)
    
    + CA3cueCueRecall-CA3contCueRecall: all to all excitatory and dinamic (STDP)
    + CA3contContRecall-CA3cueContRecall: all to all excitatory and dinamic (STDP)
    + CA3cueContRecall-CA3cueContRecall: all to all (except itself) inhibitory and static
    
    + CA3cueCueRecall-CA3cueContRecall: 1 to 1 excitatory and static
    + CA3cueCueRecall-CA3cueContRecall-inh: all to all (except itself) inhibitory and static
    + CA3cueCueRecall-CA3contCond: all to 1 inhibitory and static
    + CA3contCueRecall-CA3contCond: 1 to 1 excitatory and static
    + CA3contCond-CA3contContRecall: 1 to 1 excitatory and static
    + CA3contCueRecall-CA3contCondInt: all to all excitatory and static
    + CA3contCondInt-CA3contCond: all to all excitatory and static
    
    + CA3cueCueRecall-CA3mergeCue: 1 to 1 excitatory and static
    + CA3cueContRecall-CA3mergeCue: 1 to 1 excitatory and static
    + CA3contCueRecall-CA3mergeCont: 1 to 1 excitatory and static
    + CA3contContRecall-CA3mergeCont: 1 to 1 excitatory and static
    
    + CA3mergeCue-Output: 1 to 1 excitatory and static
    + CA3mergeCont-Output: 1 to 1 excitatory and static
"""


class Memory:
    """Spike-based bio-inspired hippocampal (CA3 only with complex interconections) memory model with forgetting and content addressable

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
       :param initCA3CueContW: list of initial weight to use in CA3cue-CA3cont synapse (initial memory content); format of each element of the list: (source_neuron_id, destination_neuron_id, initial_weight, delay)
       :type initCA3CueContW: list, optional
       :param initCA3ContCueW: list of initial weight to use in CA3cont-CA3cue synapse (initial memory content); format of each element of the list: (source_neuron_id, destination_neuron_id, initial_weight, delay)
       :type initCA3ContCueW: list, optional

       :ivar cueSize: number of cues of the memory, initial value: cueSize
       :vartype cueSize: int
       :ivar contSize: size of the content of the memory in bits/neuron, initial value: contSize
       :vartype contSize: int
       :ivar sim: object in charge of handling the simulation, initial value: sim
       :vartype sim: simulation object (spynnaker8 for spynnaker)
       :ivar ILayer: input population to the memory model, initial value: ILayer
       :vartype ILayer: population
       :ivar CA3cueCueRecallLayer: CA3cueCueRecall population
       :vartype CA3cueCueRecallLayer: population
       :ivar CA3cueContRecallLayer: CA3cueContRecall population
       :vartype CA3cueContRecallLayer: population
       :ivar CA3contCueRecallLayer: CA3contCueRecall population
       :vartype CA3contCueRecallLayer: population
       :ivar CA3contContRecallLayer: CA3contContRecall population
       :vartype CA3contContRecallLayer: population
       :ivar CA3contCondLayer: CA3contCond population
       :vartype CA3contCondLayer: population
       :ivar CA3contCondIntLayer: CA3contCond population
       :vartype CA3contCondIntLayer: population
       :ivar CA3mergeCueLayer: CA3mergeCue population
       :vartype CA3mergeCueLayer: population
       :ivar CA3mergeContLayer: CA3mergeCont population
       :vartype CA3mergeContLayer: population
       :ivar OLayer: output population of the memory model, initial value: OLayer
       :vartype OLayer: population
       :ivar configFilePath: path + filename to the config file of internal model parameters, initial value: configFilePath or internal path to default config file
       :vartype configFilePath: str
       :ivar initCA3CueContW: list of initial weight to use in CA3cue-CA3cont synapse (initial memory content); format of each element of the list: (source_neuron_id, destination_neuron_id, initial_weight, delay)
       :vartype initCA3CueContW: list
       :ivar initCA3ContCueW: list of initial weight to use in CA3cont-CA3cue synapse (initial memory content); format of each element of the list: (source_neuron_id, destination_neuron_id, initial_weight, delay)
       :vartype initCA3ContCueW: list
       :ivar popNeurons: dict that contains the number of neuron of each population, at the input interface level - {"ILayer": ilInputSize, "DGLayer": dgInputSize, "CA3cueLayer": self.cueSize, "CA3contLayer": self.contSize, "CA1Layer": self.cueSize, "OLayer": ilInputSize}
       :vartype popNeurons: dict
       :ivar neuronParameters: all neuron parameters of each population (for more information see `Custom config files`_)
       :vartype neuronParameters: dict
       :ivar initNeuronParameters: init membrane potential of each population (for more information see `Custom config files`_)
       :vartype initNeuronParameters: dict
       :ivar synParameters: all synapses parameters of each synapse group (for more information see `Custom config files`_)
       :vartype synParameters: dict
       :ivar IL_CA3cueCueRecallL_conn: IL-CA3cueCueRecallL synapses
       :vartype IL_CA3cueCueRecallL_conn: synapse
       :ivar CA3cueCueRecallL_CA3cueContRecallL_conn: CA3cueCueRecallL-CA3cueContRecallL synapses
       :vartype CA3cueCueRecallL_CA3cueContRecallL_conn: synapse
       :ivar CA3cueCueRecallLL_CA3cueContRecallL_inh_conn: CA3cueCueRecallL-CA3cueContRecallL-inh synapses
       :vartype CA3cueCueRecallLL_CA3cueContRecallL_inh_conn: synapse
       :ivar IL_CA3contCueRecallL_conn: IL-CA3contCueRecallL synapses (STDP)
       :vartype IL_CA3contCueRecallL_conn: synapse
       :ivar CA3contCueRecallL_CA3contContRecallL_conn: CA3contCueRecallL-CA3contContRecallL synapses (STDP)
       :vartype CA3contCueRecallLCA3contCondL_conn: synapse
       :ivar CA3contCueRecallLCA3contCondL_conn: CA3contCueRecallL-CA3contCondL synapses
       :vartype CA3contCondLCA3contContRecallL_conn: synapse
       :ivar CA3contCueRecallLCA3contCondIntL_conn: CA3contCueRecallL-CA3contCondIntL synapses
       :vartype CA3contCueRecallLCA3contCondIntL_conn: synapse
       :ivar CA3contCondIntLCA3contCondL_conn: CA3contCondIntL-CA3contCondInt synapses
       :vartype CA3contCondIntLCA3contCondL_conn: synapse
       :ivar CA3contCondLCA3contContRecallL_conn: CA3contCondL-CA3contContRecallL synapses
       :vartype CA3cueCueRecallL_CA3contCueRecallL_conn: synapse
       :ivar CA3contContRecallL_CA3cueContRecallL_conn: CA3contContRecallL-CA3cueContRecallL synapses
       :vartype CA3contContRecallL_CA3cueContRecallL_conn: synapse
       :ivar CA3cueCueRecallL_CA3mergeCueL_conn: CA3cueCueRecallL-CA3mergeCueL synapses
       :vartype CA3cueCueRecallL_CA3mergeCueL_conn: synapse
       :ivar CA3cueContRecallL_CA3mergeCueL_conn: CA3cueContRecallL-CA3mergeCueL synapses
       :vartype CA3cueContRecallL_CA3mergeCueL_conn: synapse
       :ivar CA3contCueRecallL_CA3mergeContL_conn: CA3contCueRecallL-CA3mergeContL synapses
       :vartype CA3contCueRecallL_CA3mergeContL_conn: synapse
       :ivar CA3contContRecallL_CA3mergeContL_conn: CA3contContRecallL-CA3mergeContL synapses
       :vartype CA3contContRecallL_CA3mergeContL_conn: synapse
       :ivar CA3mergeCueL_OL_conn: CA3mergeCueL-OL synapses
       :vartype CA3mergeCueL_OL_conn: synapse
       :ivar CA3mergeContL_OL_conn: CA3mergeContL-OL synapses
       :vartype CA3mergeContL_OL_conn: synapse
    """
    def __init__(self, cueSize, contSize, sim, ILayer, OLayer, initCA3CueContW=None, initCA3ContCueW=None, configFilePath=None):
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

        self.initCA3CueContW = initCA3CueContW
        self.initCA3ContCueW = initCA3ContCueW

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
            return False

    def open_config_files(self):
        """Open configuration json file with all the internal parameters needed by the network and assign parameters to variables

            :returns:
        """
        # + Calculated memory parameters
        # Size of IN population
        inputSize = self.cueSize + self.contSize
        # Number of neurons for each population
        self.popNeurons = {"ILayer": inputSize, "CA3cueCueRecallLayer": self.cueSize, "CA3cueContRecallLayer": self.cueSize,
                           "CA3contCueRecallLayer": self.contSize, "CA3contContRecallLayer": self.contSize,
                           "CA3mergeCueLayer": self.cueSize, "CA3contCondLayer": self.contSize, "CA3contCondIntLayer": 1,
                           "CA3mergeContLayer": self.contSize, "OLayer": inputSize}

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
        #   + CueRecall
        self.CA3cueCueRecallLayer = self.sim.Population(self.popNeurons["CA3cueCueRecallLayer"],
                                                        self.sim.IF_curr_exp(**self.neuronParameters["CA3cueCueRecallL"]),
                                                        label="CA3cueCueRecallLayer")
        self.CA3cueCueRecallLayer.set(v=self.initNeuronParameters["CA3cueCueRecallL"]["vInit"])
        #   + ContRecall
        self.CA3cueContRecallLayer = self.sim.Population(self.popNeurons["CA3cueContRecallLayer"],
                                                         self.sim.IF_curr_exp(**self.neuronParameters["CA3cueContRecallL"]),
                                                         label="CA3cueContRecallLayer")
        self.CA3cueContRecallLayer.set(v=self.initNeuronParameters["CA3cueContRecallL"]["vInit"])
        # CA3cont
        #   + CueRecall
        self.CA3contCueRecallLayer = self.sim.Population(self.popNeurons["CA3contCueRecallLayer"],
                                                         self.sim.IF_curr_exp(**self.neuronParameters["CA3contCueRecallL"]),
                                                         label="CA3contCueRecallLayer")
        self.CA3contCueRecallLayer.set(v=self.initNeuronParameters["CA3contCueRecallL"]["vInit"])
        #   + ContRecall
        self.CA3contContRecallLayer = self.sim.Population(self.popNeurons["CA3contContRecallLayer"],
                                                          self.sim.IF_curr_exp(**self.neuronParameters["CA3contContRecallL"]),
                                                          label="CA3contContRecallLayer")
        self.CA3contContRecallLayer.set(v=self.initNeuronParameters["CA3contContRecallL"]["vInit"])
        # CA3contCond
        self.CA3contCondLayer = self.sim.Population(self.popNeurons["CA3contCondLayer"],
                                                          self.sim.IF_curr_exp(
                                                              **self.neuronParameters["CA3contCondL"]),
                                                          label="CA3contCondLayer")
        self.CA3contCondLayer.set(v=self.initNeuronParameters["CA3contCondL"]["vInit"])
        # CA3contCondInt
        self.CA3contCondIntLayer = self.sim.Population(self.popNeurons["CA3contCondIntLayer"],
                                                       self.sim.IF_curr_exp(
                                                           **self.neuronParameters["CA3contCondIntL"]),
                                                       label="CA3contCondIntLayer")
        self.CA3contCondIntLayer.set(v=self.initNeuronParameters["CA3contCondIntL"]["vInit"])
        # CA3merge
        #   + Cue
        self.CA3mergeCueLayer = self.sim.Population(self.popNeurons["CA3mergeCueLayer"],
                                                    self.sim.IF_curr_exp(**self.neuronParameters["CA3mergeCueL"]),
                                                    label="CA3mergeCueLayer")
        self.CA3mergeCueLayer.set(v=self.initNeuronParameters["CA3mergeCueL"]["vInit"])

        #   + Cont
        self.CA3mergeContLayer = self.sim.Population(self.popNeurons["CA3mergeContLayer"],
                                                     self.sim.IF_curr_exp(**self.neuronParameters["CA3mergeContL"]),
                                                     label="CA3mergeContLayer")
        self.CA3mergeContLayer.set(v=self.initNeuronParameters["CA3mergeContL"]["vInit"])

    def create_synapses(self):
        """Create all synapses of the memory model

            :returns:
        """

        # IL-CA3cueCueRecallL -> 1 to 1, excitatory and static (first cueSize bits/neurons)
        self.IL_CA3cueCueRecallL_conn = self.sim.Projection(
            self.sim.PopulationView(self.ILayer, range(0, self.popNeurons["CA3cueCueRecallLayer"])), self.CA3cueCueRecallLayer,
            self.sim.OneToOneConnector(),
            synapse_type=self.sim.StaticSynapse(
                weight=self.synParameters["IL-CA3cueCueRecallL"]["initWeight"],
                delay=self.synParameters["IL-CA3cueCueRecallL"]["delay"]),
            receptor_type=self.synParameters["IL-CA3cueCueRecallL"]["receptor_type"])

        # IL-CA3contCueRecallL -> 1 to 1, excitatory and static (last m neurons of IL: only the number of cues to use)
        self.IL_CA3contCueRecallL_conn = self.sim.Projection(self.sim.PopulationView(self.ILayer, range(
            self.popNeurons["CA3cueCueRecallLayer"], self.popNeurons["ILayer"], 1)),
                                                             self.CA3contCueRecallLayer,
                                                             self.sim.OneToOneConnector(),
                                                             synapse_type=self.sim.StaticSynapse(
                                                                 weight=self.synParameters["IL-CA3contCueRecallL"][
                                                                     "initWeight"],
                                                                 delay=self.synParameters["IL-CA3contCueRecallL"][
                                                                     "delay"]),
                                                             receptor_type=self.synParameters["IL-CA3contCueRecallL"][
                                                                 "receptor_type"])

        # CA3cueCueRecallL-CA3cueContRecallL -> 1 to 1, excitatory and static
        self.CA3cueCueRecallLL_CA3cueContRecallL_conn = self.sim.Projection(self.CA3cueCueRecallLayer,
                                                                            self.CA3cueContRecallLayer,
                                                                            self.sim.OneToOneConnector(),
                                                                            synapse_type=self.sim.StaticSynapse(
                                                                                weight=self.synParameters[
                                                                                    "CA3cueCueRecallL-CA3cueContRecallL"][
                                                                                    "initWeight"],
                                                                                delay=self.synParameters[
                                                                                    "CA3cueCueRecallL-CA3cueContRecallL"]["delay"]),
                                                                            receptor_type=
                                                                            self.synParameters["CA3cueCueRecallL-CA3cueContRecallL"][
                                                                                "receptor_type"])
        # CA3cueCueRecall-CA3cueContRecall-inh -> all to all (except itself) inhibitory and static
        self.CA3cueCueRecallLL_CA3cueContRecallL_inh_conn = self.sim.Projection(self.CA3cueCueRecallLayer,
                                                                            self.CA3cueContRecallLayer,
                                                                            self.sim.AllToAllConnector(allow_self_connections=False),
                                                                            synapse_type=self.sim.StaticSynapse(
                                                                                weight=self.synParameters[
                                                                                    "CA3cueCueRecallL-CA3cueContRecallL-inh"][
                                                                                    "initWeight"]*self.contSize,
                                                                                delay=self.synParameters[
                                                                                    "CA3cueCueRecallL-CA3cueContRecallL-inh"][
                                                                                    "delay"]),
                                                                            receptor_type=
                                                                            self.synParameters[
                                                                                "CA3cueCueRecallL-CA3cueContRecallL-inh"][
                                                                                "receptor_type"])
        # CA3cueCueRecall-CA3contCond -> all to 1 inhibitory and static
        for id in range(self.contSize):
            CA3cueCueRecallLCA3contCondL_conn = self.sim.Projection(self.CA3cueCueRecallLayer,
                                                                           self.sim.PopulationView(self.CA3contCondLayer, [id]),
                                                                           self.sim.AllToAllConnector(allow_self_connections=True),
                                                                           synapse_type=self.sim.StaticSynapse(
                                                                               weight=self.synParameters[
                                                                                   "CA3cueCueRecallL-CA3contCondL"][
                                                                                   "initWeight"],
                                                                               delay=self.synParameters[
                                                                                   "CA3cueCueRecallL-CA3contCondL"][
                                                                                   "delay"]),
                                                                           receptor_type=self.synParameters[
                                                                               "CA3cueCueRecallL-CA3contCondL"][
                                                                               "receptor_type"])
        # CA3contCueRecall-CA3contCond -> 1 to 1 excitatory and static
        self.CA3contCueRecallLCA3contCondL_conn = self.sim.Projection(self.CA3contCueRecallLayer,
                                                                      self.CA3contCondLayer,
                                                                      self.sim.OneToOneConnector(),
                                                                      synapse_type=self.sim.StaticSynapse(
                                                                          weight=self.synParameters[
                                                                              "CA3contCueRecallL-CA3contCondL"][
                                                                              "initWeight"],
                                                                          delay=self.synParameters[
                                                                              "CA3contCueRecallL-CA3contCondL"][
                                                                              "delay"]),
                                                                      receptor_type=self.synParameters[
                                                                          "CA3contCueRecallL-CA3contCondL"][
                                                                          "receptor_type"])
        # CA3contCond-CA3contContRecall -> 1 to 1 excitatory and static
        self.CA3contCondLCA3contContRecallL_conn = self.sim.Projection(self.CA3contCondLayer,
                                                                            self.CA3contContRecallLayer,
                                                                            self.sim.OneToOneConnector(),
                                                                            synapse_type=self.sim.StaticSynapse(
                                                                                weight=self.synParameters[
                                                                                    "CA3contCondL-CA3contContRecallL"][
                                                                                    "initWeight"],
                                                                                delay=self.synParameters[
                                                                                    "CA3contCondL-CA3contContRecallL"][
                                                                                    "delay"]),
                                                                            receptor_type=self.synParameters[
                                                                                "CA3contCondL-CA3contContRecallL"][
                                                                                "receptor_type"])
        # CA3contCueRecall-CA3contCondInt -> all to all excitatory and static
        self.CA3contCueRecallLCA3contCondIntL_conn = self.sim.Projection(self.CA3contCueRecallLayer,
                                                                    self.CA3contCondIntLayer,
                                                                    self.sim.AllToAllConnector(
                                                                        allow_self_connections=True),
                                                                    synapse_type=self.sim.StaticSynapse(
                                                                        weight=self.synParameters[
                                                                            "CA3contCueRecallL-CA3contCondIntL"][
                                                                            "initWeight"],
                                                                        delay=self.synParameters[
                                                                            "CA3contCueRecallL-CA3contCondIntL"][
                                                                            "delay"]),
                                                                    receptor_type=self.synParameters[
                                                                        "CA3contCueRecallL-CA3contCondIntL"][
                                                                        "receptor_type"])
        # CA3contCondInt-CA3contCond -> all to all excitatory and static
        self.CA3contCondIntLCA3contCondL_conn = self.sim.Projection(self.CA3contCondIntLayer,
                                                                    self.CA3contCondLayer,
                                                                    self.sim.AllToAllConnector(
                                                                        allow_self_connections=True),
                                                                    synapse_type=self.sim.StaticSynapse(
                                                                        weight=self.synParameters[
                                                                            "CA3contCondIntL-CA3contCondL"][
                                                                            "initWeight"],
                                                                        delay=self.synParameters[
                                                                            "CA3contCondIntL-CA3contCondL"][
                                                                            "delay"]),
                                                                    receptor_type=self.synParameters[
                                                                        "CA3contCondIntL-CA3contCondL"][
                                                                        "receptor_type"])

        # CA3cueCueRecall-CA3contCueRecall -> all to all STDP
        # + Time rule
        timing_rule = self.sim.SpikePairRule(tau_plus=self.synParameters["CA3cueCueRecallL-CA3contCueRecallL"]["tau_plus"],
                                        tau_minus=self.synParameters["CA3cueCueRecallL-CA3contCueRecallL"]["tau_minus"],
                                        A_plus=self.synParameters["CA3cueCueRecallL-CA3contCueRecallL"]["A_plus"],
                                        A_minus=self.synParameters["CA3cueCueRecallL-CA3contCueRecallL"]["A_minus"])
        # + Weight rule
        weight_rule = self.sim.AdditiveWeightDependence(w_max=self.synParameters["CA3cueCueRecallL-CA3contCueRecallL"]["w_max"],
                                                   w_min=self.synParameters["CA3cueCueRecallL-CA3contCueRecallL"]["w_min"])
        # + STDP model
        stdp_model = self.sim.STDPMechanism(timing_dependence=timing_rule, weight_dependence=weight_rule,
                                       weight=self.synParameters["CA3cueCueRecallL-CA3contCueRecallL"]["initWeight"],
                                       delay=self.synParameters["CA3cueCueRecallL-CA3contCueRecallL"]["delay"])
        # + Create the STDP synapses
        if self.initCA3CueContW == None:
            self.CA3cueCueRecallL_CA3contCueRecallL_conn = self.sim.Projection(self.CA3cueCueRecallLayer, self.CA3contCueRecallLayer,
                                                                               self.sim.AllToAllConnector(allow_self_connections=True),
                                                                               synapse_type=stdp_model)
        else:
            self.CA3cueCueRecallL_CA3contCueRecallL_conn = self.sim.Projection(self.CA3cueCueRecallLayer, self.CA3contCueRecallLayer,
                                                                               self.sim.FromListConnector(self.initCA3CueContW),
                                                                               synapse_type=stdp_model)

        # CA3contContRecall-CA3cueContRecall -> all to all STDP
        # + Time rule
        timing_rule = self.sim.SpikePairRule(
            tau_plus=self.synParameters["CA3contContRecallL-CA3cueContRecallL"]["tau_plus"],
            tau_minus=self.synParameters["CA3contContRecallL-CA3cueContRecallL"]["tau_minus"],
            A_plus=self.synParameters["CA3contContRecallL-CA3cueContRecallL"]["A_plus"],
            A_minus=self.synParameters["CA3contContRecallL-CA3cueContRecallL"]["A_minus"])
        # + Weight rule
        weight_rule = self.sim.AdditiveWeightDependence(
            w_max=self.synParameters["CA3contContRecallL-CA3cueContRecallL"]["w_max"],
            w_min=self.synParameters["CA3contContRecallL-CA3cueContRecallL"]["w_min"])
        # + STDP model
        stdp_model = self.sim.STDPMechanism(timing_dependence=timing_rule, weight_dependence=weight_rule,
                                            weight=self.synParameters["CA3contContRecallL-CA3cueContRecallL"][
                                                "initWeight"],
                                            delay=self.synParameters["CA3contContRecallL-CA3cueContRecallL"]["delay"])
        # + Create the STDP synapses
        if self.initCA3ContCueW == None:
            self.CA3contContRecallL_CA3cueContRecallL_conn = self.sim.Projection(self.CA3contContRecallLayer, self.CA3cueContRecallLayer,
                                                                                 self.sim.AllToAllConnector(allow_self_connections=True),
                                                                                 synapse_type=stdp_model)
        else:
            self.CA3contContRecallL_CA3cueContRecallL_conn = self.sim.Projection(self.CA3contContRecallLayer, self.CA3cueContRecallLayer,
                                                                                 self.sim.FromListConnector(self.initCA3ContCueW),
                                                                                 synapse_type=stdp_model)

        # CA3cueContRecall-CA3cueContRecall -> all to all (except itself) inhibitory and static
        self.CA3cueContRecallL_CA3cueContRecallL_conn = self.sim.Projection(self.CA3cueContRecallLayer,
                                                                             self.CA3cueContRecallLayer,
                                                                             self.sim.AllToAllConnector(
                                                                                 allow_self_connections=False),
                                                                             synapse_type=self.sim.StaticSynapse(
                                                                                  weight=self.synParameters[
                                                                                      "CA3cueContRecallL-CA3cueContRecallL"][
                                                                                      "initWeight"]*self.contSize,
                                                                                  delay=self.synParameters[
                                                                                      "CA3cueContRecallL-CA3cueContRecallL"][
                                                                                      "delay"]),
                                                                            receptor_type=self.synParameters[
                                                                                  "CA3cueContRecallL-CA3cueContRecallL"][
                                                                                  "receptor_type"])

        # CA3cueCueRecall-CA3mergeCue -> 1 to 1 excitatory and static
        self.CA3cueCueRecallL_CA3mergeCueL_conn = self.sim.Projection(self.CA3cueCueRecallLayer, self.CA3mergeCueLayer,
                                                                      self.sim.OneToOneConnector(),
                                                                      synapse_type=self.sim.StaticSynapse(
                                                                          weight=self.synParameters[
                                                                              "CA3cueCueRecallL-CA3mergeCueL"][
                                                                              "initWeight"],
                                                                          delay=self.synParameters[
                                                                              "CA3cueCueRecallL-CA3mergeCueL"][
                                                                              "delay"]),
                                                                      receptor_type=self.synParameters[
                                                                          "CA3cueCueRecallL-CA3mergeCueL"][
                                                                          "receptor_type"])
        # CA3cueContRecall-CA3mergeCue -> 1 to 1 excitatory and static
        self.CA3cueContRecallL_CA3mergeCueL_conn = self.sim.Projection(self.CA3cueContRecallLayer, self.CA3mergeCueLayer,
                                                                       self.sim.OneToOneConnector(),
                                                                       synapse_type=self.sim.StaticSynapse(
                                                                          weight=self.synParameters[
                                                                              "CA3cueContRecallL-CA3mergeCueL"][
                                                                              "initWeight"],
                                                                          delay=self.synParameters[
                                                                              "CA3cueContRecallL-CA3mergeCueL"][
                                                                              "delay"]),
                                                                       receptor_type=self.synParameters[
                                                                          "CA3cueContRecallL-CA3mergeCueL"][
                                                                          "receptor_type"])

        # CA3contCueRecall-CA3mergeCont -> 1 to 1 excitatory and static
        self.CA3contCueRecallL_CA3mergeContL_conn = self.sim.Projection(self.CA3contCueRecallLayer, self.CA3mergeContLayer,
                                                                        self.sim.OneToOneConnector(),
                                                                        synapse_type=self.sim.StaticSynapse(
                                                                           weight=self.synParameters[
                                                                               "CA3contCueRecallL-CA3mergeContL"][
                                                                               "initWeight"],
                                                                           delay=self.synParameters[
                                                                               "CA3contCueRecallL-CA3mergeContL"][
                                                                               "delay"]),
                                                                        receptor_type=self.synParameters[
                                                                           "CA3contCueRecallL-CA3mergeContL"][
                                                                           "receptor_type"])
        # CA3contContRecall-CA3mergeCont -> 1 to 1 excitatory and static
        self.CA3contContRecallL_CA3mergeContL_conn = self.sim.Projection(self.CA3contContRecallLayer, self.CA3mergeContLayer,
                                                                         self.sim.OneToOneConnector(),
                                                                         synapse_type=self.sim.StaticSynapse(
                                                                            weight=self.synParameters[
                                                                                "CA3contContRecallL-CA3mergeContL"][
                                                                                "initWeight"],
                                                                            delay=self.synParameters[
                                                                                "CA3contContRecallL-CA3mergeContL"][
                                                                                "delay"]),
                                                                         receptor_type=self.synParameters[
                                                                            "CA3contContRecallL-CA3mergeContL"][
                                                                            "receptor_type"])

        # CA3mergeCue-Output -> 1 to 1 excitatory and static
        self.CA3mergeCueL_OL_conn = self.sim.Projection(self.CA3mergeCueLayer, self.sim.PopulationView(self.OLayer, range(0, self.popNeurons["CA3mergeCueLayer"])),
                                                        self.sim.OneToOneConnector(),
                                                        synapse_type=self.sim.StaticSynapse(
                                                   weight=self.synParameters["CA3mergeCueL-OL"]["initWeight"],
                                                   delay=self.synParameters["CA3mergeCueL-OL"]["delay"]),
                                                        receptor_type=self.synParameters["CA3mergeCueL-OL"]["receptor_type"])

        # CA3mergeCont-Output -> 1 to 1 excitatory and static
        self.CA3mergeContL_OL_conn = self.sim.Projection(self.CA3mergeContLayer, self.sim.PopulationView(self.OLayer, range(self.popNeurons["CA3mergeCueLayer"], self.popNeurons["OLayer"], 1)),
                                                         self.sim.OneToOneConnector(),
                                                         synapse_type=self.sim.StaticSynapse(
                                              weight=self.synParameters["CA3mergeContL-OL"]["initWeight"],
                                              delay=self.synParameters["CA3mergeContL-OL"]["delay"]),
                                                         receptor_type=self.synParameters["CA3mergeContL-OL"]["receptor_type"])
