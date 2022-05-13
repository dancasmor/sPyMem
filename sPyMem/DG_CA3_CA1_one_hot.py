
import math
import json
from sPyBlocks.constant_spike_source import ConstantSpikeSource
from sPyBlocks.neural_decoder import NeuralDecoder
from sPyBlocks.neural_encoder import NeuralEncoder



"""
DG-CA3-CA1 one-hot memory

+ Population:
    + Input: pattern input
    + DG: one-hot codification of direction of the pattern
    + CA3cue: store direction/cue of patterns
    + CA3mem: store content/memories of patterns
    + CA1: recode the direction of the pattern to make it binary again in the output
    + Output: output of the network

+ Synapses: 
    + Input-DG: 1 to 1 excitatory and static (first n bits: corresponding to the direction of patterns)
    + Input-CA3mem: 1 to 1 excitatory and static (the rest of the bits)
    + DG-CA3cue: 1 to 1 excitatory and static
    + CA3cue-CA3mem: all to all excitatory and dinamic (STDP).
    + CA3cue-CA1: 1 to 1 excitatory and static
    + CA1-Output: 1 to 1 excitatory and static
    + CA3mem-Output: 1 to 1 excitatory and static
"""

class DG_CA3_CA1_one_hot_memory:
    def __init__(self, cueSize, memSize, sim, configFilePath, ILayer, OLayer):
        # Storing parameters
        self.cueSize = cueSize
        self.memSize = memSize
        self.sim = sim
        self.configFilePath = configFilePath
        self.ILayer = ILayer
        self.OLayer = OLayer

        # Open configurations files to get the parameters
        self.open_config_files()
        # Create the network
        self.create_population()
        self.create_synapses()

    def read_json(self):
        try:
            file = open(self.configFilePath)
            return json.load(file)
        except FileNotFoundError:
            return False

    def open_config_files(self):
        # + Calculated memory parameters
        # Input size of DG population (decoder)
        dgInputSize = math.ceil(math.log2(self.cueSize))
        # Size of IN population
        ilInputSize = dgInputSize + self.memSize
        # Number of neurons for each population
        self.popNeurons = {"ILayer": ilInputSize, "DGLayer": dgInputSize, "CA3cueLayer": self.cueSize, "CA3memLayer": self.memSize,
                      "CA1Layer": self.cueSize, "OLayer": ilInputSize}

        # + Network components parameters
        network_config = self.read_json()
        # Neurons paramaters
        self.neuronParameters = network_config["neuronParameters"]
        # Initial neuron parameters
        self.initNeuronParameters = network_config["initNeuronParameters"]
        # Synapses parameters
        self.synParameters = network_config["synParameters"]


    def create_population(self):
        # CA3cue
        self.CA3cueLayer = self.sim.Population(self.popNeurons["CA3cueLayer"], self.sim.IF_curr_exp(**self.neuronParameters["CA3cueL"]),
                                               label="CA3cueLayer")
        self.CA3cueLayer.set(v=self.initNeuronParameters["CA3cueL"]["vInit"])
        # CA3mem
        self.CA3memLayer = self.sim.Population(self.popNeurons["CA3memLayer"], self.sim.IF_curr_exp(**self.neuronParameters["CA3memL"]),
                                      label="CA3memLayer")
        self.CA3memLayer.set(v=self.initNeuronParameters["CA3memL"]["vInit"])
        # DG (decoder)
        self.DGLayer = NeuralDecoder(self.popNeurons["DGLayer"], self.sim, {"min_delay": self.synParameters["IL-DGL"]["delay"]},
                                     self.neuronParameters["DGL"], self.sim.StaticSynapse(weight=self.synParameters["IL-DGL"]["initWeight"],
                                                                                          delay=self.synParameters["IL-DGL"]["delay"]))
        # Necessary for the Decoder
        self.constant_spike_source = ConstantSpikeSource(self.sim, {"min_delay": self.synParameters["IL-DGL"]["delay"]},
                                                    self.neuronParameters["DGL"],
                                                    self.sim.StaticSynapse(weight=self.synParameters["IL-DGL"]["initWeight"],
                                                                           delay=self.synParameters["IL-DGL"]["delay"]))
        # CA1 (encoder)
        self.CA1Layer = NeuralEncoder(2 ** self.popNeurons["DGLayer"], self.sim, {"min_delay": self.synParameters["CA3cueL-CA1L"]["delay"]},
                                 self.neuronParameters["CA1L"],
                                 self.sim.StaticSynapse(weight=self.synParameters["CA3cueL-CA1L"]["initWeight"],
                                                        delay=self.synParameters["CA3cueL-CA1L"]["delay"]))

    def create_synapses(self):
        # IL-DG -> 1 to 1, excitatory and static (first dgInputSize bits/neurons)
        self.DGLayer.connect_inputs(self.sim.PopulationView(self.ILayer, range(self.popNeurons["DGLayer"])),
                               ini_pop_indexes=[[i] for i in range(self.popNeurons["DGLayer"])])
        # DG-CA3cueL -> 1 to 1, excitatory and static
        self.DGLayer.connect_outputs(self.CA3cueLayer, end_pop_indexes=[[i] for i in range(self.cueSize)],
                                     and_indexes=range(1, self.cueSize + 1),
                                     conn=self.sim.StaticSynapse(weight=self.synParameters["DGL-CA3cueL"]["initWeight"],
                                                       delay=self.synParameters["DGL-CA3cueL"]["delay"]))
        self.DGLayer.connect_constant_spikes([self.constant_spike_source.set_source, self.constant_spike_source.latch.output_neuron])

        # IL-CA3mem -> 1 to 1, excitatory and static (last m neurons of DG: only the number of cues to use)
        IL_CA3memL_conn = self.sim.Projection(self.sim.PopulationView(self.ILayer, range(self.popNeurons["DGLayer"], self.popNeurons["ILayer"], 1)),
                                               self.CA3memLayer,
                                               self.sim.OneToOneConnector(),
                                               synapse_type=self.sim.StaticSynapse(
                                                   weight=self.synParameters["IL-CA3memL"]["initWeight"],
                                                   delay=self.synParameters["IL-CA3memL"]["delay"]),
                                               receptor_type=self.synParameters["IL-CA3memL"]["receptor_type"])

        # CA3cue-CA3mem -> all to all STDP
        # + Time rule
        timing_rule = self.sim.SpikePairRule(tau_plus=self.synParameters["CA3cueL-CA3memL"]["tau_plus"],
                                        tau_minus=self.synParameters["CA3cueL-CA3memL"]["tau_minus"],
                                        A_plus=self.synParameters["CA3cueL-CA3memL"]["A_plus"],
                                        A_minus=self.synParameters["CA3cueL-CA3memL"]["A_minus"])
        # + Weight rule
        weight_rule = self.sim.AdditiveWeightDependence(w_max=self.synParameters["CA3cueL-CA3memL"]["w_max"],
                                                   w_min=self.synParameters["CA3cueL-CA3memL"]["w_min"])
        # + STDP model
        stdp_model = self.sim.STDPMechanism(timing_dependence=timing_rule, weight_dependence=weight_rule,
                                       weight=self.synParameters["CA3cueL-CA3memL"]["initWeight"],
                                       delay=self.synParameters["CA3cueL-CA3memL"]["delay"])
        # + Create the STDP synapses
        CA3cueL_CA3memL_conn = self.sim.Projection(self.CA3cueLayer, self.CA3memLayer,
                                                    self.sim.AllToAllConnector(allow_self_connections=True),
                                                    synapse_type=stdp_model)

        # CA3cue-CA1 -> 1 to 1 excitatory and static
        pop_len = len(self.CA3cueLayer)
        input_indexes = range(pop_len)
        channel_indexes = range(1, self.CA3cueLayer.size + 1)
        if len(input_indexes) != len(channel_indexes):
            raise ValueError("There is not the same number of elements in input_indexes and channel_indexes")
        for i in range(pop_len):
            i_bin = format(channel_indexes[i], "0" + str(self.CA1Layer.n_outputs) + 'b')
            i_bin_splitted = [j for j in reversed(i_bin)]
            connections = [k for k in range(0, len(i_bin_splitted)) if i_bin_splitted[k] == '1']
            self.CA1Layer.connect_inputs(self.CA3cueLayer, ini_pop_indexes=[input_indexes[i]], or_indexes=connections)
        # CA1-Output -> 1 to 1 excitatory and static
        self.CA1Layer.connect_outputs(self.sim.PopulationView(self.OLayer, range(self.popNeurons["DGLayer"])),
                                 end_pop_indexes=[[i] for i in range(self.popNeurons["DGLayer"])],
                                 conn=self.sim.StaticSynapse(weight=self.synParameters["CA1L-OL"]["initWeight"],
                                                        delay=self.synParameters["CA1L-OL"]["delay"]))

        # CA3mem-Output -> 1 to 1 excitatory and static
        CA3memL_OL_conn = self.sim.Projection(self.CA3memLayer, self.sim.PopulationView(self.OLayer, range(self.popNeurons["DGLayer"], self.popNeurons["OLayer"], 1)),
                                          self.sim.OneToOneConnector(),
                                          synapse_type=self.sim.StaticSynapse(
                                              weight=self.synParameters["CA3memL-OL"]["initWeight"],
                                              delay=self.synParameters["CA3memL-OL"]["delay"]),
                                          receptor_type=self.synParameters["CA3memL-OL"]["receptor_type"])



