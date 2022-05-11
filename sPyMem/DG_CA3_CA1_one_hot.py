
import math
import os
import sys
import json
import inspect
parent_dir_path = os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
neural_decoder_path = os.path.join(parent_dir_path, "neural_blocks")
sys.path.insert(0, neural_decoder_path)
from neural_decoder import NeuralDecoder
from neural_encoder import NeuralEncoder
from constant_spike_source import ConstantSpikeSource


"""
DG-CA3-CA1 one-hot memory

+ Population:
    + Input: pattern input
    + DG: one-hot codification of direction of the pattern
    + CA3dir: store direction of patterns
    + CA3cont: store content of patterns
    + CA1: recode the direction of the pattern to make it binary again in the output
    + Output: output of the network

+ Synapses: 
    + Input-DG: 1 to 1 excitatory and static (first n bits: corresponding to the direction of patterns)
    + Input-CA3cont: 1 to 1 excitatory and static (the rest of the bits)
    + DG-CA3dir: 1 to 1 excitatory and static
    + CA3dir-CA3cont: all to all excitatory and dinamic (STDP).
    + CA3dir-CA1: 1 to 1 excitatory and static
    + CA1-Output: 1 to 1 excitatory and static
    + CA3cont-Output: 1 to 1 excitatory and static
"""

class DG_CA3_CA1_one_hot_memory:
    def __init__(self, dirSize, contSize, sim, configFilePath, ILayer, OLayer):
        # Storing parameters
        self.dirSize = dirSize
        self.contSize = contSize
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
        dgInputSize = math.ceil(math.log2(self.dirSize))
        # Size of IN population
        ilInputSize = dgInputSize + self.contSize
        # Number of neurons for each population
        self.popNeurons = {"ILayer": ilInputSize, "DGLayer": dgInputSize, "CA3dirLayer": self.dirSize, "CA3contLayer": self.contSize,
                      "CA1Layer": self.dirSize, "OLayer": ilInputSize}

        # + Network components parameters
        network_config = self.read_json()
        # Neurons paramaters
        self.neuronParameters = network_config["neuronParameters"]
        # Initial neuron parameters
        self.initNeuronParameters = network_config["initNeuronParameters"]
        # Synapses parameters
        self.synParameters = network_config["synParameters"]


    def create_population(self):
        # PCdir
        self.CA3dirLayer = self.sim.Population(self.popNeurons["CA3dirLayer"], self.sim.IF_curr_exp(**self.neuronParameters["CA3dirL"]),
                                     label="CA3dirLayer")
        self.CA3dirLayer.set(v=self.initNeuronParameters["CA3dirL"]["vInit"])
        # PCcont
        self.CA3contLayer = self.sim.Population(self.popNeurons["CA3contLayer"], self.sim.IF_curr_exp(**self.neuronParameters["CA3contL"]),
                                      label="CA3contLayer")
        self.CA3contLayer.set(v=self.initNeuronParameters["CA3contL"]["vInit"])
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
        self.CA1Layer = NeuralEncoder(2 ** self.popNeurons["DGLayer"], self.sim, {"min_delay": self.synParameters["CA3dirL-CA1L"]["delay"]},
                                 self.neuronParameters["CA1L"],
                                 self.sim.StaticSynapse(weight=self.synParameters["CA3dirL-CA1L"]["initWeight"],
                                                        delay=self.synParameters["CA3dirL-CA1L"]["delay"]))

    def create_synapses(self):
        # IL-DG -> 1 to 1, excitatory and static (first dgInputSize bits/neurons)
        self.DGLayer.connect_inputs(self.sim.PopulationView(self.ILayer, range(self.popNeurons["DGLayer"])),
                               pop_indexes=[[i] for i in range(self.popNeurons["DGLayer"])])
        # DG-CA3dirL -> 1 to 1, excitatory and static
        self.DGLayer.connect_outputs(self.CA3dirLayer, pop_indexes=[[i] for i in range(self.dirSize)],
                                and_indexes=range(1, self.dirSize + 1),
                                conn=self.sim.StaticSynapse(weight=self.synParameters["DGL-CA3dirL"]["initWeight"],
                                                       delay=self.synParameters["DGL-CA3dirL"]["delay"]))
        self.DGLayer.connect_constant_spikes(
            [self.constant_spike_source.set_source, self.constant_spike_source.flip_flop.output_neuron])

        # IL-CA3cont -> 1 to 1, excitatory and static (last m neurons of DG: only the number of directions to use)
        IL_CA3contL_conn = self.sim.Projection(self.sim.PopulationView(self.ILayer, range(self.popNeurons["DGLayer"], self.popNeurons["ILayer"], 1)),
                                               self.CA3contLayer,
                                               self.sim.OneToOneConnector(),
                                               synapse_type=self.sim.StaticSynapse(
                                                   weight=self.synParameters["IL-CA3contL"]["initWeight"],
                                                   delay=self.synParameters["IL-CA3contL"]["delay"]),
                                               receptor_type=self.synParameters["IL-CA3contL"]["receptor_type"])

        # CA3dir-CA3cont -> all to all STDP
        # + Time rule
        timing_rule = self.sim.SpikePairRule(tau_plus=self.synParameters["CA3dirL-CA3contL"]["tau_plus"],
                                        tau_minus=self.synParameters["CA3dirL-CA3contL"]["tau_minus"],
                                        A_plus=self.synParameters["CA3dirL-CA3contL"]["A_plus"],
                                        A_minus=self.synParameters["CA3dirL-CA3contL"]["A_minus"])
        # + Weight rule
        weight_rule = self.sim.AdditiveWeightDependence(w_max=self.synParameters["CA3dirL-CA3contL"]["w_max"],
                                                   w_min=self.synParameters["CA3dirL-CA3contL"]["w_min"])
        # + STDP model
        stdp_model = self.sim.STDPMechanism(timing_dependence=timing_rule, weight_dependence=weight_rule,
                                       weight=self.synParameters["CA3dirL-CA3contL"]["initWeight"],
                                       delay=self.synParameters["CA3dirL-CA3contL"]["delay"])
        # + Create the STDP synapses
        CA3dirL_CA3contL_conn = self.sim.Projection(self.CA3dirLayer, self.CA3contLayer,
                                               self.sim.AllToAllConnector(allow_self_connections=True),
                                               synapse_type=stdp_model)

        # CA3dir-CA1 -> 1 to 1 excitatory and static
        self.CA1Layer.connect_inputs(self.CA3dirLayer, pop_indexes=[0, 2, 4], or_indexes=[0])
        self.CA1Layer.connect_inputs(self.CA3dirLayer, pop_indexes=[1, 2], or_indexes=[1])
        self.CA1Layer.connect_inputs(self.CA3dirLayer, pop_indexes=[3, 4], or_indexes=[2])
        # CA1-Output -> 1 to 1 excitatory and static
        self.CA1Layer.connect_outputs(self.sim.PopulationView(self.OLayer, range(self.popNeurons["DGLayer"])),
                                 pop_indexes=[[i] for i in range(self.popNeurons["DGLayer"])],
                                 conn=self.sim.StaticSynapse(weight=self.synParameters["CA1L-OL"]["initWeight"],
                                                        delay=self.synParameters["CA1L-OL"]["delay"]))

        # CA3cont-Output -> 1 to 1 excitatory and static
        CA3contL_OL_conn = self.sim.Projection(self.CA3contLayer, self.sim.PopulationView(self.OLayer, range(self.popNeurons["DGLayer"], self.popNeurons["OLayer"], 1)),
                                          self.sim.OneToOneConnector(),
                                          synapse_type=self.sim.StaticSynapse(
                                              weight=self.synParameters["CA3contL-OL"]["initWeight"],
                                              delay=self.synParameters["CA3contL-OL"]["delay"]),
                                          receptor_type=self.synParameters["CA3contL-OL"]["receptor_type"])



