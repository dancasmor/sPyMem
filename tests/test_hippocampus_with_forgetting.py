
import matplotlib.pyplot as plt
from sPyMem.hippocampus_with_forgetting import hippocampus_with_forgetting
import spynnaker8 as sim
import math

"""
Proof of concept of Hippocampus bio-inspired pulse-based memory with forgetting

Create an hippocampal memory with forgetting and add:
    + A source input spikes population to the input of the memory
    + A dummy output population to the output population of the memory to record memory output spikes
Given an input spikes sequences, record the output of the memory and plot it. The plot represents the neuron id of
 neurons that fired in each timestamp. 

The network has a storage capacity of 5 memories at the same time of a size of 13 neurons. Of these 13 neurons,
 the 3 least significant neurons represent the cue and the remaining 10 represent the content of the memory.

The simulation lasts 25 ms during which a learning and a recall operation is performed on the memory formed by:
    + Cue = 0
    + Cont = 0, 1, 2, 9
"""

# Parameters:
# + Number of directions of the memory
cueSize = 5
# + Size of the content of the memory in bits/neuron
contSize = 10

# + Duration of the simulation
simTime = 25
# + Time step of the simulation
timeStep = 1.0

# + Spikes of the input layer
inputSpikesCue = [[1, 2, 3, 12], [], []]
inputSpikesCont = [[1, 2, 3], [1, 2, 3], [1, 2, 3], [], [], [], [], [], [], [1, 2, 3]]
inputSpikes = inputSpikesCue + inputSpikesCont

# + Number of neurons in input layer: the number of bits neccesary to represent the number of directions in
#    binary + the size of patterns
numInputLayerNeurons = math.ceil(math.log2(cueSize)) + contSize


def test():
    # Setup simulation
    sim.setup(timeStep)

    # Create network
    # Input layer
    ILayer = sim.Population(numInputLayerNeurons, sim.SpikeSourceArray(spike_times=inputSpikes), label="ILayer")
    # Output layer: fire a spike when receive a spike
    neuronParameters = {"cm": 0.27, "i_offset": 0.0, "tau_m": 3.0, "tau_refrac": 1.0, "tau_syn_E": 0.3, "tau_syn_I": 0.3,
                        "v_reset": -60.0, "v_rest": -60.0, "v_thresh": -57.5}
    OLayer = sim.Population(numInputLayerNeurons, sim.IF_curr_exp(**neuronParameters), label="OLayer")
    OLayer.set(v=-60)
    # Create memory
    memory = hippocampus_with_forgetting.Memory(cueSize, contSize, sim, ILayer, OLayer)

    # Record spikes from output layer
    OLayer.record(["spikes"])

    # Begin simulation
    sim.run(simTime)

    # Get spike information of the output layer
    outputSpikes = OLayer.get_data(variables=["spikes"]).segments[0].spiketrains

    # End simulation
    sim.end()

    # Represent the spikes:
    plt.figure(figsize=(19, 12))
    listXticks = [0, simTime]
    label = "Input"
    # + Add the spikes as a vertical line: blue for input spikes and red for output spikes
    for spikes in inputSpikes:
        plt.vlines(spikes, ymin=0, ymax=0.5, color="blue", label=label)
        label = "_nolegend_"
        listXticks = listXticks + spikes
    label = "Output"
    for spikes in outputSpikes:
        plt.vlines(spikes, ymin=0, ymax=0.5, color="red", label=label)
        label = "_nolegend_"
        listXticks = listXticks + spikes.as_array().tolist()
    # + Annotate which neurons fire in each time stamp
    for stamp in range(0, simTime):
        labelTimeStamp = ""
        sublabel = "IN="
        for index, spikes in enumerate(inputSpikes):
            if stamp in spikes:
                labelTimeStamp = labelTimeStamp + sublabel + str(index)
                sublabel = "-"
        sublabel = "OUT="
        for index, spikes in enumerate(outputSpikes):
            if stamp in spikes:
                labelTimeStamp = labelTimeStamp + sublabel + str(index)
                sublabel = "-"
        plt.annotate(labelTimeStamp, xy=(stamp + 0.1, 0.01), rotation=90)
    # + Meta information about the plot
    plt.ylim([-0.05, 0.5 + 0.05])
    plt.xlim(-0.5, simTime + 1.5)
    plt.xlabel("Simulation time (ms)")
    plt.ylabel("Spikes")
    listXticks = list(set(listXticks))
    plt.xticks(listXticks, rotation=90)
    plt.yticks([])
    plt.legend()
    plt.show()
    plt.close()

    print("Finished!")


if __name__ == "__main__":
    test()
