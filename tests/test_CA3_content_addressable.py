
from sPyMem.CA3_content_addressable import CA3_content_addressable
import spynnaker8 as sim
import os
import matplotlib.pyplot as plt


# Parameters:
# + Number of directions of the memory
cueSize = 5
# + Size of the content of the memory in bits/neuron
contSize = 10

# + Time step of the simulation
timeStep = 1.0

# + Experiment:
experiment = 1
#   TEST:
#   1) Learn and recall
#   2) Learn, recall and relearning with forget
#   3) A mix of several operations at max frequency (4 ms space between activations of CA3 cont)
#       - 7 ms after learning
#       - 6 ms after recall
#   4) A mix of learn and recall by cue and content (only 1 content neuron at a time)
#   5) A mix of learn and recall by cue and content (several content neurons at a time)

if experiment == 1:
    # + Spikes of the input layer
    inputSpikesCue = [[0, 1, 2, 10], [], [], [], []]
    inputSpikesCont = [[0, 1, 2], [0, 1, 2], [0, 1, 2], [], [], [], [], [], [], [0, 1, 2]]
    # + Duration of the simulation
    simTime = 20
    # + Name of the experiment, i.e., name of the generated figure
    experiment_name = "1_learn_recall"
    recordWeight = False
elif experiment == 2:
    inputSpikesCue = [[0, 1, 2, 10, 20, 21, 22, 30], [], [], [], []]
    inputSpikesCont = [[0, 1, 2], [0, 1, 2], [0, 1, 2, 20, 21, 22], [20, 21, 22], [], [], [], [], [20, 21, 22], [0, 1, 2]]
    simTime = 40
    experiment_name = "2_learn_forget_relearn"
    recordWeight = False
elif experiment == 3:
    # op -> t=0[0,2](5)=[7,8,9], t=7[2](4)=[6,7,8], t=14[0,1](3)=[5,6,7], t=21[0,2](5), t=27[2](4),
    #       t=33[2](4)=[0,1,5,6,7], t=40[0,1](3)=[0,1,8,9], t=47[2](4), t=53[0,1](3)
    inputSpikesCue = [[], [], [14, 15, 16, 40, 41, 42, 53], [7, 8, 9, 27, 33, 34, 35, 47], [0, 1, 2, 21]]
    inputSpikesCont = [[33, 34, 35, 40, 41, 42], [33, 34, 35, 40, 41, 42], [], [], [], [14, 15, 16, 33, 34, 35],
                       [7, 8, 9, 14, 15, 16, 33, 34, 35], [0, 1, 2, 7, 8, 9, 14, 15, 16, 33, 34, 35],
                       [0, 1, 2, 7, 8, 9, 40, 41, 42],
                       [0, 1, 2, 40, 41, 42]]
    simTime = 60
    experiment_name = "3_combined_operations"
    recordWeight = False
elif experiment == 4:
    """
    inputSpikesCue = [[0, 1, 2, 20, 80, 81, 82, 100], [], [], [], []]
    inputSpikesCont = [[0, 1, 2, 60], [0, 1, 2, 70], [0, 1, 2, 40, 80, 81, 82], [80, 81, 82], [80, 81, 82], [], [], [], [], [0, 1, 2, 50]]
    simTime = 110
    experiment_name = "4_learn_recall_by_content_01"
    """
    inputSpikesCue = [[0, 1, 2, 50, 60, 61, 62, 70], [10, 11, 12], [], [], []]
    inputSpikesCont = [[0, 1, 2, 10, 11, 12, 20, 80], [0, 1, 2], [0, 1, 2, 40, 60, 61, 62, 90], [60, 61, 62], [], [],
                       [], [10, 11, 12],
                       [0, 1, 2, 10, 11, 12], [0, 1, 2, 10, 11, 12, 30, 60, 61, 62, 100]]
    simTime = 110
    experiment_name = "4_learn_recall_by_content_one_at_time"

    recordWeight = False
elif experiment == 5:
    inputSpikesCue = [[0, 1, 2], [10, 11, 12, 50], [20, 21, 22], [], []]
    inputSpikesCont = [[0, 1, 2, 30], [0, 1, 2, 10, 11, 12, 60], [0, 1, 2, 10, 11, 12, 20, 21, 22, 40],
                       [10, 11, 12, 20, 21, 22, 60],
                       [20, 21, 22, 30], [], [], [], [], []]
    simTime = 70
    experiment_name = "5_learn_recall_by_content_several_at_time"

    recordWeight = False

else:
    inputSpikesCue = [[], [], []]
    inputSpikesCont = [[], [], [], [], [], [], [], [], [], []]
    simTime = 1
    experiment_name = "none"
    recordWeight = False
inputSpikes = inputSpikesCue + inputSpikesCont


def test():
    # + Number of neurons in input layer: the number of bits neccesary to represent the number of directions
    #       in binary + the size of patterns
    numInputLayerNeurons = cueSize + contSize

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
    memory = CA3_content_addressable.Memory(cueSize, contSize, sim)
    memory.connect_in(ILayer)
    memory.connect_out(OLayer)

    # Record spikes from output layer
    memory.CA3cueCueRecallLayer.record(["spikes"])
    memory.CA3cueContRecallLayer.record(["spikes"])
    memory.CA3contCueRecallLayer.record(["spikes"])
    memory.CA3contContRecallLayer.record(["spikes"])
    memory.CA3contCondLayer.record(["spikes"])
    memory.CA3contCondIntLayer.record(["spikes"])
    memory.CA3mergeCueLayer.record(["spikes"])
    memory.CA3mergeContLayer.record(["spikes"])
    OLayer.record(["spikes"])

    # Begin simulation
    sim.run(simTime)

    # Get spike information
    #   - CA3cueCueRecallLayer
    spikes = memory.CA3cueCueRecallLayer.get_data(variables=["spikes"]).segments[0].spiketrains
    formatSpikesCA3cueCueRecallLayer = []
    for neuron in spikes:
        formatSpikesCA3cueCueRecallLayer.append(neuron.as_array().tolist())
    #   - CA3cueContRecallLayer
    spikes = memory.CA3cueContRecallLayer.get_data(variables=["spikes"]).segments[0].spiketrains
    formatSpikesCA3cueContRecallLayer = []
    for neuron in spikes:
        formatSpikesCA3cueContRecallLayer.append(neuron.as_array().tolist())
    #   - CA3contCueRecallLayer
    spikes = memory.CA3contCueRecallLayer.get_data(variables=["spikes"]).segments[0].spiketrains
    formatSpikesCA3contCueRecallLayer = []
    for neuron in spikes:
        formatSpikesCA3contCueRecallLayer.append(neuron.as_array().tolist())
    #   - CA3contCond
    spikes = memory.CA3contCondLayer.get_data(variables=["spikes"]).segments[0].spiketrains
    formatSpikesCA3contCondLayer = []
    for neuron in spikes:
        formatSpikesCA3contCondLayer.append(neuron.as_array().tolist())
    #   - CA3contCondInt
    spikes = memory.CA3contCondIntLayer.get_data(variables=["spikes"]).segments[0].spiketrains
    formatSpikesCA3contCondIntLayer = []
    for neuron in spikes:
        formatSpikesCA3contCondIntLayer.append(neuron.as_array().tolist())
    #   - CA3contContRecallLayer
    spikes = memory.CA3contContRecallLayer.get_data(variables=["spikes"]).segments[0].spiketrains
    formatSpikesCA3contContRecallLayer = []
    for neuron in spikes:
        formatSpikesCA3contContRecallLayer.append(neuron.as_array().tolist())
    #   - CA3mergeCueLayer
    spikes = memory.CA3mergeCueLayer.get_data(variables=["spikes"]).segments[0].spiketrains
    formatSpikesCA3mergeCueLayer= []
    for neuron in spikes:
        formatSpikesCA3mergeCueLayer.append(neuron.as_array().tolist())
    #   - CA3mergeContLayer
    spikes = memory.CA3mergeContLayer.get_data(variables=["spikes"]).segments[0].spiketrains
    formatSpikesCA3mergeContLayer = []
    for neuron in spikes:
        formatSpikesCA3mergeContLayer.append(neuron.as_array().tolist())
    #   - OUT
    spikes = OLayer.get_data(variables=["spikes"]).segments[0].spiketrains
    formatSpikesOut = []
    for neuron in spikes:
        formatSpikesOut.append(neuron.as_array().tolist())

    # End simulation
    sim.end()

    # Plot results
    #   + Check if results folder exist, if not, create it
    if not os.path.isdir("results/"):
        os.mkdir("results/")
    #   + Spike activities plot
    spikes_plot([inputSpikes, formatSpikesCA3cueCueRecallLayer, formatSpikesCA3contCueRecallLayer,
                      formatSpikesCA3contCondLayer, formatSpikesCA3contCondIntLayer,
                      formatSpikesCA3cueContRecallLayer, formatSpikesCA3contContRecallLayer,
                      formatSpikesCA3mergeCueLayer, formatSpikesCA3mergeContLayer, formatSpikesOut],
                     ["IN", "CueCue", "ContCue", "ContCond", "ContCondInt", "CueCont", "ContCont", "MergeCue", "MergeCont", "OUT"],
                     ["o", "o", "o", "o", "o", "o", "o", "o", "o", "o"],
                     ["green", "orange", "gold", "black", "navy", "teal", "aqua", "darkviolet", "blue", "red"],
                     ["IN", "CA3cueCueRecall", "CA3contCueRecall", "CA3contCond", "CA3contCondInt", "CA3cueContRecall",
                      "CA3contContRecall", "CA3MergeCue", "CA3MergeCont", "OUT"],
                     "Hipocampal spikes", "results/"+experiment_name+"/", "spikes", False, True)

    print("Finished!")


def spikes_plot(spikes, popNames, pointTypes, colors, labels, title, outFilePath, baseFilename, plot, write):
    plt.figure(figsize=(26, 29))

    # Add point for each neuron of each population that fire, take y labels and x labels
    populationsXValues = []
    populationsYValues = []
    globalIndex = 0
    listYticks = []
    listXticks = []
    for indexPop, populationSpikes in enumerate(spikes):
        xvalues = []
        yvalues = []
        # Assign y value (population index) and y label
        for indexNeuron, spikesSingleNeuron in enumerate(populationSpikes):
            listYticks.append(popNames[indexPop] + str(indexNeuron))
            xvalues = xvalues + spikesSingleNeuron
            yvalues = yvalues + [indexNeuron + globalIndex for i in spikesSingleNeuron]
        globalIndex = globalIndex + len(populationSpikes)
        # Add to the populations values list
        populationsXValues.append(xvalues)
        populationsYValues.append(yvalues)
        # Add xvalues to labels
        listXticks = list(set(listXticks + xvalues))
    maxXvalue = max(listXticks)
    minXvalue = min(listXticks)

    # Lines for each points
    for indexPop in range(len(spikes)):
        plt.vlines(populationsXValues[indexPop], ymin=-1, ymax=populationsYValues[indexPop], color=colors[indexPop],
                   alpha=0.1)
        plt.hlines(populationsYValues[indexPop], xmin=-1, xmax=populationsXValues[indexPop], color=colors[indexPop],
                   alpha=0.1)

    # Add spikes to scatterplot
    for indexPop in range(len(spikes)):
        plt.plot(populationsXValues[indexPop], populationsYValues[indexPop], pointTypes[indexPop],
                 color=colors[indexPop], label=labels[indexPop], markersize=10)

    # Metadata
    plt.xlabel("Simulation time (ms)", fontsize=20)
    plt.ylabel("Neuron spikes", fontsize=20)
    plt.title(title, fontsize=20)
    plt.ylim([-1, globalIndex])
    plt.xlim(-1 + minXvalue, maxXvalue + 1)
    plt.yticks(range(len(listYticks)), listYticks, fontsize=20)
    plt.legend(fontsize=20)

    # Divide xticks list in pair or odd position
    listXticks.sort()
    listXticksOdd = [int(tick) for index, tick in enumerate(listXticks) if not (index % 2 == 0)]
    listXticksPair = [int(tick) for index, tick in enumerate(listXticks) if index % 2 == 0]
    # Write them with alternate distance
    ax = plt.gca()
    ax.set_xticklabels(listXticksOdd, minor=True)
    ax.set_xticks(listXticksOdd, minor=True)
    ax.set_xticklabels(listXticksPair, minor=False)
    ax.set_xticks(listXticksPair, minor=False)
    ax.tick_params(axis='x', which='minor', pad=35)
    ax.tick_params(axis='x', which='both', labelsize=18, rotation=90)

    # Save and/or plot
    if write:
        # Check if folder exist, if not, create it
        if not os.path.isdir(outFilePath):
            os.mkdir(outFilePath)
        plt.savefig(outFilePath + baseFilename + ".png")
    if plot:
        plt.show()


if __name__ == "__main__":
    test()
