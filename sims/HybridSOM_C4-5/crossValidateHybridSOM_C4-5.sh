# ===================================================================================================
# Script        : crossValidateHybridSOM_C4_5.sh
# Usage         : ./crossValidateHybridSOM_C4_5.sh [simDirectory]
# Purpose       : Performs a cross-validated paired test on parameter data set using HybridSOM
#                 framework configured with C4_5
#
# Preconditions : - All required simulation-related programs are present
#                 - Data set as directory name in dataset directory must exist and be in correct form
#                   (i.e. no header, one record per line, comma delimited, last value is class label)
# ===================================================================================================

#!/bin/bash

# ===================================================================================================
# Function : validateSystem(simDirectory)
# Purpose  : Validates components required for cross validation run (based on simDirectory)
# Return   : None
# ===================================================================================================

validateSystem() {
	
	# check for correct number of parameters
	if [ -z ${1} ]; then
		echo "Usage: ./crossValidate.sh [simDirectory]"
		exit
	fi
	
	echo "Status: Validating system..."
	echo "=============================================="
	
	# check for data directory
	echo -n "Simulation directory \"${1}\"...  "
	if [ ! -d ../../datasets/${1} ]; then
		echo "NOT FOUND"
		exit
	fi
	echo "FOUND"
	
	# check for original data file in simulation directory
	echo -n "Data file \"${1}/original/${1}.data\"...  "
	if [ ! -f ../../datasets/${1}/original/${1}.data ]; then
		echo "NOT FOUND"
		exit
	fi
	echo "FOUND"
	
	# check for cross validation set building script
	echo -n "Cross validation set building script...  "
	if [ ! -f ../../algorithms/BuildSets/BuildSets.class ]; then
		echo "NOT FOUND"
		exit
	fi
	echo "FOUND"
	
	# check for SOM_PAK data conversion script
	echo -n "Conversion script \"${1}/convert_som.awk\"...  "
	if [ ! -f ./../../datasets/${1}/convert_som.awk ]; then
		echo "NOT FOUND"
		exit
	fi
	echo "FOUND"
	
	# check for SOM_PAK names script
	echo -n "Names script \"${1}/names_som.sh\"...  "
	if [ ! -f ./../../datasets/${1}/names_som.sh ]; then
		echo "NOT FOUND"
		exit
	fi
	echo "FOUND"
	
	# check for SOM_PAK initialisation program
	echo -n "SOM initialisation program...  "
	if [ ! -f ../../algorithms/SOM_PAK/lininit ]; then
		echo "NOT FOUND"
		exit
	fi
	echo "FOUND"
	
	# check for SOM_PAK training program
	echo -n "SOM training program...  "
	if [ ! -f ../../algorithms/SOM_PAK/vsom ]; then
		echo "NOT FOUND"
		exit
	fi
	echo "FOUND"
	
	# check for example-centric neuron labeling program
	echo -n "Example-centric neuron labeling program...  "
	if [ ! -f./../../algorithms/ExampleCentricNeuron/outputlabels ]; then
		echo "NOT FOUND"
		exit
	fi
	echo "FOUND"
	
	# check for example-centric K-means cluster labeling program
	echo -n "Example-centric K-means cluster labeling program...  "
	if [ ! -f./../../algorithms/ExampleCentricClusterKMeans/outputLabels ]; then
		echo "NOT FOUND"
		exit
	fi
	echo "FOUND"
	
	# check for example-centric Ward cluster labeling program
	echo -n "Example-centric Ward cluster labeling program...  "
	if [ ! -f./../../algorithms/ExampleCentricClusterWard/outputLabels ]; then
		echo "NOT FOUND"
		exit
	fi
	echo "FOUND"
	
	# check for weight-centric neuron labeling program
	echo -n "Weight-centric neuron labeling program...  "
	if [ ! -f./../../algorithms/WeightCentricNeuron/outputLabels ]; then
		echo "NOT FOUND"
		exit
	fi
	echo "FOUND"
	
	# check for C4.5 data conversion script
	echo -n "Data conversion script \"${1}/convert_c4_5.awk\"...  "
	if [ ! -f ./../../datasets/${1}/convert_c4_5.awk ]; then
		echo "NOT FOUND"
		exit
	fi
	echo "FOUND"
	
	# check for C4.5 program
	echo -n "C4.5 program...  "
	if [ ! -f ../../algorithms/C4_5/Src/c4.5 ]; then
		echo "NOT FOUND"
		exit
	fi
	echo "FOUND"
	
	#check for results averaging program
	echo -n "Results averaging program...  "
	if [ ! -f ../../algorithms/AverageRunResults/AverageRuleSetResults.class ]; then
		echo "NOT FOUND"
		exit
	fi
	echo "FOUND"
	
	echo "=============================================="
	echo "Status: System OK..."
	
}

# ===================================================================================================
# Function : buildSets(simDirectory)
# Purpose  : Builds data sets required for cross validation simulations (based on simDirectory)
# Return   : None
# ===================================================================================================

buildSets() {
	
	echo "Status: Performing cross validation set builds"
	
	# build raw training and test sets (must still be converted to SOM_PAK and C4.5 formats)
	if [ ${1} == "ionosphere" ]; then
		echo "Status: Building set for IONOSPHERE database"
		echo "=============================================="
		java BuildSets ../../datasets/${1}/original/${1}.data , 30 -normalise
	elif [ ${1} == "iris" ]; then
		echo "Status: Building set for IRIS database"
		echo "=============================================="
		java BuildSets ../../datasets/${1}/original/${1}.data , 30 -normalise
	elif [ ${1} == "monks1" ]; then
		echo "Status: Building set for MONKS1 database"
		echo "=============================================="
		java BuildSets ../../datasets/${1}/original/${1}.data , 30
	elif [ ${1} == "monks2" ]; then
		echo "Status: Building set for MONKS2 database"
		echo "=============================================="
		java BuildSets ../../datasets/${1}/original/${1}.data , 30
	elif [ ${1} == "monks3" ]; then
		echo "Status: Building set for MONKS3 database"
		echo "=============================================="
		java BuildSets ../../datasets/${1}/original/${1}.data , 30 5
	elif [ ${1} == "pima" ]; then
		echo "Status: Building set for PIMA database"
		echo "=============================================="
		java BuildSets ../../datasets/${1}/original/${1}.data , 30 -normalise
	else
		echo "Status: Unrecognised set"
		echo "Status: Building default"
		echo "=============================================="
		java BuildSets ../../datasets/${1}/original/${1}.data , 30
	fi
	
	echo "=============================================="
	echo "Status: Cross validation sets built"
	
	echo -n "Status: Performing data set conversions...  "
	
	# create SOM names file
	./../../datasets/${1}/names_som.sh > ${1}.som.names
	
	# create C4.5 names file
	./../../datasets/${1}/names_c4_5_for_som.sh > ${1}.c4-5.names
	
	# perform conversion of data set from raw form to SOM_PAK and C4.5 format, creating [simulation].data
	for ((dataset = 0; dataset < 30; ++dataset))
	do
		gawk -f ./../../datasets/${1}/convert_som.awk ${1}.data.raw.${dataset}.train > ${1}.data.${dataset}.som.train
		gawk -f ./../../datasets/${1}/convert_c4_5.awk ${1}.data.raw.${dataset}.train > ${1}.data.${dataset}.c4-5.eval.train
		rm -f ${1}.data.raw.${dataset}.train
		gawk -f ./../../datasets/${1}/convert_c4_5.awk ${1}.data.raw.${dataset}.test > ${1}.data.${dataset}.c4-5.eval.test
		rm -f ${1}.data.raw.${dataset}.test
	done
	
	echo "DONE"
	
}

# ===================================================================================================
# Function : runCrossValidation(simDirectory, neuronLabelingMethod, mapDim, learningRate, learningRateDecay, kernelWidth,
#            radiusDecay, minOutcomes, heuristic, pruningConfidence, FisherConfidence, redundancy)
# Purpose  : Performs a single cross validation using provided parameters
# Return   : None
# ===================================================================================================

runCrossValidation () {
	
	# get description string for neuron labeling method (0 = neuron; 1 = K-means cluster; 2 = Ward cluster; 3 = weight-centric)
	if [ ${2} -eq 0 ]; then
		neuronLabelingMethodDescription="neuron"
	elif [ ${2} -eq 1 ]; then
		neuronLabelingMethodDescription="k-means"
	elif [ ${2} -eq 2 ]; then
		neuronLabelingMethodDescription="ward"
	elif [ ${2} -eq 3 ]; then
		neuronLabelingMethodDescription="weight-centric"
	fi
	
	# get description string for tree test heuristic (0 = gain ratio; 1 = gain)
	if [ ${9} -eq 0 ]; then
		heuristicDescription="gratio"
	elif [ ${9} -eq 1 ]; then
		heuristicDescription="gain"
	fi
	
	if [ ${compareResult} -eq 1 ]; then
		FisherDescription="na"
	else
		FisherDescription=${11}
	fi
	
	if [ -f hybridsom-c4-5_${1}_${neuronLabelingMethodDescription}_dim_${3}_lr_${4}_lrd_${5}_kw_${6}_kwd_${7}_minOut_${8}_heur_${heuristicDescription}_prunConf_${10}_fisher_${FisherDescription}_redun_${12}.accuracy ]; then
		rm -f hybridsom-c4-5_${1}_${neuronLabelingMethodDescription}_dim_${3}_lr_${4}_lrd_${5}_kw_${6}_kwd_${7}_minOut_${8}_heur_${heuristicDescription}_prunConf_${10}_fisher_${FisherDescription}_redun_${12}.accuracy
	fi
	
	if [ -f hybridsom-c4-5_${1}_${neuronLabelingMethodDescription}_dim_${3}_lr_${4}_lrd_${5}_kw_${6}_kwd_${7}_minOut_${8}_heur_${heuristicDescription}_prunConf_${10}_fisher_${FisherDescription}_redun_${12}.error ]; then
		rm -f hybridsom-c4-5_${1}_${neuronLabelingMethodDescription}_dim_${3}_lr_${4}_lrd_${5}_kw_${6}_kwd_${7}_minOut_${8}_heur_${heuristicDescription}_prunConf_${10}_fisher_${FisherDescription}_redun_${12}.error
	fi
	
	for ((simNum = 0; simNum < 30; ++simNum))
	do
		echo -n "${simNum} "
		
		# execute SOM_PAK
		./../../algorithms/SOM_PAK/vsom -din ${1}.data.${simNum}.som.train -vin ${1}.data.${simNum}.som.train -cin ${1}.${3}.${simNum}.map -cout ${1}_${neuronLabelingMethodDescription}_${3}_${4}_${5}_${6}_${7}_${8}_${heuristicDescription}_${10}_${FisherDescription}_${12}_${simNum}.map.out -lrate ${4} -lratedecay ${5} -radius ${6} -radiusdecay ${7} -rand 0 >> /dev/null 2> /dev/null
		
		# execute appropriate neuron labeling algorithm on SOM_PAK output, producing training data file for C4.5 tree building
		if [ ${2} -eq 0 ]; then
			./../../algorithms/ExampleCentricNeuron/outputLabels ${1}_${neuronLabelingMethodDescription}_${3}_${4}_${5}_${6}_${7}_${8}_${heuristicDescription}_${10}_${FisherDescription}_${12}_${simNum}.map.out ${1}.data.${simNum}.som.train ${1}.som.names -c4.5 ${1}_${neuronLabelingMethodDescription}_${3}_${4}_${5}_${6}_${7}_${8}_${heuristicDescription}_${10}_${FisherDescription}_${12}_${simNum}.data > /dev/null
		elif [ ${2} -eq 1 ]; then
			./../../algorithms/ExampleCentricClusterKMeans/outputLabels ${1}_${neuronLabelingMethodDescription}_${3}_${4}_${5}_${6}_${7}_${8}_${heuristicDescription}_${10}_${FisherDescription}_${12}_${simNum}.map.out ${1}.data.${simNum}.som.train ${1}.som.names -c4.5 ${1}_${neuronLabelingMethodDescription}_${3}_${4}_${5}_${6}_${7}_${8}_${heuristicDescription}_${10}_${FisherDescription}_${12}_${simNum}.data > /dev/null
		elif [ ${2} -eq 2 ]; then
			./../../algorithms/ExampleCentricClusterWard/outputLabels ${1}_${neuronLabelingMethodDescription}_${3}_${4}_${5}_${6}_${7}_${8}_${heuristicDescription}_${10}_${FisherDescription}_${12}_${simNum}.map.out ${1}.data.${simNum}.som.train ${1}.som.names -c4.5 ${1}_${neuronLabelingMethodDescription}_${3}_${4}_${5}_${6}_${7}_${8}_${heuristicDescription}_${10}_${FisherDescription}_${12}_${simNum}.data > /dev/null
		elif [ ${2} -eq 3 ]; then
			./../../algorithms/WeightCentricNeuron/outputLabels ${1}_${neuronLabelingMethodDescription}_${3}_${4}_${5}_${6}_${7}_${8}_${heuristicDescription}_${10}_${FisherDescription}_${12}_${simNum}.map.out ${1}.data.${simNum}.som.train ${1}.som.names -c4.5 ${1}_${neuronLabelingMethodDescription}_${3}_${4}_${5}_${6}_${7}_${8}_${heuristicDescription}_${10}_${FisherDescription}_${12}_${simNum}.data > /dev/null
		fi
		
		# prepare names file, training data, and test data set for C4.5 run
		cp ${1}.c4-5.names ${1}_${neuronLabelingMethodDescription}_${3}_${4}_${5}_${6}_${7}_${8}_${heuristicDescription}_${10}_${FisherDescription}_${12}_${simNum}.names
		cp ${1}.data.${simNum}.c4-5.eval.train ${1}_${neuronLabelingMethodDescription}_${3}_${4}_${5}_${6}_${7}_${8}_${heuristicDescription}_${10}_${FisherDescription}_${12}_${simNum}.train
		cp ${1}.data.${simNum}.c4-5.eval.test ${1}_${neuronLabelingMethodDescription}_${3}_${4}_${5}_${6}_${7}_${8}_${heuristicDescription}_${10}_${FisherDescription}_${12}_${simNum}.test
		
		# execute C4.5 to build tree
		if [ ${9} -eq 0 ]; then
			# gain ratio criterion
			./../../algorithms/C4_5/Src/c4.5 -f ${1}_${neuronLabelingMethodDescription}_${3}_${4}_${5}_${6}_${7}_${8}_${heuristicDescription}_${10}_${FisherDescription}_${12}_${simNum} -m ${8} > /dev/null
		else
			# gain criterion
			./../../algorithms/C4_5/Src/c4.5 -f ${1}_${neuronLabelingMethodDescription}_${3}_${4}_${5}_${6}_${7}_${8}_${heuristicDescription}_${10}_${FisherDescription}_${12}_${simNum} -m ${8} -g > /dev/null
		fi
		
		# check for error in tree build
		if [ $? -ne 0 ]; then
			echo -n "[error_tree] "
		else
			# execute C4.5 rule inducer, output results to tmp file
			if [ ${compareResult} = 1 ]; then
				# don't use Fisher's exact pruning if the pruning confidence is negative
				./../../algorithms/C4_5/Src/c4.5rules -f ${1}_${neuronLabelingMethodDescription}_${3}_${4}_${5}_${6}_${7}_${8}_${heuristicDescription}_${10}_${FisherDescription}_${12}_${simNum} -u -t -c ${10} -r ${12} > ${1}_${neuronLabelingMethodDescription}_${3}_${4}_${5}_${6}_${7}_${8}_${heuristicDescription}_${10}_${FisherDescription}_${12}_${simNum}.tmp
			else
				# use Fisher's exact pruning
				./../../algorithms/C4_5/Src/c4.5rules -f ${1}_${neuronLabelingMethodDescription}_${3}_${4}_${5}_${6}_${7}_${8}_${heuristicDescription}_${10}_${FisherDescription}_${12}_${simNum} -u -t -c ${10} -r ${12} -F ${11} > ${1}_${neuronLabelingMethodDescription}_${3}_${4}_${5}_${6}_${7}_${8}_${heuristicDescription}_${10}_${FisherDescription}_${12}_${simNum}.tmp
			fi
			
			# check for error in rule extraction
			if [ $? -ne 0 ]; then
				echo -n "[error_rules] "
			else
				# extract training and generalization accuracy, misclassification accuracy (which is simply 0, because it isn't defined), and unclassified accuracy (which is simply 0, because it isn't defined)
				grep "<<" ${1}_${neuronLabelingMethodDescription}_${3}_${4}_${5}_${6}_${7}_${8}_${heuristicDescription}_${10}_${FisherDescription}_${12}_${simNum}.tmp | gawk '{ if (FNR != 1) printf "%s\t0.0\t0.0\t", 100.0-$6 }' >> hybridsom-c4-5_${1}_${neuronLabelingMethodDescription}_dim_${3}_lr_${4}_lrd_${5}_kw_${6}_kwd_${7}_minOut_${8}_heur_${heuristicDescription}_prunConf_${10}_fisher_${FisherDescription}_redun_${12}.accuracy
				
				# extract training and generalization error, misclassification error (same as overall error), and unclassified error (which is simply 0, because C4.5 classifies every example)
				grep "<<" ${1}_${neuronLabelingMethodDescription}_${3}_${4}_${5}_${6}_${7}_${8}_${heuristicDescription}_${10}_${FisherDescription}_${12}_${simNum}.tmp | gawk '{ if (FNR != 1) printf "%s\t%s\t0.0\t", $6, $6 }' >> hybridsom-c4-5_${1}_${neuronLabelingMethodDescription}_dim_${3}_lr_${4}_lrd_${5}_kw_${6}_kwd_${7}_minOut_${8}_heur_${heuristicDescription}_prunConf_${10}_fisher_${FisherDescription}_redun_${12}.error
				
				# get total condition count (+1 for default rule), number of rules (+1 for default rule), and average conditions per rule (considering default rule to be a single rule with 1 condition)
				java GetRuleSetConditionCountC4_5 ${1}_${neuronLabelingMethodDescription}_${3}_${4}_${5}_${6}_${7}_${8}_${heuristicDescription}_${10}_${FisherDescription}_${12}_${simNum}.tmp >> hybridsom-c4-5_${1}_${neuronLabelingMethodDescription}_dim_${3}_lr_${4}_lrd_${5}_kw_${6}_kwd_${7}_minOut_${8}_heur_${heuristicDescription}_prunConf_${10}_fisher_${FisherDescription}_redun_${12}.accuracy
				java GetRuleSetConditionCountC4_5 ${1}_${neuronLabelingMethodDescription}_${3}_${4}_${5}_${6}_${7}_${8}_${heuristicDescription}_${10}_${FisherDescription}_${12}_${simNum}.tmp >> hybridsom-c4-5_${1}_${neuronLabelingMethodDescription}_dim_${3}_lr_${4}_lrd_${5}_kw_${6}_kwd_${7}_minOut_${8}_heur_${heuristicDescription}_prunConf_${10}_fisher_${FisherDescription}_redun_${12}.error
			fi
			
		fi
		
		if [ ${simNum} -ne 29 ]; then
			echo "" >> hybridsom-c4-5_${1}_${neuronLabelingMethodDescription}_dim_${3}_lr_${4}_lrd_${5}_kw_${6}_kwd_${7}_minOut_${8}_heur_${heuristicDescription}_prunConf_${10}_fisher_${FisherDescription}_redun_${12}.accuracy
			echo "" >> hybridsom-c4-5_${1}_${neuronLabelingMethodDescription}_dim_${3}_lr_${4}_lrd_${5}_kw_${6}_kwd_${7}_minOut_${8}_heur_${heuristicDescription}_prunConf_${10}_fisher_${FisherDescription}_redun_${12}.error
		fi
		
		# clean up
		rm -f *.map.out *.unpruned *.tree *.rules *.tmp
		
		rm -f ${1}_${neuronLabelingMethodDescription}_${3}_${4}_${5}_${6}_${7}_${8}_${heuristicDescription}_${10}_${FisherDescription}_${12}_${simNum}.names
		rm -f ${1}_${neuronLabelingMethodDescription}_${3}_${4}_${5}_${6}_${7}_${8}_${heuristicDescription}_${10}_${FisherDescription}_${12}_${simNum}.train
		rm -f ${1}_${neuronLabelingMethodDescription}_${3}_${4}_${5}_${6}_${7}_${8}_${heuristicDescription}_${10}_${FisherDescription}_${12}_${simNum}.test
		rm -f ${1}_${neuronLabelingMethodDescription}_${3}_${4}_${5}_${6}_${7}_${8}_${heuristicDescription}_${10}_${FisherDescription}_${12}_${simNum}.data
	done
	
	# calculate confidence intervals
	java AverageRuleSetResults -accuracy hybridsom-c4-5_${1}_${neuronLabelingMethodDescription}_dim_${3}_lr_${4}_lrd_${5}_kw_${6}_kwd_${7}_minOut_${8}_heur_${heuristicDescription}_prunConf_${10}_fisher_${FisherDescription}_redun_${12}.accuracy hybridsom-c4-5_${1}_${neuronLabelingMethodDescription}.accuracy.summary
	java AverageRuleSetResults -error hybridsom-c4-5_${1}_${neuronLabelingMethodDescription}_dim_${3}_lr_${4}_lrd_${5}_kw_${6}_kwd_${7}_minOut_${8}_heur_${heuristicDescription}_prunConf_${10}_fisher_${FisherDescription}_redun_${12}.error hybridsom-c4-5_${1}_${neuronLabelingMethodDescription}.error.summary
	
	echo ""
	
}

# ===================================================================================================
# Function : runSims(simDirectory)
# Purpose  : Runs simulations for specified simDirectory [assumes built data sets]
# Return   : None
# ===================================================================================================

runSims () {
	
	echo "Status: Running simulations"
	echo "=============================================="
	
	# number of current simulation
	currentSimulation=1
	
	# neuron labeling method (0 = neuron; 1 = K-means cluster; 2 = Ward cluster; 3 = weight-centric)
	neuronLabelingMethod=0
	
	# SOM map dimension: (integer) [2:inf] (No default)
	mapDimensions=()
	
	# SOM initial learning rate (floating point) [0.0:inf] (Default: 1)
	learningRates=()
	
	# SOM learning rate decay constant (floating point) (0.0:inf] (Default: 1000)
	learningRateDecays=()
	
	# SOM initial kernel width (floating point) [1.0:inf] (Default: 10)
	kernelWidths=()
	
	# SOM radius decay constant (floating point) (0.0:inf] (Default: 1000)
	kernelWidthDecays=()
	
	# minimum examples per tree test outcome: (integer) [1:inf] (Default: 2)
	minOutcomes=()
	
	# tree test heuristic: 0 = gain ratio; 1 = gain (Default: 0)
	heuristics=()
	
	# pessimistic rule pruning confidence level: (float) [0:100] (default: 25)
	pruningConfidences=()
	
	# Fisher's exact rule pruning confidence level: (float) [0:100] (default: off [negative], otherwise 25)
	fisherConfidences=()
	
	# rule pruning redundancy factor: (float) [0:10000] (default 1.0)
	redundancies=()
	
	# check that all arrays are the same length (otherwise we can't run through all parameter settings)
	if [ ${#mapDimensions[*]} -ne ${#learningRates[*]} ] || [ ${#mapDimensions[*]} -ne ${#learningRateDecays[*]} ] || [ ${#mapDimensions[*]} -ne ${#kernelWidths[*]} ] || [ ${#mapDimensions[*]} -ne ${#kernelWidthDecays[*]} ] || [ ${#mapDimensions[*]} -ne ${#minOutcomes[*]} ] || [ ${#mapDimensions[*]} -ne ${#heuristics[*]} ] || [ ${#mapDimensions[*]} -ne ${#pruningConfidences[*]} ] || [ ${#mapDimensions[*]} -ne ${#fisherConfidences[*]} ] || [ ${#mapDimensions[*]} -ne ${#redundancies[*]} ]; then
		echo "Error: parameter array sizes do not match"
		rm -f *.train *.test
		exit
	fi
	
	# check for valid labeling method
	if [ ${neuronLabelingMethod} -ne 0 ] && [ ${neuronLabelingMethod} -ne 1 ] && [ ${neuronLabelingMethod} -ne 2 ] && [ ${neuronLabelingMethod} -ne 3 ]; then
		echo "Error: Invalid neuron labeling method"
		rm -f *.train *.test *.names
		exit
	fi
	
	# pre-generate initialized maps for each of the specified map dimensions (since this is a time-consuming task)
	echo "Generating maps..."
	
	for ((position = 0; position < ${#mapDimensions[*]}; ++position))
	do
		
		if [ ! -f ${1}.${mapDimensions[position]}.0.map ]; then
			echo -n "Map dimension ${mapDimensions[position]} x ${mapDimensions[position]}: "
			
			for ((simNum = 0; simNum < 30; ++simNum))
			do
				echo -n "$simNum "
				./../../algorithms/SOM_PAK/lininit -din ${1}.data.${simNum}.som.train -cout ${1}.${mapDimensions[position]}.${simNum}.map -topol hexa -neigh gaussian -xdim ${mapDimensions[position]} -ydim ${mapDimensions[position]} > /dev/null
			done
			
			echo
		fi
		
	done
	
	echo "=============================================="
	
	for ((position = 0; position < ${#mapDimensions[*]}; ++position))
	do
		echo "Simulation ${currentSimulation}"
		echo "=============================================="
		
		echo "Map dimensions: ${mapDimensions[position]} x ${mapDimensions[position]}"
		echo "Initial learning rate: ${learningRates[position]}"
		echo "Learning rate decay constant: ${learningRateDecays[position]}"
		echo "Kernel width: ${kernelWidths[position]}"
		echo "Radius decay constant: ${kernelWidthDecays[position]}"
		
		if [ ${neuronLabelingMethod} -eq 0 ]; then
			echo "Labeling method: example-centric neuron labeling"
		elif [ ${neuronLabelingMethod} -eq 1 ]; then
			echo "Labeling method: example-centric cluster labeling (K-Means)"
		elif [ ${neuronLabelingMethod} -eq 2 ]; then
			echo "Labeling method: example-centric cluster labeling (Ward)"
		elif [ ${neuronLabelingMethod} -eq 3 ]; then
			echo "Labeling method: weight-centric neuron labeling"
		fi
		
		echo "Minimum examples per outcome: ${minOutcomes[position]}"
		
		if [ ${heuristics[position]} -eq 0 ]; then
			echo "Tree test heuristic: Gain ratio"
		else
			echo "Tree test heuristic: Gain"
		fi
		
		echo "Rule pruning confidence: ${pruningConfidences[position]}%"
		
		compareResult=$(echo "${fisherConfidences[position]} < 0.0" | bc)
		
		if [ $compareResult = 1 ]; then
			echo "Fisher pruning confidence: Not used"
		else
			echo "Fisher pruning confidence: ${fisherConfidences[position]}%"
		fi
		
		echo "Rule pruning redundancy: ${redundancies[position]}"
		
		# run 30 cross validated simulations
		runCrossValidation ${1} ${neuronLabelingMethod} ${mapDimensions[position]} ${learningRates[position]} ${learningRateDecays[position]} ${kernelWidths[position]} ${kernelWidthDecays[position]} ${minOutcomes[position]} ${heuristics[position]} ${pruningConfidences[position]} ${fisherConfidences[position]} ${redundancies[position]}
		
		echo "=============================================="
		let currentSimulation++
	done
	
	rm -f *.map
	
	echo "Status: Simulations completed"
	
}

# ===================================================================================================
# Function : cleanUp()
# Purpose  : Cleans up files created during simulation run
# Return   : None
# ===================================================================================================

cleanUp() {
	
	echo -n "Status: Cleaning up...  "
	
	# clean up created files
	rm *.names *.train *.test
	
	echo "DONE"
	
}

# ===================================================================================================
# Main script
# ===================================================================================================

export CLASSPATH=/home/ubuntu/Documents/SOMLib/somlib/algorithms/BuildSets:/home/wvheerden/Dropbox/Documents/Simulations/algorithms/AverageRunResults:/home/wvheerden/Dropbox/Documents/Simulations/algorithms/GetRuleSetConditionCountC4_5:.

validateSystem ${1}
buildSets ${1}
runSims ${1}
cleanUp ${1}
