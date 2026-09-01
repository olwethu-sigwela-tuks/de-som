# ===================================================================================================
# Script        : crossValidateHybridSOM_CN2.sh
# Usage         : ./crossValidateHybridSOM_CN2.sh [simDirectory]
# Purpose       : Performs a cross-validated paired test on parameter data set using HybridSOM
#                 framework configured with CN2
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
	
	# check for CN2 data conversion script
	echo -n "Data conversion script \"${1}/convert_cn2.awk\"...  "
	if [ ! -f ./../../datasets/${1}/convert_cn2.awk ]; then
		echo "NOT FOUND"
		exit
	fi
	echo "FOUND"
	
	# check for CN2 program
	echo -n "CN2 program...  "
	if [ ! -f ../../algorithms/CN2/cn ]; then
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
	
	# build raw training and test sets (must still be converted to SOM_PAK and CN2 formats)
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
	./../../datasets/${1}/names_som.sh > ${1}.names
	
	# perform conversion of data set from raw form to SOM_PAK and CN2 format, creating [simulation].data
	for ((dataset = 0; dataset < 30; ++dataset))
	do
		gawk -f ./../../datasets/${1}/convert_som.awk ${1}.data.raw.${dataset}.train > ${1}.data.${dataset}.som.train
		gawk -f ./../../datasets/${1}/convert_cn2.awk ${1}.data.raw.${dataset}.train > ${1}.data.${dataset}.cn2.eval.train
		rm -f ${1}.data.raw.${dataset}.train
		gawk -f ./../../datasets/${1}/convert_cn2.awk ${1}.data.raw.${dataset}.test > ${1}.data.${dataset}.cn2.eval.test
		rm -f ${1}.data.raw.${dataset}.test
	done
	
	echo "DONE"
	
}

# ===================================================================================================
# Function : buildScript(simDirectory, errorEstimate, starSize, threshold, simNum)
# Purpose  : Builds simulation script for CN2 based on provided parameters [outputs to simscript.tmp]
# Return   : None (generates simscript.tmp)
# ===================================================================================================

buildScript () {
	
	# simulation parameters
	echo "display none quit" > simscript.tmp
	echo "algorithm ordered" >> simscript.tmp
	if [ ${2} -eq 0 ]; then
		echo "error laplacian" >> simscript.tmp
	else
		echo "error naive" >> simscript.tmp
	fi
	echo "star ${3}" >> simscript.tmp
	echo "threshold ${4}" >> simscript.tmp
	
	# execute training step 
	echo "read both ${1}.data.${5}.cn2.train" >> simscript.tmp
	echo "induce" >> simscript.tmp	
	
	# execute training data and evaluate training accuracy
	echo "read both ${1}.data.${5}.cn2.eval.train" >> simscript.tmp
	echo "xecute all quit" >> simscript.tmp
	
	# execute validation step and evaluate validation accuracy
	echo "read both ${1}.data.${5}.cn2.eval.test" >> simscript.tmp
	echo "xecute all quit" >> simscript.tmp
	
	echo "quit" >> simscript.tmp
	
}

# ===================================================================================================
# Function : runCrossValidation(simDirectory, neuronLabelingMethod, mapDim, learningRate, learningRateDecay, kernelWidth,
#            radiusDecay, errorEstimate, starSize, threshold)
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
	
	# get description string for error estimate (0 = Laplacian; 1 = Naive)
	if [ ${8} -eq 0 ]; then
		errorEstimateDescription="laplacian"
	elif [ ${8} -eq 1 ]; then
		errorEstimateDescription="naive"
	fi
	
	if [ -f hybridsom-cn2_${1}_${neuronLabelingMethodDescription}_dim_${3}_lr_${4}_lrd_${5}_kw_${6}_kwd_${7}_${errorEstimateDescription}_star_${9}_thr_${10}.accuracy ]; then
		rm -f hybridsom-cn2_${1}_${neuronLabelingMethodDescription}_dim_${3}_lr_${4}_lrd_${5}_kw_${6}_kwd_${7}_${errorEstimateDescription}_star_${9}_thr_${10}.accuracy
	fi
	
	if [ -f hybridsom-cn2_${1}_${neuronLabelingMethodDescription}_dim_${3}_lr_${4}_lrd_${5}_kw_${6}_kwd_${7}_${errorEstimateDescription}_star_${9}_thr_${10}.error ]; then
		rm -f hybridsom-cn2_${1}_${neuronLabelingMethodDescription}_dim_${3}_lr_${4}_lrd_${5}_kw_${6}_kwd_${7}_${errorEstimateDescription}_star_${9}_thr_${10}.error
	fi
	
	for ((simNum = 0; simNum < 30; ++simNum))
	do
		echo -n "${simNum} "
		
		# execute SOM_PAK
		./../../algorithms/SOM_PAK/vsom -din ${1}.data.${simNum}.som.train -vin ${1}.data.${simNum}.som.train -cin ${1}.${3}.${simNum}.map -cout ${1}.${3}.${simNum}.map.out -lrate ${4} -lratedecay ${5} -radius ${6} -radiusdecay ${7} -rand 0 >> /dev/null 2> /dev/null
		
		# execute appropriate neuron labeling algorithm on SOM_PAK output
		if [ ${2} -eq 0 ]; then
			./../../algorithms/ExampleCentricNeuron/outputLabels ${1}.${3}.${simNum}.map.out ${1}.data.${simNum}.som.train ${1}.names -cn2 ${1}.data.${simNum}.cn2.train > /dev/null
		elif [ ${2} -eq 1 ]; then
			./../../algorithms/ExampleCentricClusterKMeans/outputLabels ${1}.${3}.${simNum}.map.out ${1}.data.${simNum}.som.train ${1}.names -cn2 ${1}.data.${simNum}.cn2.train > /dev/null
		elif [ ${2} -eq 2 ]; then
			./../../algorithms/ExampleCentricClusterWard/outputLabels ${1}.${3}.${simNum}.map.out ${1}.data.${simNum}.som.train ${1}.names -cn2 ${1}.data.${simNum}.cn2.train > /dev/null
		elif [ ${2} -eq 3 ]; then
			./../../algorithms/WeightCentricNeuron/outputLabels ${1}.${3}.${simNum}.map.out ${1}.data.${simNum}.som.train ${1}.names -cn2 ${1}.data.${simNum}.cn2.train > /dev/null
		fi
		
		rm -f *.map.out
		
		# execute CN2 using built script, output results to out.tmp
		buildScript ${1} ${8} ${9} ${10} ${simNum}
		./../../algorithms/CN2/cn < simscript.tmp > out.tmp
		
		rm -f ${1}.data.${simNum}.cn2.train
		
		# extract training and generalization accuracy, misclassification accuracy (which is simply 0, because it isn't defined), and unclassified accuracy (which is simply 0, because it isn't defined)
		grep "Overall accuracy:" out.tmp | gawk '{ printf "%s\t0.0\t0.0\t", $3 }' >> hybridsom-cn2_${1}_${neuronLabelingMethodDescription}_dim_${3}_lr_${4}_lrd_${5}_kw_${6}_kwd_${7}_${errorEstimateDescription}_star_${9}_thr_${10}.accuracy
		
		# extract training and generalization error, misclassification error (same as overall error), and unclassified error (which is simply 0, because C4.5 classifies every example)
		grep "Overall accuracy:" out.tmp | gawk '{ printf "%.15f\t%.15f\t0.0\t", 100.0-$3, 100.0-$3 }' >> hybridsom-cn2_${1}_${neuronLabelingMethodDescription}_dim_${3}_lr_${4}_lrd_${5}_kw_${6}_kwd_${7}_${errorEstimateDescription}_star_${9}_thr_${10}.error
		
		# extract condition count (including default rule)
		conditionCount=$(grep "Default-inclusive condition count:" out.tmp | gawk '{ if (FNR == 1) printf "%s", $4 }')
		# extract rule count (including default rule)
		ruleCount=$(grep "Rule count (including default):" out.tmp | gawk '{ if (FNR == 1) printf "%s", $5 }')
		# extract average conditions per rule (including default rule, which is considereda single rule with 1 condition)
		aveRuleLength=$(grep "Average overall rule length:" out.tmp | gawk '{ if (FNR == 1) printf "%s", $5 }')
		
		rm -f out.tmp
		
		# output total condition count (+1 for default rule), number of rules (including default rule), and average conditions per rule (considering default rule to be a single rule with 1 condition)
		echo -n "$conditionCount" $'\t' >> hybridsom-cn2_${1}_${neuronLabelingMethodDescription}_dim_${3}_lr_${4}_lrd_${5}_kw_${6}_kwd_${7}_${errorEstimateDescription}_star_${9}_thr_${10}.accuracy
		echo -n "$ruleCount" $'\t' >> hybridsom-cn2_${1}_${neuronLabelingMethodDescription}_dim_${3}_lr_${4}_lrd_${5}_kw_${6}_kwd_${7}_${errorEstimateDescription}_star_${9}_thr_${10}.accuracy
		echo -n "$aveRuleLength" >> hybridsom-cn2_${1}_${neuronLabelingMethodDescription}_dim_${3}_lr_${4}_lrd_${5}_kw_${6}_kwd_${7}_${errorEstimateDescription}_star_${9}_thr_${10}.accuracy
		
		echo -n "$conditionCount" $'\t' >> hybridsom-cn2_${1}_${neuronLabelingMethodDescription}_dim_${3}_lr_${4}_lrd_${5}_kw_${6}_kwd_${7}_${errorEstimateDescription}_star_${9}_thr_${10}.error
		echo -n "$ruleCount" $'\t' >> hybridsom-cn2_${1}_${neuronLabelingMethodDescription}_dim_${3}_lr_${4}_lrd_${5}_kw_${6}_kwd_${7}_${errorEstimateDescription}_star_${9}_thr_${10}.error
		echo -n "$aveRuleLength" >> hybridsom-cn2_${1}_${neuronLabelingMethodDescription}_dim_${3}_lr_${4}_lrd_${5}_kw_${6}_kwd_${7}_${errorEstimateDescription}_star_${9}_thr_${10}.error
		
		if [ ${simNum} -ne 29 ]; then
			echo "" >> hybridsom-cn2_${1}_${neuronLabelingMethodDescription}_dim_${3}_lr_${4}_lrd_${5}_kw_${6}_kwd_${7}_${errorEstimateDescription}_star_${9}_thr_${10}.accuracy
			echo "" >> hybridsom-cn2_${1}_${neuronLabelingMethodDescription}_dim_${3}_lr_${4}_lrd_${5}_kw_${6}_kwd_${7}_${errorEstimateDescription}_star_${9}_thr_${10}.error
		fi
		
		# clean up
		rm -f simscript.tmp
	done
	
	# calculate confidence intervals
	java AverageRuleSetResults -accuracy hybridsom-cn2_${1}_${neuronLabelingMethodDescription}_dim_${3}_lr_${4}_lrd_${5}_kw_${6}_kwd_${7}_${errorEstimateDescription}_star_${9}_thr_${10}.accuracy hybridsom-cn2_${1}_${neuronLabelingMethodDescription}.accuracy.summary
	java AverageRuleSetResults -error hybridsom-cn2_${1}_${neuronLabelingMethodDescription}_dim_${3}_lr_${4}_lrd_${5}_kw_${6}_kwd_${7}_${errorEstimateDescription}_star_${9}_thr_${10}.error hybridsom-cn2_${1}_${neuronLabelingMethodDescription}.error.summary
	
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
	mapDimensions=(2 3 4)
	
	# SOM initial learning rate (floating point) [0.0:inf] (Default: 1)
	learningRates=(1.2 4.2 9.6)
	
	# SOM learning rate decay constant (floating point) (0.0:inf] (Default: 1000)
	learningRateDecays=(100.0 200.1 300.2)
	
	# SOM initial kernel width (floating point) [1.0:inf] (Default: 10)
	kernelWidths=(10 12 15)
	
	# SOM radius decay constant (floating point) (0.0:inf] (Default: 1000)
	kernelWidthDecays=(100.2 200.3 300.4)
	
	# star size (integer) [1:200] (Default: 5)
	starSizes=(1 2 3)
	
	# threshold (floating point) [0.0:inf] (Default: 0.0)
	thresholds=(10 20 30)
	
	# error estimate: 0 = Laplacian; 1 = Naive (Default: 0)
	errorEstimates=(0 1 1)
	
	# check that all arrays are the same length (otherwise we can't run through all parameter settings)
	if [ ${#mapDimensions[*]} -ne ${#learningRates[*]} ] || [ ${#mapDimensions[*]} -ne ${#learningRateDecays[*]} ] || [ ${#mapDimensions[*]} -ne ${#kernelWidths[*]} ] || [ ${#mapDimensions[*]} -ne ${#kernelWidthDecays[*]} ] || [ ${#mapDimensions[*]} -ne ${#starSizes[*]} ] || [ ${#mapDimensions[*]} -ne ${#thresholds[*]} ] || [ ${#mapDimensions[*]} -ne ${#errorEstimates[*]} ]; then
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
		
		if [ ${errorEstimates[position]} -eq 0 ]; then
			echo "Error estimate: Laplacian"
		else
			echo "Error estimate: Naive"
		fi
		
		echo "Star size: ${starSizes[position]}"
		echo "Threshold: ${thresholds[position]}"
		
		# run 30 cross validated simulations
		runCrossValidation ${1} ${neuronLabelingMethod} ${mapDimensions[position]} ${learningRates[position]} ${learningRateDecays[position]} ${kernelWidths[position]} ${kernelWidthDecays[position]} ${errorEstimates[position]} ${starSizes[position]} ${thresholds[position]}
		
		echo "=============================================="
		let currentSimulation++
	done
	
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
	rm -f *.names *.som.train *.cn2.eval.train *cn2.eval.test *.map
	
	echo "DONE"
	
}

# ===================================================================================================
# Main script
# ===================================================================================================

export CLASSPATH=/home/ubuntu/Documents/SOMLib/somlib/algorithms/BuildSets:/home/wvheerden/Dropbox/Documents/Simulations/algorithms/AverageRunResults:.

validateSystem ${1}
buildSets ${1}
runSims ${1}
cleanUp ${1}
