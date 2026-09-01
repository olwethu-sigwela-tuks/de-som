# ===================================================================================================
# Script        : crossValidateCN2.sh
# Usage         : ./crossValidateCN2.sh [simDirectory]
# Purpose       : Performs a cross-validated paired test on parameter data set using CN2 algorithm
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
	
	# build raw training and test sets (must still be converted to CN2 format)
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
	
	# perform conversion of data set from raw form to CN2 format, creating [simulation].data
	for ((dataset = 0; dataset < 30; ++dataset))
	do
		gawk -f ./../../datasets/${1}/convert_cn2.awk ${1}.data.raw.${dataset}.train > ${1}.data.${dataset}.train
		rm -f ${1}.data.raw.${dataset}.train
		gawk -f ./../../datasets/${1}/convert_cn2.awk ${1}.data.raw.${dataset}.test > ${1}.data.${dataset}.test
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
	
	# execute training step and evaluate training accuracy
	echo "read both ${1}.data.${5}.train" >> simscript.tmp
	echo "induce" >> simscript.tmp	
	echo "xecute all quit" >> simscript.tmp
	
	# execute validation step and evaluate validation accuracy
	echo "read both ${1}.data.${5}.test" >> simscript.tmp
	echo "xecute all quit" >> simscript.tmp
	
	echo "quit" >> simscript.tmp
	
}

# ===================================================================================================
# Function : runCrossValidation(simDirectory, errorEstimate, starSize, threshold)
# Purpose  : Performs a single cross validation using provided parameters
# Return   : None
# ===================================================================================================

runCrossValidation () {
	
	# get description string for error estimate (0 = Laplacian; 1 = Naive)
	if [ ${2} -eq 0 ]; then
		errorEstimateDescription="laplacian"
	elif [ ${2} -eq 1 ]; then
		errorEstimateDescription="naive"
	fi
	
	if [ -f cn2_${1}_${errorEstimateDescription}_star_${3}_thr_${4}.accuracy ]; then
		rm -f cn2_${1}_${errorEstimateDescription}_star_${3}_thr_${4}.accuracy
	fi
	
	if [ -f cn2_${1}_${errorEstimateDescription}_star_${3}_thr_${4}.error ]; then
		rm -f cn2_${1}_${errorEstimateDescription}_star_${3}_thr_${4}.error
	fi
	
	for ((simNum = 0; simNum < 30; ++simNum))
	do
		echo -n "${simNum} "
		
		# execute CN2 using built script, output results to out.tmp
		buildScript ${1} ${2} ${3} ${4} ${simNum}
		sleep 1
		./../../algorithms/CN2/cn < simscript.tmp > out.tmp
		
		# extract training and generalization accuracy, misclassification accuracy (which is simply 0, because it isn't defined), and unclassified accuracy (which is simply 0, because it isn't defined)
		grep "Overall accuracy:" out.tmp | gawk '{ printf "%s\t0.0\t0.0\t", $3 }' >> cn2_${1}_${errorEstimateDescription}_star_${3}_thr_${4}.accuracy
		
		# extract training and generalization error, misclassification error (same as overall error), and unclassified error (which is simply 0, because C4.5 classifies every example)
		grep "Overall accuracy:" out.tmp | gawk '{ printf "%.15f\t%.15f\t0.0\t", 100.0-$3, 100.0-$3 }' >> cn2_${1}_${errorEstimateDescription}_star_${3}_thr_${4}.error
		
		# extract condition count (including default rule)
		conditionCount=$(grep "Default-inclusive condition count:" out.tmp | gawk '{ if (FNR == 1) printf "%s", $4 }')
		# extract rule count (including default rule)
		ruleCount=$(grep "Rule count (including default):" out.tmp | gawk '{ if (FNR == 1) printf "%s", $5 }')
		# extract average conditions per rule (including default rule, which is considereda single rule with 1 condition)
		aveRuleLength=$(grep "Average overall rule length:" out.tmp | gawk '{ if (FNR == 1) printf "%s", $5 }')
		
		# output total condition count (+1 for default rule), number of rules (including default rule), and average conditions per rule (considering default rule to be a single rule with 1 condition)
		echo -n "$conditionCount" $'\t' >> cn2_${1}_${errorEstimateDescription}_star_${3}_thr_${4}.accuracy
		echo -n "$ruleCount" $'\t' >> cn2_${1}_${errorEstimateDescription}_star_${3}_thr_${4}.accuracy
		echo -n "$aveRuleLength" >> cn2_${1}_${errorEstimateDescription}_star_${3}_thr_${4}.accuracy
		
		echo -n "$conditionCount" $'\t' >> cn2_${1}_${errorEstimateDescription}_star_${3}_thr_${4}.error
		echo -n "$ruleCount" $'\t' >> cn2_${1}_${errorEstimateDescription}_star_${3}_thr_${4}.error
		echo -n "$aveRuleLength" >> cn2_${1}_${errorEstimateDescription}_star_${3}_thr_${4}.error
		
		if [ ${simNum} -ne 29 ]; then
			echo "" >> cn2_${1}_${errorEstimateDescription}_star_${3}_thr_${4}.accuracy
			echo "" >> cn2_${1}_${errorEstimateDescription}_star_${3}_thr_${4}.error
		fi
		
		# clean up
		rm -f out.tmp simscript.tmp
	done
	
	# calculate confidence intervals
	java AverageRuleSetResults -accuracy cn2_${1}_${errorEstimateDescription}_star_${3}_thr_${4}.accuracy cn2.accuracy.summary
	java AverageRuleSetResults -error cn2_${1}_${errorEstimateDescription}_star_${3}_thr_${4}.error cn2.error.summary
	
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
	
	# star size (integer) [1:200] (Default: 5)
	starSizes=(1 101 150 51 76 175 125 26 38 138 188 88 63 163 113 13 20 119 169 69 94 194 144 45 32 132 181 82 57 156 107 7 10 110 160 60 85 184 135 35 48 147 197 97 73 172 122 23 17 116 166 66 91 191 141 41 29 128 178 79 54 153 104 4 6 105 155 55 80 180 130 31 43 142 192 93 68 167 118 18 24 124 174 74 99 198 149 49 37 136 186 87 62 161 111 12 9 108 158 59 83 183 133 34 46 146 195 96 71 170 121 21 15 114 164 65 90 189 139 40 27 127 177 77 52 152 102 3 3 103 153 53 78 177 128 28 41 140 190 90 66 165 115 16 22 121 171 72 97 196 146 47 34 134 184 84 59 159 109 10 13 112 162 62 87 187 137 38 50 149 199 100 75 174 125 25 19 118 168 69 94 193 143 44 31 131 181 81 56 156 106 6 5 104 154 55 80 179 129 30 42 142 191 92 67 167 117 17 24 123 173 73 98 198 148 48 36 135 185 86 61 160 111 11 8 107 157 58 83 182 132 33 45 145 195 95 70 170 120 20 14 114 163 64 89 188 139 39 27 126 176 76 52 151 101 2 2 102 151 52 77 176 127 27 39 139 189 89 64 164 114 15 21 120 170 71 95 195 145 46 33 133 183 83 58 158 108 8 11 111 161 61 86 186 136 36 49 148 198 99 74 173 123 24 18 117 167 67 92 192 142 43 30 130 179 80 55 155 105 5 7 106 156 57 81 181 131 32 44 144 193 94 69 169 119 19 25 125 175 75 100 200 150 50 38 137 187 88 63 162 113 13 10 109 159 60 85 184 134 35 47 147 197 97 72 172 122 22 16 116 165 66 91 190 141 41 29 128 178 78 53 153 103 4 3 102 152 53 78 177 127 28 40 140 190 90 65 165 115 15 22 121 171 71 96 196 146 46 34 134 183 84 59 158 109 9 12 112 162 62 87 186 137 37 50 149 199 99 74 174 124 25 18 118 168 68 93 193 143 43 31 130 180 81 56 155 106 6 4 104 154 54 79 179 129 29 42 141 191 92 67 166 116 17 23 123 172 73 98 197 148 48 36 135 185 85 60 160 110 11 8 107 157 57 82 182 132 32 45 144 194 95 70 169 120 20 14 113 163 64 88 188 138 39 26 126 176 76 51 151 101 1)
	
	# threshold (floating point) [0.0:inf] (Default: 0.0)
	thresholds=(0 100 50 150 75 175 25 125 62.5 162.5 12.5 112.5 37.5 137.5 87.5 187.5 93.75 193.75 43.75 143.75 18.75 118.75 68.75 168.75 31.25 131.25 81.25 181.25 56.25 156.25 6.25 106.25 53.125 153.125 3.125 103.125 28.125 128.125 78.125 178.125 15.625 115.625 65.625 165.625 90.625 190.625 40.625 140.625 46.875 146.875 96.875 196.875 71.875 171.875 21.875 121.875 84.375 184.375 34.375 134.375 9.375 109.375 59.375 159.375 79.6875 179.6875 29.6875 129.6875 4.6875 104.6875 54.6875 154.6875 42.1875 142.1875 92.1875 192.1875 67.1875 167.1875 17.1875 117.1875 23.4375 123.4375 73.4375 173.4375 98.4375 198.4375 48.4375 148.4375 60.9375 160.9375 10.9375 110.9375 35.9375 135.9375 85.9375 185.9375 26.5625 126.5625 76.5625 176.5625 51.5625 151.5625 1.5625 101.5625 89.0625 189.0625 39.0625 139.0625 14.0625 114.0625 64.0625 164.0625 70.3125 170.3125 20.3125 120.3125 45.3125 145.3125 95.3125 195.3125 7.8125 107.8125 57.8125 157.8125 82.8125 182.8125 32.8125 132.8125 66.40625 166.40625 16.40625 116.40625 41.40625 141.40625 91.40625 191.40625 3.90625 103.90625 53.90625 153.90625 78.90625 178.90625 28.90625 128.90625 35.15625 135.15625 85.15625 185.15625 60.15625 160.15625 10.15625 110.15625 97.65625 197.65625 47.65625 147.65625 22.65625 122.65625 72.65625 172.65625 13.28125 113.28125 63.28125 163.28125 88.28125 188.28125 38.28125 138.28125 50.78125 150.78125 0.78125 100.78125 25.78125 125.78125 75.78125 175.78125 82.03125 182.03125 32.03125 132.03125 7.03125 107.03125 57.03125 157.03125 44.53125 144.53125 94.53125 194.53125 69.53125 169.53125 19.53125 119.53125 39.84375 139.84375 89.84375 189.84375 64.84375 164.84375 14.84375 114.84375 77.34375 177.34375 27.34375 127.34375 2.34375 102.34375 52.34375 152.34375 58.59375 158.59375 8.59375 108.59375 33.59375 133.59375 83.59375 183.59375 21.09375 121.09375 71.09375 171.09375 96.09375 196.09375 46.09375 146.09375 92.96875 192.96875 42.96875 142.96875 17.96875 117.96875 67.96875 167.96875 30.46875 130.46875 80.46875 180.46875 55.46875 155.46875 5.46875 105.46875 11.71875 111.71875 61.71875 161.71875 86.71875 186.71875 36.71875 136.71875 74.21875 174.21875 24.21875 124.21875 49.21875 149.21875 99.21875 199.21875 99.609375 199.609375 49.609375 149.609375 24.609375 124.609375 74.609375 174.609375 37.109375 137.109375 87.109375 187.109375 62.109375 162.109375 12.109375 112.109375 5.859375 105.859375 55.859375 155.859375 80.859375 180.859375 30.859375 130.859375 68.359375 168.359375 18.359375 118.359375 43.359375 143.359375 93.359375 193.359375 46.484375 146.484375 96.484375 196.484375 71.484375 171.484375 21.484375 121.484375 83.984375 183.984375 33.984375 133.984375 8.984375 108.984375 58.984375 158.984375 52.734375 152.734375 2.734375 102.734375 27.734375 127.734375 77.734375 177.734375 15.234375 115.234375 65.234375 165.234375 90.234375 190.234375 40.234375 140.234375 19.921875 119.921875 69.921875 169.921875 94.921875 194.921875 44.921875 144.921875 57.421875 157.421875 7.421875 107.421875 32.421875 132.421875 82.421875 182.421875 76.171875 176.171875 26.171875 126.171875 1.171875 101.171875 51.171875 151.171875 38.671875 138.671875 88.671875 188.671875 63.671875 163.671875 13.671875 113.671875 73.046875 173.046875 23.046875 123.046875 48.046875 148.046875 98.046875 198.046875 10.546875 110.546875 60.546875 160.546875 85.546875 185.546875 35.546875 135.546875 29.296875 129.296875 79.296875 179.296875 54.296875 154.296875 4.296875 104.296875 91.796875 191.796875 41.796875 141.796875 16.796875 116.796875 66.796875 166.796875 33.203125 133.203125 83.203125 183.203125 58.203125 158.203125 8.203125 108.203125 95.703125 195.703125 45.703125 145.703125 20.703125 120.703125 70.703125 170.703125 64.453125 164.453125 14.453125 114.453125 39.453125 139.453125 89.453125 189.453125 1.953125 101.953125 51.953125 151.953125 76.953125 176.953125 26.953125 126.953125 86.328125 186.328125 36.328125 136.328125 11.328125 111.328125 61.328125 161.328125 48.828125 148.828125 98.828125 198.828125 73.828125 173.828125 23.828125 123.828125 17.578125 117.578125 67.578125 167.578125 92.578125 192.578125 42.578125 142.578125 55.078125 155.078125 5.078125 105.078125 30.078125 130.078125 80.078125 180.078125 59.765625 159.765625 9.765625 109.765625 34.765625 134.765625 84.765625 184.765625 22.265625 122.265625 72.265625 172.265625 97.265625 197.265625 47.265625 147.265625 41.015625 141.015625 91.015625 191.015625 66.015625 166.015625 16.015625 116.015625 78.515625 178.515625 28.515625 128.515625 3.515625 103.515625 53.515625 153.515625 6.640625 106.640625 56.640625 156.640625 81.640625 181.640625 31.640625 131.640625 69.140625 169.140625 19.140625 119.140625 44.140625 144.140625 94.140625 194.140625 87.890625 187.890625 37.890625 137.890625 12.890625 112.890625 62.890625 162.890625 25.390625 125.390625 75.390625 175.390625 50.390625 150.390625 0.390625 100.390625)
	
	# error estimate: 0 = Laplacian; 1 = Naive (Default: 0)
	errorEstimates=(0 1 0 1 1 0 1 0 1 0 1 0 0 1 0 1 0 1 0 1 1 0 1 0 1 0 1 0 0 1 0 1 1 0 1 0 0 1 0 1 0 1 0 1 1 0 1 0 1 0 1 0 0 1 0 1 0 1 0 1 1 0 1 0 1 0 1 0 0 1 0 1 0 1 0 1 1 0 1 0 1 0 1 0 0 1 0 1 0 1 0 1 1 0 1 0 0 1 0 1 1 0 1 0 1 0 1 0 0 1 0 1 0 1 0 1 1 0 1 0 1 0 1 0 0 1 0 1 0 1 0 1 1 0 1 0 1 0 1 0 0 1 0 1 0 1 0 1 1 0 1 0 1 0 1 0 0 1 0 1 1 0 1 0 0 1 0 1 0 1 0 1 1 0 1 0 1 0 1 0 0 1 0 1 0 1 0 1 1 0 1 0 1 0 1 0 0 1 0 1 0 1 0 1 1 0 1 0 1 0 1 0 0 1 0 1 0 1 0 1 1 0 1 0 0 1 0 1 1 0 1 0 1 0 1 0 0 1 0 1 0 1 0 1 1 0 1 0 1 0 1 0 0 1 0 1 1 0 1 0 0 1 0 1 0 1 0 1 1 0 1 0 1 0 1 0 0 1 0 1 0 1 0 1 1 0 1 0 0 1 0 1 1 0 1 0 1 0 1 0 0 1 0 1 0 1 0 1 1 0 1 0 1 0 1 0 0 1 0 1 0 1 0 1 1 0 1 0 1 0 1 0 0 1 0 1 0 1 0 1 1 0 1 0 1 0 1 0 0 1 0 1 1 0 1 0 0 1 0 1 0 1 0 1 1 0 1 0 1 0 1 0 0 1 0 1 0 1 0 1 1 0 1 0 1 0 1 0 0 1 0 1 0 1 0 1 1 0 1 0 1 0 1 0 0 1 0 1 0 1 0 1 1 0 1 0 0 1 0 1 1 0 1 0 1 0 1 0 0 1 0 1 0 1 0 1 1 0 1 0 1 0 1 0 0 1 0 1 0 1 0 1 1 0 1 0 1 0 1 0 0 1 0 1 0 1 0 1 1 0 1 0 1 0 1 0 0 1 0 1 1 0 1 0 0 1 0 1 0 1 0 1 1 0 1 0 1 0 1 0 0 1 0 1 0 1 0 1 1 0 1 0)
	
	# check that all arrays are the same length (otherwise we can't run through all parameter settings)
	if [ ${#starSizes[*]} -ne ${#thresholds[*]} ] || [ ${#starSizes[*]} -ne ${#errorEstimates[*]} ]; then
		echo "Error: parameter array sizes do not match"
		rm -f *.train *.test
		exit
	fi
	
	for ((position = 0; position < ${#starSizes[*]}; ++position))
	do
		echo "Simulation $currentSimulation"
		echo "=============================================="
		
		if [ ${errorEstimates[position]} -eq 0 ]; then
			echo "Error estimate: Laplacian"
		else
			echo "Error estimate: Naive"
		fi
		
		echo "Star size: ${starSizes[position]}"
		echo "Threshold: ${thresholds[position]}"
		
		# run 30 cross validated simulations
		runCrossValidation ${1} ${errorEstimates[position]} ${starSizes[position]} ${thresholds[position]}
		
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
	rm -f *.train *.test
	
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
