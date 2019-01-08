import os
import subprocess
import math
import time

def filterLines(lines):
    for idx, line in enumerate(lines[:]):
        words = line.split(" ")
        if len(words) < 3 or (len(words) > 0 and len(words[0]) > 0 and words[0][0] == "#"):
            lines.remove(line)
    return lines

def checkCorrectParamNames(lines):
    netParamFile = open("model/networkParameters.py")
    netParamLines = netParamFile.readlines()
    netParamNames = [line.split(" ")[0] for line in netParamLines]
    for idx, line in enumerate(lines):
        words = line.split(" ")
        for word in words:
            if len(word) > 0:
                paramName = word
                break

        print("Param name: ", paramName)
        if not (paramName == "Default" or paramName in netParamNames):
            print("")
            print("A parameter name in the file does not exist!")
            print("Line: ", line)
            print()
            netParamFile.close()
            quit()
    netParamFile.close()


def getJobs(lines):
    paramTestNum = None
    jobs = []
    paramStack = []
    valueStack = []
    for idx, line in enumerate(lines):
        words = line.split(" ")
        print(words)

        otherWordAppeared = False
        depthCount = 0
        for word in words[:]:
            if word == "":
                if not otherWordAppeared:
                    depthCount += 1
                words.remove(word)
            else:
                otherWordAppeared = True

        if depthCount > len(paramStack):
            print("Something went wrong with line: ", line)
            print("The depth (number of spaces at start) is bigger than the depth")
            quit()

        while depthCount < len(paramStack):
            del paramStack[-1]
            del valueStack[-1]

        if len(words) != 3:
            print("Line is not correctly formatted: ", line)
            quit()

        paramName =  words[0]
        paramVal =  words[1]
        if words[2][:-1] == "&":
            paramTestNum = None
        elif words[2][-3:] == "x:\n":
            paramTestNum =  int(words[2][:-3]) #remove the "x:\n"
        else:
            print("Error in reading line: ", line)
            print("Third word is neither a & nor does it indicate the number of tests.")
            quit()

        if paramTestNum is not None:
            jobs.append((paramStack + [paramName],valueStack + [paramVal], paramTestNum))
        else:
            paramStack.append(paramName)
            valueStack.append(paramVal)

    return jobs


def displayJobs(jobs):
    for idx, job in enumerate(jobs):
        print("")
        print("Job ", idx + 1, ":")
        for paramIdx in range(len(job[0])):
            print(job[0][paramIdx], "=", job[1][paramIdx])
        print("Will be run ", job[2], " times.")
    print("")


def runJobs(email):

	fileName = "batch.txt"
	script = open(fileName, "w+")

	data = "#!/bin/bash\n"\
		+ "#SBATCH --time=00:15:00\n" \
		+ "#SBATCH --partition=gpu
		+ "#SBATCH --mem=4000" \
	       + "\n#SBATCH --nodes=1\n#SBATCH --mail-type=ALL\n#SBATCH --mail-user=" + email + "\n" \
	       + "module load Maven\n" \
	       + "mvn exec:java -Dexec.mainClass=src.main.java.Experiment.java -Dexec.args=\""+str("%s %s %s" % (tweakNumber, idx, trainAgainstPlayer))+"\"\n"
	script.write(data)
	script.close()
	try:
	    subprocess.call(["sbatch" , fileName])
	    pass
	except OSError:
	    script.close()
	    print("Command sbatch not found or filename invalid!")
	    print("Filename: ", fileName)
	    
	os.remove(fileName)

	print("Submitted job: ", fileName)
	print(data)
	time.sleep(1)
	idx += 1;


if __name__ == '__main__':
    email = "c.roest@me.com"
    runJobs(email)
    try:
            subprocess.call(["squeue", "-u", "s2701782"])
            pass
    except OSError:
        print("I tried to give you an overview of the jobs, but it did not work :(")
