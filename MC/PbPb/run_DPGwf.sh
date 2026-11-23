#!/usr/bin/env bash

# make sure O2DPG + O2 is loaded
[ ! "${O2DPG_ROOT}" ] && echo "Error: This needs O2DPG loaded" && exit 1
[ ! "${O2_ROOT}" ] && echo "Error: This needs O2 loaded" && exit 1

RNDSEED=${RNDSEED:-0}
NEVENTS=${NEVENTS:-1}
NWORKERS=${NWORKERS:-8}
NTIMEFRAMES=${NTIMEFRAMES:-1}
SIMENGINE=${SIMENGINE:-TGeant4}

# ----------- SETUP LOCAL CCDB CACHE --------------------------
export ALICEO2_CCDB_LOCALCACHE=$PWD/.ccdb

${O2DPG_ROOT}/MC/bin/o2dpg_sim_workflow.py \
       -eCM 5360 -col PbPb -gen pythia8  -e ${SIMENGINE} \
       -interactionRate 50000 -run 310000 -confKey "Diamond.width[2]=6" \
       -tf ${NTIMEFRAMES} -ns ${NEVENTS} -j ${NWORKERS}  -seed ${RNDSEED} \
       --readoutDets ZDC --with-ZDC
	     

export FAIRMQ_IPC_PREFIX=./
# run workflow (highly-parallel)
${O2DPG_ROOT}/MC/bin/o2_dpg_workflow_runner.py -f workflow.json -tt digi -jmax 1
