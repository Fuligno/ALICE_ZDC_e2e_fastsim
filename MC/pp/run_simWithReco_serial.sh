export ALICEO2_CCDB_LOCALCACHE=$PWD/.ccdb
export IGNORE_VALIDITYCHECK_OF_CCDB_LOCALCACHE=ON

o2-sim-serial -g external -n 1 --configKeyValues "GeneratorExternal.fileName=$HOME/alice/AEGIS/GeneratorSpectators/GeneratorSpectators.C;GeneratorExternal.funcName=GeneratorSingleNeutron();SimCutParams.maxRTracking=100;ZDCSimParam.recordSpatialResponse=true" -m ABSO DIPO PIPE SHIL ZDC --readoutDetectors ZDC --run 302000

o2-sim-digitizer-workflow  -b --onlyDet ZDC

o2-zdc-digits-reco -b --run

