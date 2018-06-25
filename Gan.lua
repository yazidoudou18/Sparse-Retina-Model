-- Make sure this points to your PetaVision directory
local pvDir = os.getenv("HOME") .. "/workspace/OpenPV"; 
package.path = package.path .. ";" .. pvDir .. "/parameterWrapper/?.lua";
local pv = require "PVModule";

-- Basic params

local batchSize     = 10;
local batchWidth    = 1;
local threads       = 1;
local rows          = 1;
local cols          = 1;

local numSamples    = 40;
local epochs        = 1;
local displayPeriod = 500;
local stopTime      = numSamples / batchSize * displayPeriod * epochs;
local cpInterval    = 5;

local folderName    = "Train_Gan";

local inputPath     = "/home/twatkins/Workspace/Retina/input/mixed_cifar4test.txt";
local GanOnPath     = "/home/twatkins/Workspace/Retina/output/GanOn.pvp";
local GanOffPath     = "/home/twatkins/Workspace/Retina/output/GanOff.pvp";
local inputWidth    = 32;
local inputHeight   = 32;
local imageFeatures = 1;
local pvpFeatures   = 32;
local batchmethod   = "random";
local patch         = 16;
local stride        = 1;
local basePhase     = 1;
--local dictionary    = inputFeatures * stride * stride * 4; -- 4x overcomplete
local dictionary    = 1024;
local thresh        = 0.01; --0.005;
local learningRate  = 0.1;
local imagelearningRate  = 0.009;

-- Column --

-- Uncomment this to load weights from the specified folder
local loadWeightsFrom = nil; --folderName .. "/weights/";

local pvParams = {
   column = {
      groupType = "HyPerCol";
      startTime                           = 0;
      dt                                  = 1;
      stopTime                            = stopTime;
      progressInterval                    = displayPeriod;
      writeProgressToErr                  = true;
      verifyWrites                        = false;
      outputPath                          = "../output/" .. folderName;
      printParamsFilename                 = folderName .. ".params";
      randomSeed                          = 1234567890;
      nx                                  = inputWidth;
      ny                                  = inputHeight;
      nbatch                              = batchSize;
      checkpointWrite                     = true;
      checkpointWriteDir                  = "../output/" .. folderName .. "/Checkpoints";
      checkpointWriteTriggerMode          = "step";
      checkpointWriteStepInterval         = displayPeriod * cpInterval;
      checkpointIndexWidth                = -1;
      deleteOlderCheckpoints              = true;
      suppressNonplasticCheckpoints       = false;
      initializeFromCheckpointDir         = "";
      errorOnNotANumber                   = false;
   };
};


-- Layers --

pv.addGroup(pvParams, "Image",  {
         groupType              = "ImageLayer";
         nxScale                = 1;
         nyScale                = 1;
         nf                     = imageFeatures;
         phase                  = basePhase;
         writeStep              = -1;
         initialWriteTime       = -1;
         offsetAnchor           = "cc";
         inverseFlag            = false;
         normalizeLuminanceFlag = false;
         normalizeStdDev        = false;
         autoResizeFlag         = false;
         batchMethod            = batchmethod;
         writeFrameToTimestamp  = true;
         resetToStartOnLoop     = false;
	 displayPeriod          = displayPeriod;
         xFlipEnabled           = true;
         yFlipEnabled           = false;
         jitterChangeInterval   = 1;
         maxShiftX              = 0;
         maxShiftY              = 0;
         inputPath              = inputPath;
      }
   );

pv.addGroup(pvParams, "GanOn",  {
         groupType              = "PvpLayer";
         nxScale                = 1;
         nyScale                = 1;
         nf                     = pvpFeatures;
         phase                  = basePhase;
         writeStep              = -1;
         initialWriteTime       = -1;
         offsetAnchor           = "cc";
         inverseFlag            = false;
         normalizeLuminanceFlag = false;
         normalizeStdDev        = false;
         autoResizeFlag         = false;
         batchMethod            = batchmethod;
         writeFrameToTimestamp  = true;
         resetToStartOnLoop     = false;
	 displayPeriod          = displayPeriod;
         xFlipEnabled           = true;
         yFlipEnabled           = false;
         jitterChangeInterval   = 1;
         maxShiftX              = 0;
         maxShiftY              = 0;
         inputPath              = GanOnPath;
      }
   );

pv.addGroup(pvParams, "GanOff",  {
         groupType              = "PvpLayer";
         nxScale                = 1;
         nyScale                = 1;
         nf                     = pvpFeatures;
         phase                  = basePhase;
         writeStep              = -1;
         initialWriteTime       = -1;
         offsetAnchor           = "cc";
         inverseFlag            = false;
         normalizeLuminanceFlag = false;
         normalizeStdDev        = false;
         autoResizeFlag         = false;
         batchMethod            = batchmethod;
         writeFrameToTimestamp  = true;
         resetToStartOnLoop     = false;
	 displayPeriod          = displayPeriod;
         xFlipEnabled           = true;
         yFlipEnabled           = false;
         jitterChangeInterval   = 1;
         maxShiftX              = 0;
         maxShiftY              = 0;
         inputPath              = GanOffPath;
      }
   );

pv.addGroup(pvParams, "GanOnError", {
         groupType        = "HyPerLayer";
         nxScale          = 1;
         nyScale          = 1;
         nf               = pvpFeatures;
         phase            = basePhase+1;
         writeStep        = -1;
         initialWriteTime = -1;
      }
   );

pv.addGroup(pvParams, "GanOffError", {
         groupType        = "HyPerLayer";
         nxScale          = 1;
         nyScale          = 1;
         nf               = pvpFeatures;
         phase            = basePhase+1;
         writeStep        = -1;
         initialWriteTime = -1;
      }
   );

pv.addGroup(pvParams, "GanOnSparse", {
         groupType        = "HyPerLayer";
         nxScale          = 1;
         nyScale          = 1;
         nf               = imageFeatures;
         phase            = basePhase+1;
         writeStep        = -1;
         initialWriteTime = -1;
      }
   );

pv.addGroup(pvParams, "GanOffSparse", {
         groupType        = "HyPerLayer";
         nxScale          = 1;
         nyScale          = 1;
         nf               = imageFeatures;
         phase            = basePhase+1;
         writeStep        = -1;
         initialWriteTime = -1;
      }
   );

pv.addGroup(pvParams, "GanSparse", {
         groupType        = "HyPerLayer";
         nxScale          = 1;
         nyScale          = 1;
         nf               = imageFeatures;
         phase            = basePhase+2;
         writeStep        = -1;
         initialWriteTime = -1;
      }
   );


pv.addGroup(pvParams, "GanLCA", {
         groupType              = "HyPerLCALayer";
         nxScale                = 1 / stride;
         nyScale                = 1 / stride;
         nf                     = dictionary;
         phase                  = basePhase+2;
         InitVType              = "ConstantV";
         valueV                 = thresh * 0.9;
         triggerLayerName       = NULL;
         sparseLayer            = true;
         writeSparseValues      = true;
         updateGpu              = true;
         dataType               = nil;
         VThresh                = thresh;
         AMin                   = 0;
         AMax                   = infinity;
         AShift                 = thresh;
         VWidth                 = 0;
         timeConstantTau        = 100;
         selfInteract           = true;
         adaptiveTimeScaleProbe = "AdaptProbe";
      }
   );

pv.addGroup(pvParams, "ImageRecon",  {
         groupType        = "HyPerLayer";
         nxScale          = 1;
         nyScale          = 1;
         nf               = inputFeatures;
         phase            = basePhase+3;
         InitVType        = "ZeroV";
         triggerLayerName = "Image";
         writeStep        = -1;
         initialWriteTime = -1;
      }
   );

pv.addGroup(pvParams, "ImageError",  {
         groupType        = "HyPerLayer";
         nxScale          = 1;
         nyScale          = 1;
         nf               = inputFeatures;
         phase            = basePhase+4;
         InitVType        = "ZeroV";
         triggerLayerName = "Image";
         writeStep        = -1;
         initialWriteTime = -1;
      }
   );


-- Connections --

pv.addGroup(pvParams, "GanOnToGanOnError", {
         groupType     = "IdentConn";
         preLayerName  = "GanOn";
         postLayerName = "GanOnError";
         channelCode   = 0;
      }
   );

pv.addGroup(pvParams, "GanOffToGanOffError", {
         groupType     = "IdentConn";
         preLayerName  = "GanOff";
         postLayerName = "GanOffError";
         channelCode   = 0;
      }
   );

pv.addGroup(pvParams, "GanOnToGanOnSparse", {
         groupType               = "HyPerConn";
         preLayerName            = "GanOn";
         postLayerName           = "GanOnSparse";
         channelCode             = 0;
         plasticityFlag          = false;
         sharedWeights           = true;
         weightInitType          = "UniformRandomWeight";
         weightInit              = 1/32;
         pvpatchAccumulateType   = "convolve";
         nxp                     = 1;
         nyp                     = 1;
         normalizeMethod         = "none";
         initialWriteTime        = -1;
         writeStep               = -1;
      }
   );

pv.addGroup(pvParams, "GanOffToGanOffSparse", {
         groupType               = "HyPerConn";
         preLayerName            = "GanOff";
         postLayerName           = "GanOffSparse";
         channelCode             = 0;
         plasticityFlag          = false;
         sharedWeights           = true;
         weightInitType          = "UniformRandomWeight";
         weightInit              = 1/32;
         pvpatchAccumulateType   = "convolve";
         nxp                     = 1;
         nyp                     = 1;
         normalizeMethod         = "none";
         initialWriteTime        = -1;
         writeStep               = -1;
      }
   );

pv.addGroup(pvParams, "GanOffSparseToGanSparse", {
         groupType     = "IdentConn";
         preLayerName  = "GanOffSparse";
         postLayerName = "GanSparse";
         channelCode   = 1;
      }
   );

pv.addGroup(pvParams, "GanOnSparseToGanSparse", {
         groupType     = "IdentConn";
         preLayerName  = "GanOnSparse";
         postLayerName = "GanSparse";
         channelCode   = 0;
      }
   );


pv.addGroup(pvParams, "GanOnErrorToGanLCA", {
         groupType                     = "TransposeConn";
         preLayerName                  = "GanOnError";
         postLayerName                 = "GanLCA";
         channelCode                   = 0;
         receiveGpu                    = true;
         updateGSynFromPostPerspective = true;
         pvpatchAccumulateType         = "convolve";
         writeStep                     = -1;
         originalConnName              = "GanLCAToGanOnError";
      }
   );

pv.addGroup(pvParams, "GanOffErrorToGanLCA", {
         groupType                     = "TransposeConn";
         preLayerName                  = "GanOffError";
         postLayerName                 = "GanLCA";
         channelCode                   = 0;
         receiveGpu                    = true;
         updateGSynFromPostPerspective = true;
         pvpatchAccumulateType         = "convolve";
         writeStep                     = -1;
         originalConnName              = "GanLCAToGanOffError";
      }
   );

pv.addGroup(pvParams, "GanLCAToGanOnError", {
         groupType               = "MomentumConn";
         preLayerName            = "GanLCA";
         postLayerName           = "GanOnError";
         channelCode             = 1;
         plasticityFlag          = true;
         sharedWeights           = true;
         normalizeMethod         = "normalizeL2";
         weightInitType          = "UniformRandomWeight";
         wMinInit                = -1;
         wMaxInit                = 1;
         minNNZ                  = 1;
         sparseFraction          = 0.99;
         triggerLayerName        = "Image";
         pvpatchAccumulateType   = "convolve";
         nxp                     = patch;
         nyp                     = patch;
         strength                = 1;
         normalizeOnInitialize   = true;
         normalizeOnWeightUpdate = true;
         minL2NormTolerated      = 0;
         dWMax                   = learningRate; 
         momentumTau             = 500;
         momentumMethod          = "viscosity";
         momentumDecay           = 0;
         initialWriteTime        = -1;
         writeStep               = -1;
      }
   );

if loadWeightsFrom then
   pvParams.GanLCAToGanOnError.weightInitType = "FileWeight";
   pvParams.GanLCAToGanOnError.initWeightsFile = loadWeightsFrom .. "GanLCAToGanOnError_W.pvp";
end

pv.addGroup(pvParams, "GanLCAToGanOffError", {
         groupType               = "MomentumConn";
         preLayerName            = "GanLCA";
         postLayerName           = "GanOffError";
         channelCode             = 1;
         plasticityFlag          = true;
         sharedWeights           = true;
         normalizeMethod         = "normalizeGroup";
         normalizeGroupName      = "GanLCAToGanOffError";
         weightInitType          = "UniformRandomWeight";
         wMinInit                = -1;
         wMaxInit                = 1;
         minNNZ                  = 1;
         sparseFraction          = 0.99;
         triggerLayerName        = "Image";
         pvpatchAccumulateType   = "convolve";
         nxp                     = patch;
         nyp                     = patch;
         strength                = 1;
         normalizeOnInitialize   = true;
         normalizeOnWeightUpdate = true;
         minL2NormTolerated      = 0;
         dWMax                   = learningRate; 
         momentumTau             = 500;
         momentumMethod          = "viscosity";
         momentumDecay           = 0;
         initialWriteTime        = -1;
         writeStep               = -1;
      }
   );

if loadWeightsFrom then
   pvParams.GanLCAToGanOffError.weightInitType = "FileWeight";
   pvParams.GanLCAToGanOffError.initWeightsFile = loadWeightsFrom .. "GanLCAToGanOffError_W.pvp";
end

pv.addGroup(pvParams, "GanLCAToImageError", {
         groupType               = "HyPerConn";
         preLayerName            = "GanLCA";
         postLayerName           = "ImageError";
         channelCode             = -1;
         plasticityFlag          = true;
         sharedWeights           = true;
         normalizeMethod         = "normalizeL2";
         weightInitType          = "UniformRandomWeight";
         wMinInit                = -1;
         wMaxInit                = 1;
         minNNZ                  = 1;
         sparseFraction          = 0.95;
         triggerLayerName        = "Image";
         pvpatchAccumulateType   = "convolve";
         nxp                     = patch;
         nyp                     = patch;
         strength                = 1;
         normalizeOnInitialize   = true;
         normalizeOnWeightUpdate = true;
         minL2NormTolerated      = 0;
         dWMax                   = imagelearningRate; 
         momentumTau             = 500;
         momentumMethod          = "viscosity";
         momentumDecay           = 0;
         initialWriteTime        = -1;
         writeStep               = -1;
      }
   );

if loadWeightsFrom then
   pvParams.GanLCAToImageError.weightInitType = "FileWeight";
   pvParams.GanLCAToImageError.initWeightsFile = loadWeightsFrom .. "GanLCAToImageError_W.pvp";
end

pv.addGroup(pvParams, "GanLCAToImageRecon", {
         groupType             = "CloneConn";
         preLayerName          = "GanLCA";
         postLayerName         = "ImageRecon";
         channelCode           = 0;
         pvpatchAccumulateType = "convolve";
         originalConnName      = "GanLCAToImageError";
      }
   );

pv.addGroup(pvParams, "ImageReconToImageError", {
         groupType     = "IdentConn";
         preLayerName  = "ImageRecon";
         postLayerName = "ImageError";
         channelCode   = 1;
      }
   );

pv.addGroup(pvParams, "ImageToImageError", {
         groupType     = "IdentConn";
         preLayerName  = "Image";
         postLayerName = "ImageError";
         channelCode   = 0;
      }
   );


-- Probes --

pv.addGroup(pvParams, "AdaptProbe", {
         groupType        = "KneeTimeScaleProbe";
         targetName       = "EnergyProbe";
         message          = NULL;
         textOutputFlag   = true;
         probeOutputFile  = "AdaptiveTimeScales.txt";
         triggerLayerName = "Image";
         triggerOffset    = 0;
         baseMax          = 0.022;
         baseMin          = 0.02;
         tauFactor        = 0.0125;
         growthFactor     = 0.025;
         writeTimeScales  = true;
         kneeThresh       = 1.5;
         kneeSlope        = 0.025;
      }
   );

pv.addGroup(pvParams, "EnergyProbe", {
         groupType        = "ColumnEnergyProbe";
         message          = nil;
         textOutputFlag   = true;
         probeOutputFile  = "EnergyProbe.txt";
         triggerLayerName = nil;
         energyProbe      = nil;
      }
   );

pv.addGroup(pvParams, "ImageErrorL2Probe", {
         groupType       = "L2NormProbe";
         targetLayer     = "ImageError";
         message         = nil;
         textOutputFlag  = true;
         probeOutputFile = "ImageErrorL2.txt";
         energyProbe     = "EnergyProbe";
         coefficient     = 0.5;
         maskLayerName   = nil;
         exponent        = 2;
      }
   );

pv.addGroup(pvParams, "GanLCAL1Probe", {
         groupType       = "L1NormProbe";
         targetLayer     = "GanLCA";
         message         = nil;
         textOutputFlag  = true;
         probeOutputFile = "GanLCAL1Probe.txt";
         energyProbe     = "EnergyProbe";
         coefficient     = pvParams["GanLCA"].VThresh;
         maskLayerName   = nil;
      }
   );

pv.addGroup(pvParams, "GanLCAL0Probe", {
         groupType       = "L0NormProbe";
         targetLayer     = "GanLCA";
         message         = nil;
         textOutputFlag  = true;
         probeOutputFile = "GanLCAL0Probe.txt";
         energyProbe     = nil;
         coefficient     = 1.0;
         maskLayerName   = nil;
      }
   );

-- Build and run the params file --

os.execute("mkdir -p " .. "output/" .. folderName);
local file = io.open("output/" .. folderName .. "/Gan.params", "w");
io.output(file);
pv.printConsole(pvParams);
io.close(file);
-- The & makes it run without blocking execution
os.execute(pvDir .. "/python/draw -p -a " .. "output/" .. folderName .. "/Gan.params &");
--os.execute("mpiexec -np " .. batchWidth * rows * cols .. " " .. pvDir .. "/build/tests/BasicSystemTest/Release/BasicSystemTest -p " .. "output/" .. folderName .. "/Gan.params -t " .. threads .. " -batchwidth " .. batchWidth .. " -rows " .. rows .. " -columns " .. cols);
-- Make sure we close the draw tool
os.execute("killall draw -KILL"); 
