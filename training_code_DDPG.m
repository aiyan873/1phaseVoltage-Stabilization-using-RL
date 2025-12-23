% ==================================================================
% TRAIN_RESO_CONTROLLER_DDPG_CPU.M -- DDPG for Reso_controller
% ==================================================================
% Optimized hyperparameters based on grid search results
% ==================================================================

model = "Reso_controller";
agentBlock = "Reso_controller/Output voltage/RL Agent";

% ==================================================================
% OBSERVATION AND ACTION SPECIFICATIONS
% ==================================================================
obsInfo = rlNumericSpec([3 1],...
    'LowerLimit', [-inf; -inf; 0],...
    'UpperLimit', [inf; inf; inf]);
obsInfo.Name = "observations";
obsInfo.Description = "three observations (state, error, integrated error)";

actInfo = rlNumericSpec([1 1], 'LowerLimit', -1, 'UpperLimit', 1);
actInfo.Name = "action";
actInfo.Description = "single continuous action (controller output)";

% ==================================================================
% CREATE ENVIRONMENT
% ==================================================================
env = rlSimulinkEnv(model, agentBlock, obsInfo, actInfo);
env.ResetFcn = @(in) localResetFcn(in);

% ==================================================================
% TIMING PARAMETERS
% ==================================================================
Ts = 0.1;   % Sample time (control frequency)
Tf = 30;    % Episode duration in seconds
maxSteps = ceil(Tf / Ts);  % 300 steps per episode

% ==================================================================
% CRITIC NETWORK - 3 Hidden Layers [128, 128, 64]
% ==================================================================
obsPath = [
    featureInputLayer(obsInfo.Dimension(1), Name="obsInLyr")
];

actPath = [
    featureInputLayer(actInfo.Dimension(1), Name="actInLyr")
];

commonPath = [
    concatenationLayer(1, 2, Name="concat")
    fullyConnectedLayer(128, Name="criticFC1")
    reluLayer(Name="criticRelu1")
    fullyConnectedLayer(128, Name="criticFC2")
    reluLayer(Name="criticRelu2")
    fullyConnectedLayer(64, Name="criticFC3")
    reluLayer(Name="criticRelu3")
    fullyConnectedLayer(1, Name="QValue")
];

criticLG = layerGraph();
criticLG = addLayers(criticLG, obsPath);
criticLG = addLayers(criticLG, actPath);
criticLG = addLayers(criticLG, commonPath);
criticLG = connectLayers(criticLG, "obsInLyr", "concat/in1");
criticLG = connectLayers(criticLG, "actInLyr", "concat/in2");

rng(0, "twister");
criticNet = dlnetwork(criticLG);

critic = rlQValueFunction(criticNet, obsInfo, actInfo, ...
    'ObservationInputNames', "obsInLyr", ...
    'ActionInputNames', "actInLyr");

% ==================================================================
% ACTOR NETWORK - 2 Hidden Layers [128, 64]
% ==================================================================
actorLayers = [
    featureInputLayer(obsInfo.Dimension(1), Name="actorObs")
    fullyConnectedLayer(128, Name="actorFC1")
    reluLayer(Name="actorRelu1")
    fullyConnectedLayer(64, Name="actorFC2")
    reluLayer(Name="actorRelu2")
    fullyConnectedLayer(actInfo.Dimension(1), Name="actorOut")
    tanhLayer(Name="actorTanh")
];

rng(0, "twister");
actorNet = dlnetwork(actorLayers);

actor = rlContinuousDeterministicActor(actorNet, obsInfo, actInfo);

% ==================================================================
% CREATE DDPG AGENT
% ==================================================================
agent = rlDDPGAgent(actor, critic);

% Sample time
agent.AgentOptions.SampleTime = Ts;

% Discount factor
agent.AgentOptions.DiscountFactor = 0.995;

% Mini-batch size
agent.AgentOptions.MiniBatchSize = 512;

% Experience buffer
agent.AgentOptions.ExperienceBufferLength = 1e6;

% Target network update
agent.AgentOptions.TargetSmoothFactor = 1e-3;

% N-step returns
agent.AgentOptions.NumStepsToLookAhead = 2;

% ==================================================================
% OPTIMIZER OPTIONS
% ==================================================================
% Actor optimizer
actorOpts = rlOptimizerOptions(...
    'LearnRate', 1e-4, ...
    'GradientThreshold', 1, ...
    'L2RegularizationFactor', 1e-3);

% Critic optimizer
criticOpts = rlOptimizerOptions(...
    'LearnRate', 1e-4, ...
    'GradientThreshold', 1, ...
    'L2RegularizationFactor', 1e-4);

agent.AgentOptions.ActorOptimizerOptions = actorOpts;
agent.AgentOptions.CriticOptimizerOptions = criticOpts;

% ==================================================================
% EXPLORATION NOISE
% ==================================================================
agent.AgentOptions.NoiseOptions.StandardDeviation = 0.2;
agent.AgentOptions.NoiseOptions.StandardDeviationDecayRate = 1e-4;

% ==================================================================
% TRAINING OPTIONS
% ==================================================================
trainOpts = rlTrainingOptions(...
    'MaxEpisodes', 1000, ...
    'MaxStepsPerEpisode', maxSteps, ...
    'Verbose', true, ...
    'Plots', "training-progress", ...
    'StopTrainingCriteria', "AverageReward", ...
    'StopTrainingValue', 0, ...
    'UseParallel', false);

% Evaluator
evl = rlEvaluator('EvaluationFrequency', 10, 'NumEpisodes', 5);

% ==================================================================
% TRAINING EXECUTION
% ==================================================================
doTraining = true;

if doTraining
    fprintf("Starting DDPG training (CPU only) on model: %s\n", model);
    rng(0, "twister");
    trainingStats = train(agent, env, trainOpts, 'Evaluator', evl);
    save("Reso_controller_ddpg_cpu_trained.mat", "agent", "trainingStats");
else
    load("Reso_controller_ddpg_cpu_trained.mat", "agent");
end

% ==================================================================
% POST-TRAINING SIMULATION
% ==================================================================
simOpts = rlSimulationOptions('MaxSteps', maxSteps, 'StopOnError', "on");
experiences = sim(env, agent, simOpts);
fprintf("Sim completed. Total reward: %.3f\n", sum(experiences.Reward.Data));

% ==================================================================
% RESET FUNCTION
% ==================================================================
function in = localResetFcn(in)
    % Reset conditions between episodes
    
    persistent episodeCount;
    if isempty(episodeCount)
        episodeCount = 0;
    end
    episodeCount = episodeCount + 1;
    
    % Randomize reference voltage
    try
        blk = "Reso_controller/Reference";
        if get_param(blk, "Handle")
            % Curriculum learning factor
            curriculum_factor = min(1.0, episodeCount / 200);
            
            % Target: 300V with expanding variance
            v = 220 + 30 * curriculum_factor * randn();
            v = max(min(v, 350), 150);
            
            in = setBlockParameter(in, blk, 'Value', num2str(v));
        end
    catch
        % Ignore if block doesn't exist
    end
    
    % Reset integrator initial condition
    try
        blkI = "Reso_controller/generate_observations/Integrator";
        if ~isempty(find_system(bdroot, 'BlockType', 'Integrator'))
            % Curriculum learning factor
            curriculum_factor = min(1.0, episodeCount / 200);
            
            val = 10 * curriculum_factor * randn();
            val = max(min(val, 50), -50);
            
            in = setBlockParameter(in, blkI, 'InitialCondition', num2str(val));
        end
    catch
        % Ignore if not found
    end
end
