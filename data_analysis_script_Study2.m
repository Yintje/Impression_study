Mydata=readtable("study2_full_data2.csv");

T1={'6679', '2600', '4858', '1569', '5114', '4255', '4907', '1934', '7836', '5706', ...
         '4537', '5503', '8184', '4050', '8322', '5477', '1498', '5769', '2919', '1960', ...
         '1201', '8398', '3715', '2885', '6184', '8105', '2788', '8609', '1909', '8388', ...
         '6466', '3289', '6992', '5250', '2789', '5518', '4781', '1079', '9573', '5557', ...
         '2348', '4515', '6033', '5436', '3908', '6955', '1709', '6636', '4267', '7255'};

T2={'2757', '4021', 'lettie', '1648', '6790', '1142', '1690', '3464', '7500', '4462', ...
         '5066', '2979', '5262', '5323', '6878', '9738', '8861', '4017', '8259', '7963', ...
         '9940', '7023', '3951', '5251', '6212', '9111', '5322', '9619', '6796', '8125', ...
         '6919', '1769', '3326', '7387', '3380', '8673', '1660', '4854', '6694', '9911', ...
         '5846', '3994', '3324', '8206', '2170', '5982', '5744', '6584', '9870', '9309'};

Power_scores=nonzeros(Mydata.Sum_power);
gender=Mydata.Gender_multipleChoice_index;
count_female=sum(gender==2);
count_male=sum(gender==1);
age=Mydata.Age_open;
age=age(~isnan(age));
mean_age=mean(age);

mean_of_group_power=mean(Power_scores);%19.1798
% Get unique participant indices (aliases)

Mydata.alias=string(Mydata.alias);
participants =unique(Mydata.alias, 'stable');

n_participant=numel(participants);
n_model=2;
% create empty holder for model evidence (for model comparison later)
logEvidence_social=NaN(n_model,n_participant); %each participant has 4 models
logEvidence_nonsocial=NaN(n_model,n_participant);
logEvidence_getting_better=NaN(n_model,n_participant);
logEvidence_getting_worse=NaN(n_model,n_participant);

logEvidence_highpower_social=NaN(n_model,n_participant);
logEvidence_highpower_nonsocial=NaN(n_model,n_participant);
logEvidence_lowpower_social=NaN(n_model,n_participant);
logEvidence_lowpower_nonsocial=NaN(n_model,n_participant);

end_point_social_getting_better=NaN(1,n_participant); %one middle point per condition
end_point_social_getting_worse=NaN(1,n_participant);
end_point_nonsocial_getting_better=NaN(1,n_participant);
end_point_nonsocial_getting_worse=NaN(1,n_participant);

general_frame_social_getting_better=NaN(n_participant,6); %for extracting the time series for each participant for social
general_frame_social_getting_worse=NaN(n_participant,6); 
general_frame_nonsocial_getting_better=NaN(n_participant,6);%for extracting the time series for each participant for nonsocial
general_frame_nonsocial_getting_worse=NaN(n_participant,6);

%range_social=NaN(n_participant,6);
%range_nonsocial=NaN(n_participant,6);

n_group1= 50;
n_group2= 50;
social_BA=NaN(1,n_participant);
nonsocial_BA=NaN(1,n_participant);
social_getting_better=NaN(1,n_participant);
social_getting_worse=NaN(1,n_participant);
nonsocial_getting_better=NaN(1,n_participant);
nonsocial_getting_worse=NaN(1,n_participant);



% % Create an empty struct holder for model1 VS model2, and social VS non
% social

posterior_social = repmat(struct('muTheta',NaN),n_model,n_participant);
posterior_nonsocial = repmat(struct('muTheta',NaN), n_model,n_participant);


posterior_getting_better = repmat(struct('muTheta',NaN), n_model,n_participant);
posterior_getting_worse = repmat(struct('muTheta',NaN), n_model,n_participant);

predicted_social_getting_better_generic_model = NaN(n_participant,6);
predicted_social_getting_worse_generic_model = NaN(n_participant,6);
predicted_nonsocial_getting_better_generic_model = NaN(n_participant,6);
predicted_nonsocial_getting_worse_generic_model = NaN(n_participant,6);

predicted_social_getting_better_base = NaN(n_participant,6);
predicted_social_getting_worse_base = NaN(n_participant,6);
predicted_nonsocial_getting_better_base = NaN(n_participant,6);
predicted_nonsocial_getting_worse_base = NaN(n_participant,6);

predicted_social_getting_better_contr = NaN(n_participant,6);
predicted_social_getting_worse_contr = NaN(n_participant,6);
predicted_nonsocial_getting_better_contr = NaN(n_participant,6);
predicted_nonsocial_getting_worse_contr = NaN(n_participant,6);

power_above_average=[];
power_below_average=[];

for i = 1:length(participants)
    % Extract data for the current participant
    participant_data = Mydata(strcmp(Mydata.alias, participants{i}), :);
    
    % %assign it to the power groups
    % if Power_scores(i)>mean_of_group_power
    %     power_above_average=[power_above_average;participant_data];
    % else
    %     power_below_average=[power_below_average;participant_data];
    % end
    % 
    % inputs
    N_increase = [0;0;1;2;3;3];%sequence of social good stimulus in condition 1
    N_decrease = [3;3;2;1;0;0]; %sequence of social bad stimulus in condition 1

    decision_social=participant_data.decision_employee;
    decision_nonsocial=participant_data.decision_house;
    
    %find the condition of the participant
    alias=string(unique(participant_data.alias));
    % Check if the alias exists in either array
    if ismember(alias, T1)
        condition = "T1";
    elseif ismember(alias, T2)
        condition = "T2";
    else
        disp([alias, ' does not exist in either array']);
    % Optionally handle case where alias is not found
    end
    days=numel(decision_social);
    t=1:days;
        
    if condition== "T1"
        u_social_getting_better=[N_increase(1:days),N_decrease(1:days),t']';
        u_nonsocial_getting_worse=[N_decrease(1:days),N_increase(1:days),t']';
    else
        u_nonsocial_getting_better=[N_increase(1:days),N_decrease(1:days),t']';
        u_social_getting_worse=[N_decrease(1:days),N_increase(1:days),t']';
    end



    % Extract the first two columns and assign them to variable 
    
    social = [participant_data.LowerBoundImpressionEmployee,participant_data.upperBoundImpressionEmployee];
    nonsocial= [participant_data.lowerBoundImpressionHouse,participant_data.upperBoundImpressionHouse];
    
    %calculate the means
    sum_social=social(:,1)+social(:,2);
    mean_social=sum_social/2;
    %range_social(i,:)=abs(social.upperBoundImpressionEmployee-social.LowerBoundImpressionEmployee);
    

    sum_nonsocial=nonsocial(:,1)+nonsocial(:,2);
    mean_nonsocial=sum_nonsocial/2;
    %range_nonsocial(i,:)=abs(nonsocial.upperBoundImpressionHouse-nonsocial.lowerBoundImpressionHouse);
    
    if condition== "T1"
        if days<6
            end_point_social_getting_worse(i)=NaN;
        else
            end_point_social_getting_better(i)=mean_social(6);
            end_point_nonsocial_getting_worse(i)=mean_nonsocial(6);
        end
    elseif condition== "T2"
        if days<6
            end_point_social_getting_worse(i)=NaN;
        else
            end_point_social_getting_worse(i)=mean_social(6);
            end_point_nonsocial_getting_better(i)=mean_nonsocial(6);
        end
    else
         disp('Found a weird participant!')
    end
    

    %dif=A(:,2)-A(:,1);
    % observations
    mean_social = mean_social';
    mean_nonsocial = mean_nonsocial';
    if condition=="T1"
        general_frame_social_getting_better(i,1:days)=mean_social;
        general_frame_nonsocial_getting_worse(i,1:days)=mean_nonsocial;
    else
        general_frame_social_getting_worse(i,1:days)=mean_social;
        general_frame_nonsocial_getting_better(i,1:days)=mean_nonsocial;
    end

    

    % specify model
    % =========================================================================
    
    f_fname1 = @Evolution_function_1parameters_sig; % evolution function (by default)
    
    f_fname2 = @Evolution_function_rationalBayesian; % baseline
    
    f_fname3= @Evolution_function_1parameters_sigcontra; %contrast hypothesis, only update when there is a difference
    
    f_fname4= @Evolution_function_recency; 
    %f_fname4= @Evolution_function_1parameters_hyst3;
    %f_fname5= @Evolution_function_1parameters_hyst4;
    %f_fname6= @Evolution_function_1parameters_hyst5;
   
    
   
    g_fname = @Observation_function_added_bias; % observation function (softmax mapping)
    
    %put all evolution functions into one cell array
    f_fnames = { f_fname1, f_fname2,f_fname3, f_fname4};
    
    % provide dimensions
    dim = struct( ...
        'n', 2, ... number of hidden states (good and bad subjective'count')
        'p',1, ...
        'n_theta', 1, ... number of evolution parameters (1: alpha value)
        'n_phi', 1 ... number of observation parameters (none)
       );
        
    
    %continuous y
    %options.sources.type = 0;
    
    % options for the simulation
    % -------------------------------------------------------------------------
    % set options
    options_s.priors.muX0 = [0;0];
    options_s.priors.SigmaX0 = 100*eye(2);
    options_s.priors.muTheta = 0;
    options_s.priors.SigmaTheta = 10;
    options_s.priors.b_sigma = var(mean_social);
    optione_s.prior.muPhi=0;
    optione_s.prior.SigmaPhi=100;
    options_s.DisplayWin=0;%do not show plots
    
    
    if days<6
        options.isYout=[zeros(1,days),ones(1,6-days)];
    else
        options.isYout=[zeros(1,6)];
    end


    options_n.priors.muX0 = [0; 0];
    options_n.priors.SigmaX0 = 100*eye(2);
    options_n.priors.muTheta = 0;
    options_n.priors.SigmaTheta = 10;
    options_n.priors.b_sigma = var(mean_nonsocial);
    optione_n.prior.muPhi=0;
    optione_n.prior.SigmaPhi=100;
    options_n.DisplayWin=0;%do not show plots
   
    
    
    % number of trials
    n_t = days; 
    t=1;
    
    options_s.dim = dim;
    
    % Initialize cell arrays to store the results
    posteriors_s = cell(1, numel(f_fnames));
    outs_s = cell(1, numel(f_fnames));
    posteriors_n = cell(1, numel(f_fnames));
    outs_n = cell(1, numel(f_fnames));
    
    % invert model
    
    if condition== "T1"
    % social part
      for m = 1:numel(f_fnames)
        % Social
        [posteriors_s{m}, outs_s{m}] = VBA_NLStateSpaceModel(mean_social, u_social_getting_better, f_fnames{m}, g_fname, dim, options_s);
        % Nonsocial
        [posteriors_n{m}, outs_n{m}] = VBA_NLStateSpaceModel(mean_nonsocial, u_nonsocial_getting_worse, f_fnames{m}, g_fname, dim, options_n);
        
        % Populate posterior structures for social
        posterior_social(m,i).muTheta = posteriors_s{m}.muTheta;       
        % Populate posterior structures for nonsocial
        posterior_nonsocial(m,i).muTheta = posteriors_n{m}.muTheta;
        
        predicted_muXs=exp(posteriors_s{m}.muX);
        predicted_impression_s=100*(predicted_muXs(1, :) ./ (predicted_muXs(1, :) + predicted_muXs(2, :)));
        if m==1
            predicted_social_getting_better_generic_model(i,1:days)=predicted_impression_s;
        elseif m==2
            predicted_social_getting_better_base(i,1:days)=predicted_impression_s;
        else
            predicted_social_getting_better_contr(i,1:days)=predicted_impression_s;
        end
        
        predicted_muXn=exp(posteriors_n{m}.muX);
        predicted_impression_n=100*(predicted_muXn(1, :) ./ (predicted_muXn(1, :) + predicted_muXn(2, :)));
        if m==1
            predicted_nonsocial_getting_worse_generic_model(i,1:days)=predicted_impression_n;
        elseif m==2
            predicted_nonsocial_getting_worse_base(i,1:days)=predicted_impression_n;
        else
            predicted_nonsocial_getting_worse_contr(i,1:days)=predicted_impression_n;
        end
        
        % Collect model evidence for model comparison
        logEvidence_social(m,i) = outs_s{m}.F;
        logEvidence_nonsocial(m,i) = outs_n{m}.F;
        
        % record it in better VS worse label
        posterior_getting_better(m,i).muTheta=posteriors_s{m}.muTheta;
        posterior_getting_worse(m,i).muTheta=posteriors_n{m}.muTheta;
        logEvidence_getting_better(m,i)= logEvidence_social(m,i);
        logEvidence_getting_worse(m,i)= logEvidence_nonsocial(m,i);
      end
      
      
      
    elseif condition == "T2"
        for m = 1:numel(f_fnames)
        % Social
            [posteriors_s{m}, outs_s{m}] = VBA_NLStateSpaceModel(mean_social, u_social_getting_worse, f_fnames{m}, g_fname, dim, options_s);
        % Nonsocial
            [posteriors_n{m}, outs_n{m}] = VBA_NLStateSpaceModel(mean_nonsocial, u_nonsocial_getting_better, f_fnames{m}, g_fname, dim, options_n);
            
        % Populate posterior structures for social
        posterior_social(m,i).muTheta = posteriors_s{m}.muTheta;
        % Populate posterior structures for nonsocial
        posterior_nonsocial(m,i).muTheta = posteriors_n{m}.muTheta;
        
        predicted_muXs=exp(posteriors_s{m}.muX);
        predicted_impression_s=100*(predicted_muXs(1, :) ./ (predicted_muXs(1, :) + predicted_muXs(2, :)));
        if m==1
            predicted_social_getting_worse_generic_model(i,1:days)=predicted_impression_s;
        elseif m==2
            predicted_social_getting_worse_base(i,1:days)=predicted_impression_s;
        else
            predicted_social_getting_worse_contr(i,1:days)=predicted_impression_s;
        end
        
        predicted_muXn=exp(posteriors_n{m}.muX);
        predicted_impression_n=100*(predicted_muXn(1, :) ./ (predicted_muXn(1, :) + predicted_muXn(2, :)));
        if m==1
            predicted_nonsocial_getting_better_generic_model(i,1:days)=predicted_impression_n;
        elseif m==2
            predicted_nonsocial_getting_better_base(i,1:days)=predicted_impression_n;
        else
            predicted_nonsocial_getting_better_contr(i,1:days)=predicted_impression_n;
        end
        % Collect model evidence for model comparison
        logEvidence_social(m,i) = outs_s{m}.F;
        logEvidence_nonsocial(m,i) = outs_n{m}.F;
         % record it in better VS worse label
        posterior_getting_better(m,i).muTheta=posteriors_n{m}.muTheta;
        posterior_getting_worse(m,i).muTheta=posteriors_s{m}.muTheta;
        logEvidence_getting_worse(m,i)= logEvidence_social(m,i);
        logEvidence_getting_better(m,i)= logEvidence_nonsocial(m,i);
        end
        
    else
        disp('Found a weird participant!')
    end

    
    
    
    %assign the value for negative bias
    if condition== "T1"
        social_getting_better(i)=posterior_social(3,i).muTheta;
        nonsocial_getting_worse(i)=posterior_nonsocial(3,i).muTheta;
        
        
    else
        social_getting_worse(i)=posterior_social(3,i).muTheta;
        nonsocial_getting_better(i)=posterior_nonsocial(3,i).muTheta;
        
    end
    
    %collect model evidence for model comparison
    % first the social VS non-social model evidence


   




end
%% prepare the average trend of raw data
general_frame_social_getting_better_cleaned = general_frame_social_getting_better(all(~isnan(general_frame_social_getting_better), 2), :);
general_frame_social_getting_worse_cleaned = general_frame_social_getting_worse(all(~isnan(general_frame_social_getting_worse), 2), :);
general_frame_nonsocial_getting_better_cleaned = general_frame_nonsocial_getting_better(all(~isnan(general_frame_nonsocial_getting_better), 2), :);
general_frame_nonsocial_getting_worse_cleaned = general_frame_nonsocial_getting_worse(all(~isnan(general_frame_nonsocial_getting_worse), 2), :);
general_average_social_better=mean(general_frame_social_getting_better_cleaned);
general_average_social_worse=mean(general_frame_social_getting_worse_cleaned);
general_average_nonsocial_better=mean(general_frame_nonsocial_getting_better_cleaned);
general_average_nonsocial_worse=mean(general_frame_nonsocial_getting_worse_cleaned);

%prepare the predicted values
predicted_social_getting_better_generic_model_cleaned = predicted_social_getting_better_generic_model(all(~isnan(predicted_social_getting_better_generic_model), 2), :);
predicted_social_getting_worse_generic_model_cleaned = predicted_social_getting_worse_generic_model(all(~isnan(predicted_social_getting_worse_generic_model), 2), :);
predicted_nonsocial_getting_better_generic_model_cleaned = predicted_nonsocial_getting_better_generic_model(all(~isnan(predicted_nonsocial_getting_better_generic_model), 2), :);
predicted_nonsocial_getting_worse_generic_model_cleaned = predicted_nonsocial_getting_worse_generic_model(all(~isnan(predicted_nonsocial_getting_worse_generic_model), 2), :);
predicted_average_social_better=mean(predicted_social_getting_better_generic_model_cleaned);
predicted_average_social_worse=mean(predicted_social_getting_worse_generic_model_cleaned);
predicted_average_nonsocial_better=mean(predicted_nonsocial_getting_better_generic_model_cleaned);
predicted_average_nonsocial_worse=mean(predicted_nonsocial_getting_worse_generic_model_cleaned);

%prepare the contrast model predictions
predicted_social_getting_better_contr_cleaned = predicted_social_getting_better_contr(all(~isnan(predicted_social_getting_better_contr), 2), :);
predicted_social_getting_worse_contr_cleaned = predicted_social_getting_worse_contr(all(~isnan(predicted_social_getting_worse_contr), 2), :);
predicted_nonsocial_getting_better_contr_cleaned = predicted_nonsocial_getting_better_contr(all(~isnan(predicted_nonsocial_getting_better_contr), 2), :);
predicted_nonsocial_getting_worse_contr_cleaned = predicted_nonsocial_getting_worse_contr(all(~isnan(predicted_nonsocial_getting_worse_contr), 2), :);
predicted_average_social_better_contr=mean(predicted_social_getting_better_contr_cleaned);
predicted_average_social_worse_contr=mean(predicted_social_getting_worse_contr_cleaned);
predicted_average_nonsocial_better_contr=mean(predicted_nonsocial_getting_better_contr_cleaned);
predicted_average_nonsocial_worse_contr=mean(predicted_nonsocial_getting_worse_contr_cleaned);


%prepare the baseline model predictions
predicted_social_getting_better_base_cleaned = predicted_social_getting_better_base(all(~isnan(predicted_social_getting_better_base), 2), :);
predicted_social_getting_worse_base_cleaned = predicted_social_getting_worse_base(all(~isnan(predicted_social_getting_worse_base), 2), :);
predicted_nonsocial_getting_better_base_cleaned = predicted_nonsocial_getting_better_base(all(~isnan(predicted_nonsocial_getting_better_base), 2), :);
predicted_nonsocial_getting_worse_base_cleaned = predicted_nonsocial_getting_worse_base(all(~isnan(predicted_nonsocial_getting_worse_base), 2), :);
predicted_average_social_better_base=mean(predicted_social_getting_better_base_cleaned);
predicted_average_social_worse_base=mean(predicted_social_getting_worse_base_cleaned);
predicted_average_nonsocial_better_base=mean(predicted_nonsocial_getting_better_base_cleaned);
predicted_average_nonsocial_worse_base=mean(predicted_nonsocial_getting_worse_base_cleaned);
% plottng the trend against time
% Create the plot
figure; % Open a new figure window

% Time points
time = 1:6;

% Midpoint for reference
midpoint = 50;

% Panel 1: Visualization of Averaged Data
subplot(2, 2, 1); % Create the first subplot (1 row, 4 columns, first panel)
plot(time, general_average_social_better, '-r', 'DisplayName', 'Social start negative'); % Red line
hold on;
plot(time, general_average_social_worse, '--r', 'DisplayName', 'Social start positive'); % Red dashed line
plot(time, general_average_nonsocial_better, '-b', 'DisplayName', 'Nonsocial start negative'); % Blue line
plot(time, general_average_nonsocial_worse, '--b', 'DisplayName', 'Nonsocial start positive'); % Blue dashed line

% Compute overall midline
overall_midline = mean([general_average_social_better; general_average_social_worse; ...
                        general_average_nonsocial_better; general_average_nonsocial_worse], 1);

% Overlay the overall midline
plot(time, overall_midline, '-k', 'LineWidth', 2, 'DisplayName', 'Overall midline'); % Black solid line

% Add horizontal reference line
yline(midpoint, '--k', 'DisplayName', 'Midpoint Reference'); % Red dashed line at midpoint

% Adjust axis and labels
axis([1 6 0 100]);
xlabel('Time');
ylabel('Impression');
title('A');
xticks(1:1:6);
legend('show','Location', 'northeast');

% Panel 2: Visualization of Predicted Values (contrast model)
subplot(2, 2, 2); % Create the second subplot (1 row, 4 columns, second panel)
plot(time, predicted_average_social_better_contr, '-r', 'DisplayName', 'Social start negative'); % Red line
hold on;
plot(time, predicted_average_social_worse_contr, '--r', 'DisplayName', 'Social start positive'); % Red dashed line
plot(time, predicted_average_nonsocial_better_contr, '-b', 'DisplayName', 'Nonsocial start negative'); % Blue line
plot(time, predicted_average_nonsocial_worse_contr, '--b', 'DisplayName', 'Nonsocial start positive'); % Blue dashed line

% Compute overall midline
overall_midline_contr = mean([predicted_average_social_better_contr; predicted_average_social_worse_contr; ...
                               predicted_average_nonsocial_better_contr; predicted_average_nonsocial_worse_contr], 1);

% Overlay the overall midline
plot(time, overall_midline_contr, '-k', 'LineWidth', 2, 'DisplayName', 'Overall midline');

% Add horizontal reference line
yline(midpoint, '--k', 'DisplayName', 'Midpoint Reference');

% Adjust axis and labels
axis([1 6 0 100]);
xlabel('Time');
ylabel('Impression');
title('B');
xticks(1:1:6);
legend('show','Location', 'northeast');

% Panel 3: Visualization of Predicted Values (generic model)
subplot(2, 2, 3); % Create the third subplot (1 row, 4 columns, third panel)
plot(time, predicted_average_social_better, '-r', 'DisplayName', 'Social start negative'); % Red line
hold on;
plot(time, predicted_average_social_worse, '--r', 'DisplayName', 'Social start positive'); % Red dashed line
plot(time, predicted_average_nonsocial_better, '-b', 'DisplayName', 'Nonsocial start negative'); % Blue line
plot(time, predicted_average_nonsocial_worse, '--b', 'DisplayName', 'Nonsocial start positive'); % Blue dashed line

% Compute overall midline
overall_midline_generic = mean([predicted_average_social_better; predicted_average_social_worse; ...
                                 predicted_average_nonsocial_better; predicted_average_nonsocial_worse], 1);

% Overlay the overall midline
plot(time, overall_midline_generic, '-k', 'LineWidth', 2, 'DisplayName', 'Overall midline');

% Add horizontal reference line
yline(midpoint, '--k', 'DisplayName', 'Midpoint Reference');

% Adjust axis and labels
axis([1 6 0 100]);
xlabel('Time');
ylabel('Impression');
title('C');
xticks(1:1:6);
legend('show','Location', 'northeast');

% Panel 4: Visualization of Predicted Values (baseline model)
subplot(2, 2, 4); % Create the fourth subplot (1 row, 4 columns, fourth panel)
plot(time, predicted_average_social_better_base, '-r', 'DisplayName', 'Social start negative'); % Red line
hold on;
plot(time, predicted_average_social_worse_base, '--r', 'DisplayName', 'Social start positive'); % Red dashed line
plot(time, predicted_average_nonsocial_better_base, '-b', 'DisplayName', 'Nonsocial start negative'); % Blue line
plot(time, predicted_average_nonsocial_worse_base, '--b', 'DisplayName', 'Nonsocial start positive'); % Blue dashed line

% Compute overall midline
overall_midline_baseline = mean([predicted_average_social_better_base; predicted_average_social_worse_base; ...
                                  predicted_average_nonsocial_better_base; predicted_average_nonsocial_worse_base], 1);

% Overlay the overall midline
plot(time, overall_midline_baseline, '-k', 'LineWidth', 2, 'DisplayName', 'Overall midline');

% Add horizontal reference line
yline(midpoint, '--', 'DisplayName', 'Midpoint Reference','Color', 'k');

% Adjust axis and labels
axis([1 6 0 100]);
xlabel('Time');
ylabel('Impression');
title('D');
xticks(1:1:6);
legend('show','Location', 'northeast');
%%
% Create the plot
figure; % Open a new figure window

% Midpoint for flipping
midpoint = 50;

% Panel 1: Visualization of Averaged Data
subplot(1, 3, 1); % Create the first subplot (1 row, 3 columns, first panel)
time = 1:6;

% Plot original lines (all green, distinguish social/nonsocial with shape and better/worse with line style)
plot(time, general_average_social_better, '-og', 'DisplayName', 'Social start negative (original)'); % Green solid with circles
hold on;
plot(time, general_average_social_worse, '--og', 'DisplayName', 'Social start positive (original)'); % Green dashed with circles
plot(time, general_average_nonsocial_better, '-sg', 'DisplayName', 'Nonsocial start negative (original)'); % Green solid with squares
plot(time, general_average_nonsocial_worse, '--sg', 'DisplayName', 'Nonsocial start positive (original)'); % Green dashed with squares

% Calculate flipped lines
flipped_social_better = midpoint + (midpoint - general_average_social_better);
flipped_social_worse = midpoint + (midpoint - general_average_social_worse);
flipped_nonsocial_better = midpoint + (midpoint - general_average_nonsocial_better);
flipped_nonsocial_worse = midpoint + (midpoint - general_average_nonsocial_worse);

% Plot flipped lines (all orange, distinguish social/nonsocial with shape and better/worse with line style)
plot(time, flipped_social_better, '-o', 'Color', [1 0.5 0], 'DisplayName', 'Social start negative (flipped)'); % Orange solid with circles
plot(time, flipped_social_worse, '--o', 'Color', [1 0.5 0], 'DisplayName', 'Social start positive (flipped)'); % Orange dashed with circles
plot(time, flipped_nonsocial_better, '-s', 'Color', [1 0.5 0], 'DisplayName', 'Nonsocial start negative (flipped)'); % Orange solid with squares
plot(time, flipped_nonsocial_worse, '--s', 'Color', [1 0.5 0], 'DisplayName', 'Nonsocial start positive (flipped)'); % Orange dashed with squares

% Adjust axis and labels
axis([1 6 0 100]);
xlabel('Time');
ylabel('Impression');
title('Visualization of Averaged Data');
xticks(1:1:6);
legend('show');

% Panel 2: Visualization of Predicted Values (generic model)
subplot(1, 3, 2); % Create the second subplot (1 row, 3 columns, second panel)

% Plot original lines (all green, distinguish social/nonsocial with shape and better/worse with line style)
plot(time, predicted_average_social_better, '-og', 'DisplayName', 'Social start negative (original)');
hold on;
plot(time, predicted_average_social_worse, '--og', 'DisplayName', 'Social start positive (original)');
plot(time, predicted_average_nonsocial_better, '-sg', 'DisplayName', 'Nonsocial start negative (original)');
plot(time, predicted_average_nonsocial_worse, '--sg', 'DisplayName', 'Nonsocial start positive (original)');

% Calculate flipped lines
flipped_predicted_social_better = midpoint + (midpoint - predicted_average_social_better);
flipped_predicted_social_worse = midpoint + (midpoint - predicted_average_social_worse);
flipped_predicted_nonsocial_better = midpoint + (midpoint - predicted_average_nonsocial_better);
flipped_predicted_nonsocial_worse = midpoint + (midpoint - predicted_average_nonsocial_worse);

% Plot flipped lines (all orange, distinguish social/nonsocial with shape and better/worse with line style)
plot(time, flipped_predicted_social_better, '-o', 'Color', [1 0.5 0], 'DisplayName', 'Social start negative (flipped)');
plot(time, flipped_predicted_social_worse, '--o', 'Color', [1 0.5 0], 'DisplayName', 'Social start positive (flipped)');
plot(time, flipped_predicted_nonsocial_better, '-s', 'Color', [1 0.5 0], 'DisplayName', 'Nonsocial start negative (flipped)');
plot(time, flipped_predicted_nonsocial_worse, '--s', 'Color', [1 0.5 0], 'DisplayName', 'Nonsocial start positive (flipped)');

% Adjust axis and labels
axis([1 6 0 100]);
xlabel('Time');
ylabel('Impression');
title('Visualization of Predicted Values (generic model)');
xticks(1:1:6);
legend('show');

% Panel 3: Visualization of Predicted Values (baseline model)
subplot(1, 3, 3); % Create the third subplot (1 row, 3 columns, third panel)

% Plot original lines (all green, distinguish social/nonsocial with shape and better/worse with line style)
plot(time, predicted_average_social_better_base, '-og', 'DisplayName', 'Social start negative (original)');
hold on;
plot(time, predicted_average_social_worse_base, '--og', 'DisplayName', 'Social start positive (original)');
plot(time, predicted_average_nonsocial_better_base, '-sg', 'DisplayName', 'Nonsocial start negative (original)');
plot(time, predicted_average_nonsocial_worse_base, '--sg', 'DisplayName', 'Nonsocial start positive (original)');

% Calculate flipped lines
flipped_predicted_social_better_base = midpoint + (midpoint - predicted_average_social_better_base);
flipped_predicted_social_worse_base = midpoint + (midpoint - predicted_average_social_worse_base);
flipped_predicted_nonsocial_better_base = midpoint + (midpoint - predicted_average_nonsocial_better_base);
flipped_predicted_nonsocial_worse_base = midpoint + (midpoint - predicted_average_nonsocial_worse_base);

% Plot flipped lines (all orange, distinguish social/nonsocial with shape and better/worse with line style)
plot(time, flipped_predicted_social_better_base, '-o', 'Color', [1 0.5 0], 'DisplayName', 'Social start negative (flipped)');
plot(time, flipped_predicted_social_worse_base, '--o', 'Color', [1 0.5 0], 'DisplayName', 'Social start positive (flipped)');
plot(time, flipped_predicted_nonsocial_better_base, '-s', 'Color', [1 0.5 0], 'DisplayName', 'Nonsocial start negative (flipped)');
plot(time, flipped_predicted_nonsocial_worse_base, '--s', 'Color', [1 0.5 0], 'DisplayName', 'Nonsocial start positive (flipped)');

% Adjust axis and labels
axis([1 6 0 100]);
xlabel('Time');
ylabel('Impression');
title('Visualization of Predicted Values (baseline model)');
xticks(1:1:6);
legend('show');

%%  model selection
%a general model selection
[p_general, o_general] = VBA_groupBMC ([logEvidence_social,logEvidence_nonsocial]);


%% 
% mean of each cell
nanmean(social_getting_better)%  0.8350
nanmean(nonsocial_getting_better)%0.7284
nanmean(nonsocial_getting_worse)%-1.0050
nanmean(social_getting_worse)%-0.9587

getting_worse_abs=abs([posterior_getting_worse(3,:).muTheta]);
getting_better_abs=abs([posterior_getting_better(3,:).muTheta]);

[h5, p5, ci5, stats5] = ttest2(getting_better_abs, getting_worse_abs)%sig!


cohen_d_bw = calculateCohensD_EqualSampleSizes(getting_better_abs, getting_worse_abs)

[h6, p6, ci6, stats6] = ttest2([nonsocial_getting_better_abs,nonsocial_getting_worse_abs], [social_getting_better_abs,social_getting_worse_abs])%nonsig

mean([nonsocial_getting_better,social_getting_better]);

social_getting_better = social_getting_better(~isnan(social_getting_better));
social_getting_worse = social_getting_worse(~isnan(social_getting_worse));
nonsocial_getting_better = nonsocial_getting_better(~isnan(nonsocial_getting_better));
nonsocial_getting_worse = nonsocial_getting_worse(~isnan(nonsocial_getting_worse));



%compare it with 0
mean([posterior_getting_worse(3,:).muTheta])%-0.9816
[h1, p1, ci1, stats1] = ttest([posterior_getting_worse(3,:).muTheta], 0);%sig
mean([posterior_getting_better(3,:).muTheta])% 0.7811
[h2, p2, ci2, stats2] = ttest([posterior_getting_better(3,:).muTheta], 0)%sig

[h, p, ci, stats] = ttest2([posterior_getting_worse(3,:).muTheta], [posterior_getting_better(3,:).muTheta]);

mean([posterior_social(3,:).muTheta])% -0.0711
std([posterior_social(3,:).muTheta])
[h3, p3, ci3, stats3] = ttest([posterior_social(3,:).muTheta], 0)%nonsig

mean([posterior_nonsocial(3,:).muTheta])%-0.1294
[h4, p4, ci4, stats4] = ttest([posterior_nonsocial(3,:).muTheta], 0)%nonsig

[h_sn, p_sn, ci_sn, stats_sn] = ttest2([posterior_social(3,:).muTheta], [posterior_nonsocial(3,:).muTheta])% nonsig
cohen_d_sn = calculateCohensD_EqualSampleSizes([posterior_social(3,:).muTheta], [posterior_nonsocial(3,:).muTheta]); %effect size


%% calculate the correlation between power and bias strength (optional)
general_strength=[posterior_social(1,:).muTheta;posterior_nonsocial(1,:).muTheta];

%for social part
[r,p]=corrcoef([posterior_social(1,:).muTheta],Power_scores) %non.sig
[r2,p2]=corrcoef([posterior_nonsocial(1,:).muTheta],Power_scores) %non.sig


[r,p]=corrcoef([posterior_getting_better(1,:).muTheta],Power_scores)%non.sig
[r,p]=corrcoef([posterior_getting_worse(1,:).muTheta],Power_scores)%non.sig


%%
figure;
hist([posterior_getting_better(1,:).muTheta])