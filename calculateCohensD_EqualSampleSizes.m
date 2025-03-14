function d = calculateCohensD_EqualSampleSizes(group1, group2)
    % Function to calculate Cohen's d for two groups with equal sample sizes
    % Input: group1 and group2 - numeric arrays of the two groups
    % Output: d - Cohen's d effect size
    
    % Ensure the groups have equal number of participants
    if length(group1) ~= length(group2)
        error('The two groups must have an equal number of participants.');
    end
    
    % Calculate means and standard deviations for each group
    mean1 = mean(group1);
    mean2 = mean(group2);
    std1 = std(group1);
    std2 = std(group2);
    
    % Calculate the pooled standard deviation
    pooled_std = sqrt((std1^2 + std2^2) / 2);
    
    % Calculate Cohen's d
    d = (mean1 - mean2) / pooled_std;
end