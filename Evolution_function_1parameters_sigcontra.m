function[x]=Evolution_function_1parameters_sigcontra(x, P, u, in) %input: current impression, bias coefficient, new evidence, additional info(empty)

% Check if u(3) is 3, 4, or 5
    if u(3) == 1|| u(3) == 3 || u(3) == 4 || u(3) == 5
        % Convert state to exponential scale
        x = exp(x);
        
        % Reweigh new evidence (by design)
        subjective_info = [u(1) * VBA_sigmoid(P(1)); u(2) * (1 - VBA_sigmoid(P(1)))];
        
        % Update accumulated internal state
        x(1) = x(1) + subjective_info(1);
        x(2) = x(2) + subjective_info(2);
        
        % Convert state back to log scale
        x = log(x);
    else
        x = x;
    end
    % If u(3) is not 3, 4, or 5, x remains unchanged
end