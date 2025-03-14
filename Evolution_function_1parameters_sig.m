function[x]=Evolution_function_1parameters_sig(x, P, u, in) %input: current impression, bias coefficient, new evidence, additional info(empty)

x = exp(x);
% reweigh new evidence (by design)
    subjective_info=[u(1)*VBA_sigmoid(P); u(2)*(1 -VBA_sigmoid(P))];
  
% Update accumulated internal state
x(1) = x(1) + subjective_info(1);
x(2) = x(2) + subjective_info(2);
x = log(x);
end


