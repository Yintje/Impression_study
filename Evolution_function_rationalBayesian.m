function[x]=Evolution_function_rationalBayesian(x, P, u, in) %input: current impression, bias coefficient, new evidence, additional info(empty)

x = exp(x);

% don't reweigh new evidence 

    subjective_info=u;

% Update accumulated internal state
x(1) = x(1) + subjective_info(1);
x(2) = x(2) + subjective_info(2);

x = log(x);

end