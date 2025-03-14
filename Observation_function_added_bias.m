function[gx]=Observation_function_added_bias(x, Phi, u, in) 
x = exp(x);

impression = x(1) ./ (x(1) + x(2)+eps);

gx = 100*impression+Phi;