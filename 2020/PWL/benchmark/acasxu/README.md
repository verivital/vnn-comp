Using all of the ACAS-Xu networks, with timing, verification, etc. results to be generated as a table for each participant. Probably ~5min for each problem instance in ACAS-Xu(six hours for the hard ones), which would consist 
of the network, the property, and the input set.

ACASXU-HARD:
1. net 4-6, prop 1
2. net 4-8, prop 1
3. net 3-3, prop 2
4. net 4-2, prop 2
5. net 4-9, prop 2
6. net 5-3, prop 2
7. net 3-6, prop 3
8. net 5-1, prop 3
9. net 1-9, prop 7
10. net 3-3, prop 9

Input / output scaling when using ACASXu, which is not in the ONNX files. The code Stanley Bak used to do this for input scaling is:

  `means_for_scaling = [19791.091, 0.0, 0.0, 650.0, 600.0, 7.5188840201005975]`
  
  `range_for_scaling = [60261.0, 6.28318530718, 6.28318530718, 1100.0, 1200.0]`

  `for i in range(5):`
  
      input[i] = (input[i] - means_for_scaling[i]) / range_for_scaling[i]

  Output scaling doesn't matter since the advisories use relative values of outputs, except for property 1, which checks an absolute limit on the output.

The code he used to construct a linear constraint for property 1 is this:

`# unsafe if COC >= 1500`

`# Output scaling is 373.94992 with a bias of 7.518884`

`output_scaling_mean = 7.5188840201005975`

`output_scaling_range = 373.94992`
        
`# (1500 - 7.518884) / 373.94992 = 3.991125`

`threshold = (1500 - output_scaling_mean) / output_scaling_range`

`# spec is unsafe if output[0] >= threshold`

The threshold value of 3.991125 is consistent with what Marabou uses for property 1: https://github.com/guykatzz/Marabou/blob/master/resources/properties/acas_property_1.txt
