We provide the complete code implementation and experiments records, including the Morse neural network models and the training logs. The oracle models are not provided due to their excessive sizes. Should access to the oracle models be necessary, they can be reproduced upon request.


To reproduce the results presented in the paper, you can directly run the script:

```
bash train.sh
```

To reproduce the result on a specific environment and a specific random seed, you can run the code:
```
python cql_query.py --env $your_env --seed $your_seed
```