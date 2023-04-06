# Modified-Adversarial-Network
Machine learning model that out perform Deep neural network for analysis of tau lepton and neutrino final states.

# mAN

## A brief to git
Before clone the respiratory please follow the instruction,
* fork the respiratory to your github account
* clone the respiratory from your github
* create a new branch
* Done !!
## Signal Sample Production
The prodction details of the **W'-> tau nu** process are given below,

Madgraph2.99 used for sample production

Mass points are 3,4,5,6 TeV

## DNN training 
The exclusion limits obtained from the test statics of **W'** invariant mass. The invariant mass obtained from the ML regression technique. The ML model can find at the ``` training/regression.py``` file. Type ```python3 training/regression.py --help``` in terminal which gives the details code inputs and usage.

```
   -tr,"--train",help="to train the model",action="store_true")
   -ts,"--test",help="to test the model",action="store_true")
   -b,"--back",help="boolean set for background")

```


Once the training is performed, the model parameter saved to the ```models``` directory.

An example for how to train the code,
```
python3 mAN.py -tr -b
```

for testing,
```
python3 mAN.py -ts -b
```

if you want to test the model for signal samples, remove ```-b``` boolean


for converting csv files to root file use convertroot.py
