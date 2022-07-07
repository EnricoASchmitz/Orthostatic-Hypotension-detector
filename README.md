# Orthostatic hypotenstion detector



# Input data
The expected input for the data are 40 seconds of sitting and 150 seconds during standing of oxygenated heamoglobin and deoxigneated heamoglobin signals.
In the config file settings can be changed to use it on your own data, adjust the data preposessing or the used model.


## To use the current preprossor
You need the following features in a matlab file:
- "BP", blood pressure signal
- "oxyvals", oxygenated heamoglobin signal
- "dxyvals", deoxygenated heamoglobin signal
- "ADvalues", contains the movement sensor
- "markerstijd", contains names and times of transition
expected timestep names:
  - "start1",
  - "move1",
  - "stand1",
  - "stop1",
  - "start2",
  - "move2",
  - "stand2",
  - "stop2",
  - "start3",
  - "move3",
  - "stand3",
  - "stop3" 
- "nirs_time", column with the timesteps
- "fs_nirs", number of Hz

## How to implement your own method(model or preprocessing your own data.)
  1. Create a class based on the abtract class
  2. Add the class to the corresponding creator and the enums file in the write column
  3. Implement all abstract methods
  4. Use the new method in the config file

# predicting
From the input data the algorithm will infer characterisitcs about the SBP and DBP after standing up.
The infered characteristics can be used to make a diagnosis if the patient has OH.

# Data will be saved to MLflow
The data is saved with MLflow, this contains a ui which we can use to compare trained and optimized models.
Also including used the model

# Optimization
Optimzation is done with Optuna, this performs a number of trials and save the best trial to MLflow
After optimizing we need to use the optimized parameter in a fitting task to get the model saved.
