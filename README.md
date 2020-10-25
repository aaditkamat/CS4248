# CS4248 Assignments

Both assignments involve creating POS taggers that can be trained on a given
data set. These POS taggers are then run against the test data set to assess
the accuracy of the model.

In Assignment 1, we are supposed to model the weights by calculating the emission and transition probabilities based on the relative frequencies of the
POS tag and word pairs as they appear in the training data set.

In Assignment 2, on the other hand, the model weights are obtained by training an LSTM based POS tagger on the training data set.

Since the training process is time consuming, it is advised to run the scripts on a machine that has GPU cores to provide sufficient computing power.

An xgpd machine can be obtained from the SoC Computing Cluster by following
[this guide](https://dochub.comp.nus.edu.sg/cf/services/compute-cluster)(requires you to have access to the NUS SoC network).
