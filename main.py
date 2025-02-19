import bernoulli
import randomforest
import syntheticrandomforest

#mode = input("Bernoulli (1); RandomForest (2); RandomForest sul dataset sintetico (3)")
mode = 1
if mode == 1:
    bernoulli.train()
elif mode == 2:
    randomforest.train()
elif mode == 3:
    syntheticrandomforest.train()