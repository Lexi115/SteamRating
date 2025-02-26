import bernoulli
import randomforest
import syntheticrandomforest

mode = 1
if mode == 1:
    bernoulli.train()
elif mode == 2:
    randomforest.train()
elif mode == 3:
    syntheticrandomforest.train()