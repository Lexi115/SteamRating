import bernoulli, multinomial
import randomforest
import syntheticrandomforest

#mode = input("Bernoulli (0); Multinomial (1); RandomForest (2); RandomForest sul dataset sintetico (3)")
mode = 2
if mode == 0:
    bernoulli.train()
elif mode == 1:
    multinomial.train()
elif mode == 2:
    randomforest.train()
elif mode == 3:
    syntheticrandomforest.train()