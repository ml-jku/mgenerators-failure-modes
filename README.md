# Tricky generators

Some papers introduced molecule generators that create molecules based on the predictions of a learned property prediction model.

Here we show that this might go wrong.

## Genetic Algorithm can exploit classifiers

We show that some models can exploit classifiers
Some more experiments we could do include:

Possible causes include
- Features: maybe ECFPs are not well suited here, try MACCS
- Classifier: maybe RFs are easy to trick
- To little data: maybe models trained on little data are bad
- Optimizer is bad: maybe other optimizers would do better

I think that this list is sorted by importance in ascending order.
It will be interesting to find out if this is really the case.

