# On failure modes of molecule generators and optimizers
Philipp Renz <sup>a</sup>, 
Dries Van Rompaey  <sup>b</sup>, 
Jörg Kurt Wegner  <sup>b</sup>, 
Sepp Hochreiter  <sup>a</sup>, 
Günter Klambauer  <sup>a</sup>, 

<sup>a</sup> LIT AI Lab & Institute for Machine Learning, Johannes Kepler University Linz, Altenberger Strasse 69, A-4040 Linz, Austria

<sup>b</sup> High Dimensional Biology and Discovery Data Sciences, Janssen Research & Development, Janssen Pharmaceutica N.V., Turnhoutseweg 30, Beerse B-2340, Belgium

## Abstract
There has been a wave of generative models for molecules triggered 
by advances in the field of Deep Learning.
These generative models are often used to optimize chemical 
compounds towards particular properties or a desired 
biological activity. The evaluation of 
generative models remains challenging and 
suggested performance metrics or scoring functions 
often do not cover all relevant aspects of drug design projects. 
In this work, we highlight some unintended failure modes of 
generative models and how these evade 
detection by current performance metrics.

## Code..
..is being refactored and cleaned up and will come soon.

## Teaser
.. when optimizing molecules guided by a machine learning model the optimizer exploits biases in the model
![Scores](https://raw.githubusercontent.com/ml-jku/mgenerators-failure-modes/master/controlscores.png)
