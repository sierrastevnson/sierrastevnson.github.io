Notes:
* Your report should be in the region of 2000-2500 words with three to four figures, and written in a scientific language and style.
* Need to include dataset with code
* Report delivered via website (submit link to website); can write up report directly into website or as a pdf and linked in website
* Code must be submitted on BruinLearn (could be shared in a google colab notebook)

### Introduction
<img src="https://cdn.britannica.com/46/109746-050-9511BBEF/differences-layers-ionosphere-Earth.jpg" align="right" alt="Encyclopædia Britannica, Inc." width="40%"/> 
The ionosphere is the layer of the Earth's atmosphere with a significant concentration of particles charged via radiation, particularly from the sun. Shortwave (high frequency) radio waves emitted from the Earth are reflected by the charged ions and electrions within the ionosphere back towards the Earth, rather than be sent out to space.

I retrieved data from the UC Irvine Machine Learning Repository at [this link](https://archive.ics.uci.edu/dataset/52/ionosphere]). The data was collected by a radar system in Goose Bay, Labrador, affiliated with the Space Physics Group of The Johns Hopkins University Applied Physics Laboratory. The radar system consists of 16 high frequency antennas which transmit a multi-pulse pattern to the ionosphere and, between pulses, measure the phase shift of the returns.

The paper behind the dataset, [Sigillito et al. (1989)](https://secwww.jhuapl.edu/techdigest/Content/techdigest/pdf/V10-N03/10-03-Sigillito_Class.pdf), uses the following equations to denote the received signal at time $t$ $C(t)$ and the autocorrelation function $R(t,k)$ as:

$$
\begin{align*}
C(t) &= A(t) + iB(t) \\
R(t,k) &= \sum_{i=0}^{16} C(t+iT)  \overline{C} \left(t + (i + k)T\right)
\end{align*}
$$

where $T$ is the pulse repetition period, $k$ is the pulse number from 0 to 16, and $\overline{C}$ is the complex conjugate of $C$.

Because the signal is complex, the autocorrelation function is also complex. The dataset consists of 17 pairs of numbers; 17 corresponding real and imaginary components of the autocorrelation function. In signal processing, a real part of a received signal corresponds to the component of the signal that lies in-phase with the reference signal (in our case, the original pulse) and the imaginary component lies orthogonal to it.
Prior to the publication, filtering out noisy signals was a labor-intensive task for researchers, but the use of machine learning allows for the process to be automated with high accuracy. The paper compared single- and multi-layered feedforward neural networks to produce their results.

### Project Overview
For my project, I wanted to compare a variety of machine learning techniques to determine which prove most effective for such a task. Since each signal return is classified in the dataset as either "good" or "bad," this is a binary classification task. Each attribute of the data — in this case, each of the 17 real or 17 imaginary parts of the autocorrelation function from a certain pulse — is labeled only as Attribute1, Attribute 2, etc, so there is little opportunity for reviewing the data on a humanistic basis before running quantitative analytical tools.

I will perform the following:

1. PCA for dimension reduction
2. Logistic regression with regularization for model prediction
3. Random forest model for feature ranking
4. Neural network (type TBD?) as alternative method for model prediction

### 1. Principal Component Analysis






