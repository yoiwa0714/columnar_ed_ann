# _Error Diffusion Learning Algorithm Sample Program_

_Released: July 12, 1999  
July 16, 1999: Details added  
August 6, 1999: Partial modifications  
August 19, 1999: Paper added  
October 27, 1999: Progress report_

---

[ED Method Sample Program (UNIX general purpose, tgz, 4KB)](https://web.archive.org/web/19991124023203/http://village.infoweb.ne.jp:80/~fwhz9346/ed.tgz)

This is a sample program for the Error Diffusion Learning Algorithm (ED method), which is a supervised learning algorithm for hierarchical neural networks.

For comparison, we also show a sample program for the Backpropagation method (BP method) used for comparison.

[BP Method with Momentum Sample Program (UNIX general purpose, tgz, 4KB)](https://web.archive.org/web/19991124023203/http://village.infoweb.ne.jp:80/~fwhz9346/bp.tgz)

Note that these programs calculate hierarchical structures by considering them as a type of recurrent structure. Also, when inputting parameters, simply pressing return will use the default values (values in parentheses), so you can just hit return repeatedly during execution. Additionally, X-Window is used to display total error graphs, so please properly set the DISPLAY environment variable when executing.  
# Otherwise it will core dump (^^;

Added on 8/19

We are releasing the paper on the Error Diffusion Learning Algorithm that is planned for submission. Note that this ED method-related material is scheduled to be completely removed after a while.

[Paper on Error Diffusion Learning Algorithm (PDF file, 575KB)](https://web.archive.org/web/19991124023203/http://village.infoweb.ne.jp:80/~fwhz9346/edla.pdf)

Also, if you don't understand BP itself, you won't understand what this is about at all, so please check [this area](https://web.archive.org/web/19991124023203/http://www.gifu-nct.ac.jp/elec/deguchi/sotsuron/takemura/node13.html). Note: This is an unauthorized link (^^;

Added on 10/27

Since I've already cut the link from the front page, anyone who notices this is quite unusual (laugh). Here's a progress report on this.

The submitted paper is currently back for review and under revision. I plan to return it by the end of the year. Note that "review" means the reviewers had questions and asked for corrections.

The main opinion coming out is something like "it's unclear whether this is brain learning theory or engineering-meaningful neural theory, so please do something about this ambiguity..." I'm pushing for the former, so I'm changing the paper content to make Figures 1 and 2 below the main focus of the paper, and discuss what was previously the main content (Figures 3 and 4) in the consideration section. The experimental results section doesn't change particularly.

Eventually, after all this back and forth, it will probably go to a third review, but I'd be happy if the paper passes. The paper.

---

## Author's Comments

First, let me clarify that unlike my other pages which are done for fun, this page is serious. The others are just hobby "programming," but this page alone is a hobby "research" page (well, this is also a hobby, though (^^;)).

Now, let me explain from the basics. First, when it comes to supervised hierarchical neural network learning algorithms, the Backpropagation method (BP method) is famous. Simple BP targets feedforward hierarchical neural networks without recursive structures for learning. However, unlike perceptrons, BP can learn hierarchical neural networks with three or more layers having intermediate layers. This solved the problem of first-generation neural boom perceptrons that could only solve linearly separable problems, leading to the second neural boom.

Now then. Let's consider what the problems with this BP method are. Of course, specialists would first mention BP's slow convergence and the heuristic nature of various parameters like the number of intermediate layers, but as a simulation specialist, what I first think about BP is that when considered as a simulation of actual neural systems, BP is far too strange. With perceptrons, as they are considered models of the cerebellum (Marr's theory), they are natural as models of actual neural systems. However, the way neural networks are conceived from BP onwards is far too unnatural as models of actual neural systems. I particularly cannot accept the part where error information in BP is computed while flowing backward through axons (BP also has strange concepts like momentum terms).

So, are learning processes not being performed in multilayer neural networks with intermediate layers in actual neural systems? That would also be strange. With perceptron-type learning methods that don't learn intermediate layers, the solvable problems are very limited to only linearly separable ones. Famously, they cannot even learn exclusive OR (XOR).

## The Core Problem

The fundamental question is: How can biological neural systems perform complex learning without the artificial mechanisms required by backpropagation?

### Biological Implausibility of BP

1. **Information Flow Direction**: In BP, error signals must flow backward through the same pathways that carry forward signals, which is biologically impossible
2. **Global Coordination**: BP requires precise coordination of forward and backward passes across the entire network
3. **Detailed Gradient Information**: Neurons would need to compute and transmit precise gradient information, which exceeds biological capabilities

### The ED Method Solution

The Error Diffusion method addresses these issues by:

1. **Chemical Diffusion**: Using amine-like neurotransmitter concentrations to carry error information
2. **Local Learning Rules**: Each connection updates based only on locally available information
3. **Biological Structures**: Incorporating excitatory/inhibitory neuron distinctions and spatial diffusion

## Technical Implementation

### Network Architecture

The ED method implements several key biological principles:

1. **Excitatory/Inhibitory Pairs**: Each input is represented by both excitatory and inhibitory neurons
2. **Dale's Principle**: Neurons are either purely excitatory or purely inhibitory
3. **Spatial Organization**: Neurons are organized in columns that share error information

### Learning Mechanism

Instead of backpropagating errors through connections, the ED method:

1. **Diffuses Error**: Error information spreads as chemical concentrations
2. **Local Updates**: Each synapse updates based on local activity and error concentration
3. **Natural Constraints**: Weight updates respect biological constraints automatically

## Experimental Validation

### Comparison with BP

The original implementation showed:

1. **Equivalent Performance**: ED method achieved similar learning outcomes to BP
2. **Biological Plausibility**: No requirement for backward information flow
3. **Stability**: More robust learning without gradient explosion/vanishing issues

### XOR Problem

The classic XOR problem, unsolvable by simple perceptrons, was successfully learned by the ED method, demonstrating its capability for non-linearly separable problems.

## Historical Context

### Development Timeline

- **1999**: Initial development and publication by Isamu Kaneko
- **Focus**: Bridging the gap between biological reality and artificial learning
- **Motivation**: Creating learning algorithms that could actually be implemented by biological neural systems

### Research Impact

The ED method represented a significant departure from the prevailing focus on computational efficiency toward biological plausibility. While backpropagation became the dominant training method for artificial neural networks, the ED method provided insights into how biological systems might actually perform complex learning.

## Modern Relevance

### Neuromorphic Computing

With the rise of neuromorphic computing and brain-inspired hardware, the ED method's biological plausibility makes it particularly relevant for:

1. **Low-Power Computing**: Energy-efficient learning without global coordination
2. **Local Processing**: Learning rules that can be implemented in distributed hardware
3. **Fault Tolerance**: Robust learning that doesn't depend on precise global information

### Spiking Neural Networks

The integration of ED method with Spiking Neural Networks (as implemented in this project) represents a natural evolution, combining:

1. **Temporal Dynamics**: Spike-based information processing
2. **Biological Learning**: Error diffusion through chemical-like mechanisms
3. **Local Plasticity**: Synaptic changes based on local activity patterns

## Conclusion

The Error Diffusion Learning Algorithm represents a fundamental rethinking of how neural networks can learn. By abandoning the artificial constraints of backpropagation in favor of biologically plausible mechanisms, it opens new possibilities for both understanding biological intelligence and implementing efficient artificial learning systems.

While backpropagation has dominated artificial neural networks due to its computational efficiency, the ED method's biological realism makes it invaluable for neuromorphic applications and our understanding of how brains actually learn.

---

**Note**: This document is based on Isamu Kaneko's original research and implementation from 1999. The ED method continues to inspire research into biologically plausible learning algorithms and neuromorphic computing systems.

**Historical Links**: The original links in this document point to archived versions of Kaneko's original materials, preserved for historical and research purposes.