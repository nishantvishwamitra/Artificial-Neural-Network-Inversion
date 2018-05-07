#ANN Inversion and Garbage Pattern Filtering
Nishant Vishwamitra 

ANN Inversion Preliminaries
In the ANN inversion method developed in Linden, Alexander, and J. Kindermann [1], an input vector is iteratively computed by minimizing the error between the output and the target output, while keeping the weights and the output vector fixed. A backpropagation algorithm is suggested to propagate back the error signals [1] from the output layer to the input layer, where the input pattern is iteratively corrected using the error signals to minimize error. A valuable use of the ANN inversion is to evaluate the generalization of a trained MLFF ANN. As shown in [2],
  
inversion can be thought of as a constraint satisfaction problem with fixed weights and outputs and variable inputs.
Generalized Delta Rule for ANN Inversion
In this section, a generalized delta rule is derived to compute deltas for different MLFF ANN layers of inverted ANN. These deltas are used in back propagation procedure to train the ANN to iteratively compute garbage inputs for fixed weights and outputs.
Delta at Output Layer
Figure 2: Application of chain rule for an output unit
The error at the output layer (Figure 2) is defined as follows. 𝐸"# = %& (𝑡"# − 𝑎"#)& (i)
In Figure 2, note that the weights (wji) and the target (tjP) are both fixed. Therefore, to change the input, we find the rate of change of the error at the output (EjP) with respect to the net input to the unit (netj).
A formulation for sensitivity dj is as follows.
𝛿" = − ./ (ii)
.0123
Therefore, a solution to back propagate the sensitivities is developed as follows [1, 2]. Δ𝑛𝑒𝑡" = 𝜖𝛿" (iii)
𝑎"89% = 𝑎"8 + Δ𝑛𝑒𝑡" (iv)
In Equation (iv), k is the number of iteration. Applying chain rule to the dj formulation above.
𝛿"=−./ =−./ .;3 (v) .0123 .;3 .0123
=−12×2×𝑡"#−𝑎?# ×−1×𝑓A(𝑛𝑒𝑡")
           
 𝛿"= 𝑡"#−𝑎?# 𝑓A(𝑛𝑒𝑡")
(vi)
The above equation gives the formulation of delta at the output layer units.
Delta at Hidden and Input Layers
Figure 3: Application of chain rule at hidden/input unit. The error (Ep) is propagated back to input layer by backpropagating deltas.
The sensitivity at the hidden/input layer units (Figure 3) is defined as follows. 𝛿B = − ./ (vii)
    Applying chain rule.
.012C
𝛿B=− ./ =− ./ .0123 (viii) .012C .0123 .012C
   But from Equation (vi), we have 𝛿" = − ./ . Applying chain rule further. .0123
= −𝛿 .0123 .;C (ix) " .;C .012C
(x)
Therefore, from the Equation (x), we can see that the deltas (or sensitivities [2]) are propagated from the output layers to the input layers. The input vector corrections can be made in the same way as given in the Equations (iii) and (iv).
Approach
In this section, the development of the ANN inversion process is discussed in detail.
    𝛿B= "𝜕"𝑤"B𝑓A(𝑛𝑒𝑡B)
 
Training a MLFF ANN to Classify Numbers
First, a MLFF ANN is trained to classify handwritten digits (0-9) from the MNIST [7] dataset. This dataset consists of image samples that are of size 28 X 28. In this experiment, 10 samples of each digit were randomly selected for training from the MNIST dataset (a total of 100 patterns for training). The ANN configuration of 784/40/4 was used in this experiment. All input patterns were converted to flat vectors so that they could be input into the ANN. The logistic sigmoid squasher function [2] was used in the units of the hidden layer and output layer. In this ANN, the input layer units were just holds. Standard backpropagation based on the generalized delta [2] rule was used to train this ANN. The training was performed for 50000 iterations (train by pattern method).
Figure 4: Error Vs. Iterations Plot for MLFF ANN Digits Classification
Figure 4 depicts the Error Vs. Iterations plot for the digits classifier training. It can be seen that a local minimum is approached around iteration 40000, after which the computation is stopped.
ANN Inversion to Generate Garbage Inputs
Second, the same ANN architecture given in the previous section is used in the ANN inversion process. In this experiment, the input layer units also map (logistic sigmoid squasher) and are not just holds. The Equations (vi) and (x) are used to propagate sensitivities from the output layer back to the input layer to make input corrections according to Equations (iii) and (iv), while keeping the weights and the target fixed. The training was done for 200000 iterations (train by pattern method) for different targets. An example of the Error Vs. Iterations plot for target 0110 is depicted in Figure 5.
 
 Figure 5: Error Vs. Iterations Plot for MLFF ANN for Inversion Training
The Error Vs. Iterations plot shows that a local minimum is approached around the 130000 iterations. But the more interesting aspect is the garbage input that was found to generate this target (0110, or 6). After converting the flat input back to an image matrix, the result is as shown below, in Figure 6.
Figure 6: A sample garbage input for target 0110. The figure on the left is the garbage input generated for an I0 and the figure of the right is a sample input from the dataset that corresponds to the output 0110.
From Figure 6, the garbage input does not compare to the real input in the original dataset. More such samples have been provided in the Appendix. Such garbage inputs could be used as adversarial patches [3,4,5] to fool a classifier into misclassifying inputs. For example, a garbage input corresponding to a high-speed limit could be used to attack a self-driving car, to make it go faster on a low speed limit road. Therefore, a mitigation for this problem could be to re-train the original MLFF ANN to ignore the garbage inputs by classifying them into different outputs. The next section further explains this procedure.
  
Re-train MLFF ANN to Ignore Garbage Inputs
In the third experiment, the MLFF ANN is re-trained to ignore the garbage input computed in the previous section. The same dataset is used in this experiment as the one in the training of the MLFF ANN for digits classification. The ANN configuration of 784/40/5 was used in this experiment. Note that in the output layer, an additional unit has been added. This units is trained to activate when the garbage input is passed into the network. As in the first experiment, logistic sigmoid squasher function is used in the units of the hidden layer and output layer. Standard backpropagation based on the generalized delta [2] rule was used to train this ANN. The training was performed for 50000 iterations (train by pattern method).
Figure 6: Error Vs. Iterations Plot for MLFF ANN for Garbage Input Training
The garbage input training results are shown in the Figure 6. Therefore, it has been shown that these garbage inputs, or adversarial attacks can be mitigated by re-training the ANN to ignore them.
Code Listing
A map of the attached code and functionality of each resource is provided in this section.
• data/: Directory that contains all saved objects
• garbage_mlff_ann_training.txt: Error Vs. iteration plot data
• garbage_train_mlff_ann.py: Code to train ANN to ignore garbage input
• inversion_ann_training.txt: Error Vs. iteration plot for ANN inversion
• invert_mlff_ann.py: Code to compute garbage inputs
• mlff_ann_training.txt: Error Vs. iteration plot data for digits classifier training
• mnist_reader.py: Code to read MNIST dataset [6]
• my_mlff_ann.py: Code to train ANN to classify digits
 
data/ contents (these are saved objects):
• X: Input vectors
• bh: Hidden layer bias
• bout: Output layer bias
• garbage_input: A garbage input sample • wh: Hidden layer weights
• wout: Output layer weights
• y: Output vectors
Effort Summary
In this project, back propagation equations for ANN inversion were derived. An MLFF ANN was first trained to classify digits from the MNIST dataset. Then, garbage inputs were computed with fixed weights of the trained ANN. Next, an ANN was re-trained to ignore the garbage input by training it to classify the garbage input as a different output. In this way, it was shown that adversarial attacks can be defended against by training the ANN to ignore the garbage inputs.
For future work, the procedure would be enhanced to include real world image datasets. Another improvement is to device a procedure to find all such garbage inputs in the input space and not find them one by one (a direction could be random search [2]).
References
1.
2.
3. 4.
5.
6. MNIST reader. https://gist.github.com/akesling/5358964
7.
Appendix
Some additional garbage input samples have been computed and depicted in this section.
1. Pattern 1000 (8)
 Linden, Alexander, and J. Kindermann. "Inversion of multilayer nets." Proc. Int. Joint Conf. Neural
 Networks. Vol. 2. 1989.
 Schalkoff, Robert J. Artificial neural networks. Vol. 1. New York: McGraw-Hill, 1997.
 Brown, Tom B., et al. "Adversarial patch." arXiv preprint arXiv:1712.09665 (2017).
 Szegedy, Christian, et al. "Intriguing properties of neural networks." arXiv preprint
 arXiv:1312.6199 (2013).
 Athalye, Anish, Nicholas Carlini, and David Wagner. "Obfuscated Gradients Give a False Sense of
 Security: Circumventing Defenses to Adversarial Examples." arXiv preprint arXiv:1802.00420 (2018).
  Deng, Li. "The MNIST database of handwritten digit images for machine learning research [best of
 the web]." IEEE Signal Processing Magazine 29.6 (2012): 141-142.

  Observation: Some patterns (like this one) need large number of iterations and momentum (need uncomment this part in invert_mlff_ann.py).
2. Pattern 0001 (1)
3. Pattern 0101 (5)
4. Pattern 0000 (0)
    
  5. Pattern 0010 (2)
  
