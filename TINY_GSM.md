- https://huggingface.co/datasets/TinyGSM/TinyGSM
- https://arxiv.org/pdf/2312.09241

To our surprise, in the case of GSM8K, we are able to bridge the performance gap between the student and teacher, 
by utilizing a tiny amount of labeled real data (the original GSM8K training set of 7k questions) to `train an
independent verifier model`. At test time, the verifier score and select among multiple candidate answers generated
from the student, and then we output the highest score generation as the final submission.

Note the idea of usin a verifier is proposed by the seminal GSM8K paper https://arxiv.org/pdf/2110.14168

We introduce TinyGSM, a synthetic dataset containing GSM8K-style math word problems paired with Python
solutions, generated fully by GPT-3.5-turbo. TinyGSM consists of 12.3M questions which amount to 1.8B tokens.
We demonstrate TinyGSM’s high-quality by `finetuning the Phi-1.5 1.3B model` (before the use of verifiers) which
improves its accuracy `from 44.6% to 68.2%` on the GSM8K test set. Notably, our `smallest 125M model can also
achieve 63.1% after finetuning on TinyGSM`.

When integrated with a verifier for scoring generations, 1.3B model achieves 81.5% accuracy, significantly outperforming
existing open-source models and even rivaling the 77.4% accuracy of GPT-3.5, from which TinyGSM is generated.

- - -

TinyGSM: augmenting GSM8K with synthetic generations Despite the high quality, the GSM8K training set
only contains 7473 problems, which is too small for training a reasonably sized language mode.

We prompt GPT-3.5-turbo to generate problem variants similar to a given question (but not the solution) randomly
sampled from the GSM8K training set. Each problem variant contains both a question and the corresponding solution
written in Python, as shown in Figure 2.1 Using code allows us to leverage a Python interpreter, circumventing
language models’ known limitation regarding numerical calculations and code execution

![](img/tiny-gsm-00.jpg)

To enhance robustness, we also generated synthetic problems whose questions contain `irrelevant information`. This
is achieved by augmenting the GSM-IC dataset (Shi et al., 2023a), which is an augmentation of GSM8K specifically
designed to introduce irrelevant context (IC) to the question statement. These GSM-IC variants constitute to
approximately `one third` of TinyGSM.

The resulting synthetic dataset contains 12.3M problems (i.e. question-solution pairs) 2 with, based on the
original 7.4k training set questions and their IC variants. For each question in the GSM8K train set, the prompt
based on this question is shared across API calls, and the source of randomness comes entirely from the generation
process. To encourage diversity, we use `temperature sampling` and `specify in the prompt` to encourage the problem
variants to be `grammatically diverse` and `contain multiple steps`; the exact prompts are provided in Figure 3 and
in Appendix A.1.

![](img/tiny-gsm-01.jpg)

