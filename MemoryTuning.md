- https://arxiv.org/abs/2406.17642

LLMs augmented with a massive Mixture of Memory Experts (MoME) can easily memorize large datasets

simple neural networks trained to predict the next token hallucinate when the training loss 
is above a threshold as it usually does in practice when training on internet scale data

design a model for removing hallucinations that stores facts in a massive mixture of 
millions of memory experts that are retrieved dynamically

![](img/memory-tuning-00.jpg)

![](img/memory-tuning-01.jpg)

## Insights

- LLM có thể dễ dàng nhớ được random labels mà **không làm tăng tỉ lệ lỗi tổng quan hóa**
  Nó có đủ năng lực để nhớ facts một cách chính xác kể cả dữ liệu huấn luyện bị nhiễu hoặc random.

- Điểm tổng quan hóa không phản ánh LLMs có hallu hay không.
  Một LLM rất hallu với 1 LLM không hallu có thể có điểm số MMLU là như nhau.

- Sẽ tốn computing để remove hallu

![](img/memory-tuning-02.jpg)

The massive MoME is designed to cut down on the amount of computation required to memorize
facts. This is accomplished by the following training algorithm:

1. `For a given question`, select a subset of experts, e.g. `32 out of the array of one million`.

2. Freeze the weights of the backbone network and the cross attention used to select the expert.

3. Take gradient descent steps until the loss is reduced sufficiently to memorize the fact.

One problem is that the `same expert may be selected multiple times` for different facts during training. This can be mitigated by first `training the cross attention selection mechanism during generalization training, e.g. for one epoch`, followed by freezing its weights. This results in the same expert being selected for each fact on each training step.

- - -

Một vấn đề là `cùng một chuyên gia có thể được chọn nhiều lần` cho các sự kiện khác nhau trong quá trình huấn luyện. Điều này có thể được giảm thiểu bằng cách `huấn luyện cơ chế chọn lựa cross attention trong quá trình huấn luyện tổng quát, ví dụ trong một epoch`, sau đó cố định các trọng số của nó. Kết quả là cùng một chuyên gia sẽ được chọn cho mỗi sự kiện trong từng bước huấn luyện. (chưa hiểu lắm)

![](img/memory-tuning-03.jpg)
