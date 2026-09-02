# MyTinyTransformer
It just my tiny transformer

## How to Train & Test
### Setup env
* uv sync --extra cu130

### Troubleshooting: CUDA 13.0 not working
If PyTorch with CUDA 13.0 doesn't work, install CUDA 12.6 build.

* uv sync --extra cu126

### Train
* Copy 'config_example.yaml' to 'config.yaml'
* run 'uv run train.py'
### Test
* run 'uv run tester.py'
### Quantization
I used TorchAO's Weight-Only Quantization (`Int8WeightOnlyConfig`).  
* Because my model is too small, it's showing almost no speedup.
* Dramatically reduced weight file size.

#### Quantize and save
* run 'uv run quantize.py' for 
#### Test quantized model
* run 'uv run tester_quant.py'

## Train & Test results
### Basic imple.
<details> <summary>Training log</summary>

```
(mytinytransformer) PS E:\MyTinyTransformer> uv run train.py
device: cuda (NVIDIA GeForce RTX 3080 Ti, 12.0GB)
train: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 13941/13941 [07:20<00:00, 31.63it/s, loss=1.5198]
epoch 0 | train loss 1.8343 | val loss 1.5979
checkpoint saved: checkpoints/model.pt
train: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 13941/13941 [07:12<00:00, 32.24it/s, loss=1.4273]
epoch 1 | train loss 1.4793 | val loss 1.5121
checkpoint saved: checkpoints/model.pt
train: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 13941/13941 [07:20<00:00, 31.65it/s, loss=1.3032]
epoch 2 | train loss 1.3939 | val loss 1.4863
checkpoint saved: checkpoints/model.pt
train: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 13941/13941 [07:23<00:00, 31.41it/s, loss=1.2933]
epoch 3 | train loss 1.3468 | val loss 1.4678
checkpoint saved: checkpoints/model.pt
train: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 13941/13941 [07:12<00:00, 32.27it/s, loss=1.3192]
epoch 4 | train loss 1.3146 | val loss 1.4626
checkpoint saved: checkpoints/model.pt
train: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 13941/13941 [07:15<00:00, 32.01it/s, loss=1.2381]
epoch 5 | train loss 1.2903 | val loss 1.4602
checkpoint saved: checkpoints/model.pt
train: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 13941/13941 [07:08<00:00, 32.52it/s, loss=1.2867]
epoch 6 | train loss 1.2706 | val loss 1.4591
checkpoint saved: checkpoints/model.pt
train: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 13941/13941 [07:12<00:00, 32.26it/s, loss=1.2496]
epoch 7 | train loss 1.2543 | val loss 1.4651
checkpoint saved: checkpoints/model.pt
train: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 13941/13941 [07:16<00:00, 31.90it/s, loss=1.2673]
epoch 8 | train loss 1.2404 | val loss 1.4684
checkpoint saved: checkpoints/model.pt
train: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 13941/13941 [07:10<00:00, 32.40it/s, loss=1.2447]
epoch 9 | train loss 1.2282 | val loss 1.4699
checkpoint saved: checkpoints/model.pt
training finished in 1:15:03
```

</details>

<details> <summary>Test log</summary>

```
(mytinytransformer) PS E:\MyTinyTransformer> uv run tester.py
device: cuda (NVIDIA GeForce RTX 3080 Ti, 12.0GB)
test loss: 1.7678 | perplexity: 5.86
chat mode, type a prompt (or 'exit' to quit)
> hello. my friend.
hello. my friend.

KING EDWARD IV:
Now fare your lady day of our worship:
So so told my heart misfortunes to the ose.

GLOUCESTER:
Curse him at his son of dead of their house,
For I would say 'twere the father was ne'
> Adam:
Adam:
Ay, for hear the Lady Bona; and, though groans with
in what thou livest me part consider, and he is,
Hath through thy father from season their vault,
Showing may I see they to see him in his part
Sha
> Dear my princess.
Dear my princess.

LUCIO:
How methinks, my lord, my lord.

PAULINA:
Cousin, lime to a confirm, as the father,
When my swords are beg answer to be of me.

TYRREL:
I beseech you, brother, which you do defend my sin?

BU
> q
quick, for our respected
What is at her bound you will swear your continue,
That it to the noble secret of the fair doom?

LADY ANNE:
Tush, that we prove to me?

GLOUCESTER:
Now, by your hath well marr
```

</details>

### Batched MHA Implementation
Achieved ~1.9x speedup (1:15:03 → 0:39:26, total training).

<details> <summary>Training log</summary>

```
(mytinytransformer) PS E:\MyTinyTransformer> uv run train.py 
device: cuda (NVIDIA GeForce RTX 3080 Ti, 12.0GB)
train: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 13941/13941 [03:46<00:00, 61.49it/s, loss=1.6035]
epoch 0 | train loss 1.8305 | val loss 1.5924
checkpoint saved: checkpoints/model.pt
train: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 13941/13941 [03:45<00:00, 61.71it/s, loss=1.4220]
epoch 1 | train loss 1.4789 | val loss 1.5081
checkpoint saved: checkpoints/model.pt
train: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 13941/13941 [03:42<00:00, 62.67it/s, loss=1.3877]
epoch 2 | train loss 1.3989 | val loss 1.4791
checkpoint saved: checkpoints/model.pt
train: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 13941/13941 [03:40<00:00, 63.28it/s, loss=1.3316]
epoch 3 | train loss 1.3492 | val loss 1.4636
checkpoint saved: checkpoints/model.pt
train: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 13941/13941 [03:46<00:00, 61.42it/s, loss=1.3455]
epoch 4 | train loss 1.3149 | val loss 1.4554
checkpoint saved: checkpoints/model.pt
train: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 13941/13941 [03:46<00:00, 61.46it/s, loss=1.1955]
epoch 5 | train loss 1.2902 | val loss 1.4540
checkpoint saved: checkpoints/model.pt
train: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 13941/13941 [03:47<00:00, 61.29it/s, loss=1.2645]
epoch 6 | train loss 1.2706 | val loss 1.4532
checkpoint saved: checkpoints/model.pt
train: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 13941/13941 [03:49<00:00, 60.63it/s, loss=1.1969]
epoch 7 | train loss 1.2543 | val loss 1.4588
checkpoint saved: checkpoints/model.pt
train: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 13941/13941 [03:48<00:00, 61.13it/s, loss=1.2651]
epoch 8 | train loss 1.2404 | val loss 1.4580
checkpoint saved: checkpoints/model.pt
train: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 13941/13941 [03:51<00:00, 60.16it/s, loss=1.2403]
epoch 9 | train loss 1.2281 | val loss 1.4628
checkpoint saved: checkpoints/model.pt
training finished in 0:39:26
```

</details>

<details> <summary>Test log</summary>

```
(mytinytransformer) PS E:\MyTinyTransformer> uv run tester.py
device: cuda (NVIDIA GeForce RTX 3080 Ti, 12.0GB)
test loss: 1.7551 | perplexity: 5.78
chat mode, type a prompt (or 'exit' to quit)
> hello. my friend.
hello. my friend.

Nurse:
No, madam, yours, my friends is no need unto:
Lest the censure can and break to earth,
The fine of England, chasting the dead off,
The market wherein the land of her contempt,
Which is this b
> Adam:
Adam:
That looks yet heaven the east hearts of your trick
Against the more than will shake you.

HASTINGS:
What fear the royal kneels
In the cause of his soul to have sons are prove all,
For cannot do not
> Dear my princess.
Dear my princess. This is the prince
And do welcomes by the giance, thou art too late
God and deserves my best he wear to shame;
And, if thou must deliver'd my life,
And shall be since the hours of the child,
In the w
> q
qlast would be loyal be strong
That their brothers. Forbid a home!
Go to: fear not be gone, God he say, spirit the
And so do meet work, sir.

MERCUTIO:
You can not say should have expose it
with lost:
>
```

</details>

### MHA with KV Cache Implementation

<details> <summary>Training log</summary>

```
(mytinytransformer) PS E:\MyTinyTransformer> uv run train.py  
device: cuda (NVIDIA GeForce RTX 3080 Ti, 12.0GB)                                                                                                                               
train: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 13941/13941 [03:44<00:00, 62.00it/s, loss=1.5136]
epoch 0 | train loss 1.8323 | val loss 1.5915
checkpoint saved: checkpoints/model.pt
train: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 13941/13941 [03:43<00:00, 62.31it/s, loss=1.4016]
epoch 1 | train loss 1.4740 | val loss 1.5062
checkpoint saved: checkpoints/model.pt
train: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 13941/13941 [03:43<00:00, 62.36it/s, loss=1.4051]
epoch 2 | train loss 1.3900 | val loss 1.4767
checkpoint saved: checkpoints/model.pt
train: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 13941/13941 [03:43<00:00, 62.45it/s, loss=1.3373]
epoch 3 | train loss 1.3408 | val loss 1.4594
checkpoint saved: checkpoints/model.pt
train: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 13941/13941 [03:47<00:00, 61.40it/s, loss=1.2533]
epoch 4 | train loss 1.3089 | val loss 1.4546
checkpoint saved: checkpoints/model.pt
train: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 13941/13941 [03:43<00:00, 62.35it/s, loss=1.2913]
epoch 5 | train loss 1.2852 | val loss 1.4562
checkpoint saved: checkpoints/model.pt
train: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 13941/13941 [03:43<00:00, 62.36it/s, loss=1.3077]
epoch 6 | train loss 1.2661 | val loss 1.4543
checkpoint saved: checkpoints/model.pt
train: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 13941/13941 [03:43<00:00, 62.48it/s, loss=1.2822]
epoch 7 | train loss 1.2501 | val loss 1.4623
checkpoint saved: checkpoints/model.pt
train: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 13941/13941 [03:44<00:00, 62.23it/s, loss=1.2524]
epoch 8 | train loss 1.2365 | val loss 1.4625
checkpoint saved: checkpoints/model.pt
train: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 13941/13941 [03:43<00:00, 62.42it/s, loss=1.1832]
epoch 9 | train loss 1.2248 | val loss 1.4658
checkpoint saved: checkpoints/model.pt
training finished in 0:38:56
```

</details>

<details> <summary>Test log</summary>

```
(mytinytransformer) PS E:\MyTinyTransformer> uv run tester.py
device: cuda (NVIDIA GeForce RTX 3080 Ti, 12.0GB)
test loss: 1.7649 | perplexity: 5.84
chat mode, type a prompt (or 'exit' to quit)
> hello. my friend.
hello. my friend.

GLOUCESTER:
And that all art left thee from the queen,
Which with all complose my feeming strives,
Being once
> Adam:
Adam:
He is no better bed foul there woe!

LADY ANNE:
Are you be gone.

KING RICHARD II:
For this we bethink me for a face?

QUE
> Dear my princess.
Dear my princess.

QUEEN MARGARET:
Romeo! How shall do speak the prince,
Thy side my earth, should thou come to the third;
And w
> q
queen
Yet a worst in the life lion of this death;
And would we go follow the ladies of more ears!
Fie, my sovereign, not should
>
```

</details>


### Quantization Test

Achieved an 87.9% size reduction(9.74MB → 1.17MB).

<details> <summary>Test log</summary>

```
(mytinytransformer) PS E:\MyTinyTransformer> uv run python -X utf8 .\tester_quant.py
W0902 23:28:46.863000 12052 .venv\Lib\site-packages\torch\utils\_pytree.py:630] <enum 'KernelPreference'> is an Enum subclass and is now natively supported by torch.compile as an opaque value type. Calling register_constant() on Enum subclasses is deprecated and will be an error in a future release.
device: cuda (NVIDIA GeForce RTX 3080 Ti, 12.0GB)
test loss: 1.7553 | perplexity: 5.79
chat mode, type a prompt (or 'exit' to quit)
> hello. my friend.
hello. my friend.

KING RICHARD IIII:
Why, then, I desire the consul?

QUEEN ELIZABETH:
I can tell my tent, or no; for I have wo
> Adam:
Adam:
Go, that thou with thy best thee hence to thee
Your will speak with it the house of you.

SICINIUS:
How! Warwick that's th
> Dear my princess.
Dear my princess.

CLIFFORD:
I thank your gentleman's straight three,
The excuse have fall of business!

ROMEO:
Your fairery, my
> q
quarrel, and be more revenge
Be not the way right and not die.

GLOUCESTER:
Who shall be your grace up whether I dare thee,
And
>
```

</details>