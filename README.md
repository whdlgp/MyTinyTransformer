# MyTinyTransformer
It just my tiny transformer

## How to Train & Test
### Setup env
* run 'uv sync'
### Train
* Copy 'config_example.yaml' to 'config.yaml'
* run 'uv run train.py'
### Test
* run 'uv run tester.py'

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