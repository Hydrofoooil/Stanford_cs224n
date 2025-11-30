## 一些实际训练时需要修改的地方：

### a3:

根据我们之前的排查，为了让你的模型真正按照你想要的参数（512维）训练，并解决 PyTorch 版本兼容性问题，你需要对文件做以下修改的总结：

#### 1. 修改 `run.py` 

##### A. 修复“参数被写死”的 Bug

问题： 你的代码第 119-122 行强制将模型尺寸设为了 1024/768，导致命令行参数失效。

操作： 注释掉写死的部分，恢复读取 args 的部分，使得在训练时能够自己调整模型尺寸提高模型性能。

- 修改前 (第 115-122 行)：

  ```python
  # model = NMT(embed_size=int(args['--embed-size']),                                 # EDIT: 4X EMBED AND HIDDEN SIZES 
  #             hidden_size=int(args['--hidden-size']),
  #             dropout_rate=float(args['--dropout']),
  #             vocab=vocab)
  
  model = NMT(embed_size=1024,
              hidden_size=768,
              dropout_rate=float(args['--dropout']),
              vocab=vocab)
  ```

- 修改后：

  ```python
  # 恢复这一段（读取命令行参数）：
  model = NMT(embed_size=int(args['--embed-size']),
              hidden_size=int(args['--hidden-size']),
              dropout_rate=float(args['--dropout']),
              vocab=vocab)
  
  # 注释掉这一段（写死的参数）：
  # model = NMT(embed_size=1024,
  #             hidden_size=768,
  #             ...
  ```

##### B. 修复 PyTorch 2.6 兼容性 (Pickle Error)

问题： 训练过程中如果触发 patience 需要加载旧模型时，会因为 PyTorch 2.6 的安全策略报错。

操作： 在 train 函数中加载模型的地方添加 weights_only=False。

- 位置： 约第 281 行。

- 修改后：

  ```py
  params = torch.load(model_save_path, map_location=lambda storage, loc: storage, weights_only=False)
  ```

同时在 `run.sh`中也需要修改 `nmt_model.py`。

- 操作： 在 `nmt_model.py` 的 `load` 函数中（约第 519 行），同样加上 `weights_only=False`：

  ```py
  params = torch.load(model_path, map_location=lambda storage, loc: storage, weights_only=False)
  ```

#### 2. 修改 `run.sh` 

详见`my_training`文件夹下的每次训练配置，从这些训练的对应输出和tensorboard log可以看出模型表现的提升。因为在实际训练中发现作业里原本的配置并不能训出非常好的表现，故自己调整了部分参数。

### a4：

#### 1.修改 `dataset.py` 

在`dataset.py`的`charCorruptionDatase`中需要将从训练集中读取的`doc`随机截断，理应是截断后长度（`trunc_len`）和截断起点(`start_index`)都随机。但是经过实际尝试，若截断起点也随机，训练出来的模型将表现很差，在`dev`集上的准确率只有5%左右，原因应该是训练的语料库不够大。所以截断位置还是设成0比较合适，即保留从头开始长度为`trunc_len`的字符串。这样模型看到的绝大多数训练样本都是从**人物姓名开始**的。这无意中简化了模型的学习任务，模型总是看到相似的句式结构（“某某某是...”、“某某某出生于...”）。它很快就能学会这种固定模式，loss会迅速下降。
#### 2.修改 `run.py` 

建议修改pretrain时的`max_epoch`。作业里原先设置的650感觉不太够，改成1000轮训出来的结果会好很多。同时别忘了修改`run.py`里用于设置lr衰减的对应参数：

```python
tconf = trainer.TrainerConfig(
        max_epochs=1000,  # 650 original
        batch_size=128,
        learning_rate=args.pretrain_lr,
        lr_decay=True,
        warmup_tokens=512*20,
        final_tokens=1000 * len(pretrain_dataset)*block_size, # 650 * len(pretrain_dataset)*block_size original
        num_workers=4,
        writer=writer
    )
```