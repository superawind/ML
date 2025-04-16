基于trl 的 RewardTrainer 进行debug ，观察数据形状

本质和 rerank 的训练方式相似，二者都是基于AutoModelForSequenceClassifier ,设置num_label = 1 得到预测值，区别是二者计算loss 的方式不同

