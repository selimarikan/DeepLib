import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
from mxnet import autograd as ag


def GetMNIST():
    # Get mnist data dict
    mnist = mx.test_utils.get_mnist()
    return mnist


def GetMLPModel():
    # Auto inits with xavier
    net = nn.Sequential()
    with net.name_scope():
        net.add(nn.Dense(units=128, activation='relu'))
        net.add(nn.Dense(units=64, activation='relu'))
        net.add(nn.Dense(units=10))

    return net

def Train(ctx, model, numEpochs, trainData, trainer, lossFunc, metric):
    for i in range(numEpochs):
        # Reset iterator
        trainData.reset()

        for batch in trainData:
            # Separate and load to contexts (multi-GPU)
            data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
            label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
            outputs = []

            # scope with grad calcuation
            with ag.record():
                for x, y in zip(data, label):
                    z = model(x)
                    loss = lossFunc(z, y)
                    loss.backward()
                    outputs.append(z)
            
            metric.update(label, outputs)
            # to normalize the grad with 1/batchSize
            trainer.step(batch.data[0].shape[0])

        name, acc = metric.get()
        metric.reset()
        print('Training acc at epoch %d: %s=%f'%(i, name, acc))


def Predict(ctx, model, testData, metric):
    testData.reset()

    for batch in testData:
        data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
        outputs = []

        for x in data:
            outputs.append(model(x))

        metric.update(label, outputs)
    print('Test acc: %s=%f'%metric.get())
    assert metric.get()[1] > 0.94


if __name__ == '__main__':
    mx.random.seed(0)

    mnist = GetMNIST()
    model = GetMLPModel()
    GPUs = mx.test_utils.list_gpus()

    batchSize = 100
    trainData = mx.io.NDArrayIter(mnist['train_data'], mnist['train_label'], batch_size=batchSize, shuffle=True)
    testData = mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], batch_size=batchSize)

    ctx = [mx.gpu(0)] if GPUs else [mx.cpu(0), mx.cpu(1)]
    #ctx = mx.gpu(0)
    # Insight: Multiple elements in a context allow multi GPU + CPU or whatever
    print(ctx)
    model.initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
    trainer = gluon.Trainer(model.collect_params(), optimizer='sgd', optimizer_params={'learning_rate' : 0.02})

    epoch = 10

    accuracyMetric = mx.metric.Accuracy()
    softmaxCrossEntropy = gluon.loss.SoftmaxCrossEntropyLoss()

    Train(ctx, model, epoch, trainData, trainer, softmaxCrossEntropy, accuracyMetric)
    Predict(ctx, model, testData, accuracyMetric)
    

