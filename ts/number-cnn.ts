import {
  browser,
  tidy,
  sequential,
  layers,
  train,
  Sequential,
} from "@tensorflow/tfjs";
import * as tfvis from "@tensorflow/tfjs-vis";
// @ts-ignore
import { MnistData } from "./data.ts";

async function showExamples(data) {
  // 创建tfjs-vis容器
  const surface = tfvis.visor().surface({ name: "输入示例", tab: "输入数据" });

  // 获取样本
  const examples = data.nextTestBatch(20);
  const numExamples = examples.xs.shape[0];

  // 通过画布绘制每一个示例
  for (let i = 0; i < numExamples; i++) {
    try {
      const imageTensor = tidy(() => {
        // Reshape the image to 28x28 px
        return examples.xs
          .slice([i, 0], [1, examples.xs.shape[1]])
          .reshape([28, 28, 1]);
      });

      const canvas = document.createElement("canvas");
      canvas.width = 28;
      canvas.height = 28;
      canvas.style.margin = "4px";
      await browser.toPixels(imageTensor, canvas);
      surface.drawArea.appendChild(canvas);

      imageTensor.dispose();
    } catch (error) {
      console.log(error);
    }
  }
}

function getModel() {
  const model = sequential();

  const IMAGE_WIDTH = 28;
  const IMAGE_HEIGHT = 28;
  const IMAGE_CHANNELS = 1;
  // 添加初始层, 设置卷积
  model.add(
    layers.conv2d({
      inputShape: [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS], // [row, column, depth] ,
      kernelSize: 5, // 卷积窗口尺寸，如果是一个数字，那么他将是一个正方形， 此处指的是5 * 5的方形
      filters: 8, // 输出的空间维度，是指卷积中滤波器数量
      strides: 1, // 各维度卷积步长，如果strides是一个数字，那么两个维度的步长相等
      activation: "relu", // 激活函数， relu是修正线性单元函数
      kernelInitializer: "varianceScaling", // 随机初始化模型权重的方法
    })
  );

  // 添加隐藏层，用于空间数据的最大集合操作
  model.add(layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }));
  // 重复另一个conv2d 和  maxPooling 堆栈，在卷积中使用了更多的过滤器
  model.add(
    layers.conv2d({
      kernelSize: 5, // 卷积窗口尺寸
      filters: 16, // 过滤器数量
      strides: 1, // 各维度步长
      activation: "relu",
      kernelInitializer: "varianceScaling",
    })
  );
  model.add(layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }));

  // 数据扁平化输入，该操作没有权重，仅仅作为数据处理
  model.add(layers.flatten());

  // 最终概率分布
  const NUM_OUTPUT_CLASSES = 10;
  // 添加密集层
  model.add(
    layers.dense({
      units: NUM_OUTPUT_CLASSES, // 输出空间维度
      kernelInitializer: "varianceScaling", // 使用方差缩放内核矩阵
      activation: "softmax", // 激活程序为软上限
    })
  );

  // 选择优化器、损失函数和精度指标，然后编译返回模型
  const optimizer = train.adam();
  model.compile({
    optimizer,
    loss: "categoricalCrossentropy", // 设置损失对象和函数名称，
    metrics: ["accuracy"], // 精度函数
  });

  return model;
}

// 训练模型
async function trainModel(model: Sequential, data) {
  // 定义精度函数类型,或者设置监控指标
  const metrics = ["loss", "val_loss", "acc", "val_acc"];
  // 定义容器
  const container = {
    name: "数字模型训练",
    tab: "数字模型",
    styles: { height: "800px" },
  };
  // 该方法为tf.model.fit返回两个callback，onBatchEnd、onEpochEnd
  const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);
  // 梯度更新样本的数量，用的model.fit，默认值为32
  const BATCH_SIZE = 512;
  // 训练样本梯度递进数量
  const TRAIN_DATA_SIZE = 5500;
  // 测试样本梯度递进数量
  const TEST_DATA_SIZE = 1000;

  const [trainXs, trainYs] = tidy(() => {
    const d = data.nextTrainBatch(TRAIN_DATA_SIZE);

    return [d.xs.reshape([TRAIN_DATA_SIZE, 28, 28, 1]), d.labels];
  });

  const [textXs, testYs] = tidy(() => {
    const d = data.nextTestBatch(TEST_DATA_SIZE);
    return [d.xs.reshape([TEST_DATA_SIZE, 28, 28, 1]), d.labels];
  });

  return model.fit(trainXs, trainYs, {
    batchSize: BATCH_SIZE,
    validationData: [textXs, testYs],
    epochs: 10,
    shuffle: true,
    callbacks: fitCallbacks,
  });
}

async function run() {
  const data = new MnistData();

  await data.load();

  await showExamples(data);

  const model = getModel();

  tfvis.show.modelSummary(
    {
      name: "模型架构",
      tab: "模型",
    },
    model
  );

  await trainModel(model, data);
}

document.addEventListener("DOMContentLoaded", run);
