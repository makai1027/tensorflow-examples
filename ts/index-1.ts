import { model, layers, input, tensor, train, losses } from "@tensorflow/tfjs";
import * as tfvis from "@tensorflow/tfjs-vis";
import { dataX, dataY } from "../data/data-1";

document.addEventListener("DOMContentLoaded", () => {
  const cause = tensor(dataX); // 初始化张量
  const result = tensor(dataY); // 初始化张量

  const X = input({ shape: [13] });
  const Y = layers.dense({ units: 1 }).apply(X);
  // @ts-ignore
  const myModel = model({ inputs: X, outputs: Y });

  // 定义精度函数类型,或者设置监控指标
  const metrics = ["loss", "val_loss"];
  // 定义容器
  const container = {
    name: "数字模型训练",
    tab: "数字模型",
    styles: { height: "800px" },
  };
  // 该方法为tf.model.fit返回两个callback，onBatchEnd、onEpochEnd
  const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);

  myModel.compile({
    optimizer: train.adam(),
    loss: losses.meanSquaredError,
    metrics: ["accuracy"], // 精度函数
  });

  tfvis.show.modelSummary(
    {
      name: "模型架构",
      tab: "模型",
    },
    myModel
  );

  myModel.fit(cause, result, {
    batchSize: 39,
    epochs: 10,
    callbacks: fitCallbacks,
  });
});
