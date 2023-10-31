import * as tfvis from "@tensorflow/tfjs-vis";
import * as tf from "@tensorflow/tfjs";

// 获取谷歌模拟数据
async function getData() {
  const carsDataResponse = await fetch(
    "https://storage.googleapis.com/tfjs-tutorials/carsData.json"
  );
  const carsData = await carsDataResponse.json();
  const cleaned = carsData
    .map((car) => ({
      mpg: car.Miles_per_Gallon,
      horsepower: car.Horsepower,
    }))
    .filter((car) => car.mpg != null && car.horsepower != null);

  return cleaned;
}
// 创建tfjs模型
function createModel() {
  // 简易模型，不通用，仅支持线性参数
  const model = tf.sequential();
  // 添加一个简单的输入层级
  model.add(tf.layers.dense({ inputShape: [1], units: 1, useBias: true }));

  model.add(tf.layers.dense({ units: 50, activation: "sigmoid" }));
  // 添加输出层
  model.add(tf.layers.dense({ units: 1, useBias: true }));

  // 返回模型
  return model;
}

function convertToTensor(data) {
  return tf.tidy(() => {
    // 清洗打乱元数据
    tf.util.shuffle(data);
    // 将数据转化为张量
    const inputs = data.map((d) => d.horsepower);
    const labels = data.map((d) => d.mpg);

    const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
    const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

    // 使用最大 最小范围 将数据归一化再0 - 1的范围
    const inputMax = inputTensor.max();
    const inputMin = inputTensor.min();
    const labelMax = labelTensor.max();
    const labelMin = labelTensor.min();

    const normalizedInputs = inputTensor
      .sub(inputMin)
      .div(inputMax.sub(inputMin));
    const normalizedLabels = labelTensor
      .sub(labelMin)
      .div(labelMax.sub(labelMin));

    return {
      inputs: normalizedInputs,
      labels: normalizedLabels,
      inputMax,
      inputMin,
      labelMax,
      labelMin,
    };
  });
}

async function trainModel(model, inputs, labels) {
  // 准备训练模型
  model.compile({
    optimizer: tf.train.adam(), // 训练模型的更新算法，此处选择adam优化器
    loss: tf.losses.meanSquaredError,
    metrics: ["mse"],
  });

  const batchSize = 32;
  const epochs = 50;

  return await model.fit(inputs, labels, {
    batchSize,
    epochs,
    shuffle: true,
    callbacks: tfvis.show.fitCallbacks({ name: "训练模型" }, ["loss", "mse"], {
      height: 200,
      callbacks: ["onEpochEnd"],
    }),
  });
}

function testModel(model, inputData, normalizationdData) {
  const { inputMax, inputMin, labelMax, labelMin } = normalizationdData;
  const [xs, preds] = tf.tidy(() => {
    // 新生成100个样本
    const xs = tf.linspace(0, 1, 100);
    // 填充到模型预测，并规定样本的形状，size和深度
    const preds = model.predict(xs.reshape([100, 1]));

    const unNormXs = xs.mul(inputMax.sub(inputMin)).add(inputMin);

    const unNormPreds = preds.mul(labelMax.sub(labelMin)).add(labelMin);

    return [unNormXs.dataSync(), unNormPreds.dataSync()];
  });

  const predictedPoints = Array.from(xs).map((val, i) => {
    return { x: val, y: preds[i] };
  });

  const originalPoints = inputData.map((d) => ({
    x: d.horsepower,
    y: d.mpg,
  }));

  tfvis.render.scatterplot(
    { name: "模型预测和原始数据" },
    {
      values: [originalPoints, predictedPoints],
      series: ["原始数据", "预测数据"],
    },
    {
      xLabel: "马力",
      yLabel: "油耗",
      height: 320,
    }
  );
}

window.onload = async function () {
  const dataList = await getData();
  //   const data = {
  //     values: dataList.map((el) => ({
  //       x: el.horsepower,
  //       y: el.mpg,
  //     })),
  //     series: ["马力和每加仑油耗英里数"],
  //   };

  //   const surface = {
  //     name: "马力和每加仑油耗英里数",
  //     tab: "散点图表",
  //   };
  //   // 绘制散点图
  //   tfvis.render.scatterplot(surface, data, {
  //     xLable: "马力",
  //     yLabel: "每加仑油耗英里数",
  //     height: 400,
  //   });
  //   tfvis.show.modelSummary({ name: "modelSummary" }, model);

  // 创建模型
  const model = createModel();
  const tensorData = convertToTensor(dataList);
  const { inputs, labels } = tensorData;

  await trainModel(model, inputs, labels);

  console.log("training end");

  testModel(model, dataList, tensorData);
};
