import { data } from "@tensorflow/tfjs";
import * as mobileNet from "@tensorflow-models/mobilenet";
import * as knnClassifier from "@tensorflow-models/knn-classifier";

async function getImageModel() {
  console.log("加载图片识别模型");
  try {
    const imageModel = await mobileNet.load();
    return imageModel;
  } catch (error) {
    console.log("getImageModel error");
    return null;
  }
}

function previewFileImage(file, model) {
  console.log(file);
  const listItem = document.createElement("li");
  const listBox = document.getElementById("image-box");
  const image = document.createElement("img");
  const span = document.createElement("span");
  image.src = URL.createObjectURL(file);

  image.onload = async function () {
    listItem.appendChild(image);
    listBox?.append(listItem);

    const result = await model.classify(image);
    if (result && Array.isArray(result) && result.length) {
      let str = "";

      result.forEach((el) => {
        str += `识别结果：${el.className}，可能性为: ${
          el.probability * 100
        }%;&#10;`;
      });

      span.innerHTML = str;
      listItem.appendChild(span);
    }
  };
}
const classifier = knnClassifier.create();

document.addEventListener("DOMContentLoaded", async () => {
  const model = await getImageModel();
  if (!model) return;
  document.getElementById("fileInput")?.addEventListener("change", (value) => {
    if (!value) return;
    // @ts-ignore
    previewFileImage(value.target.files[0], model);
  });
  const videoEle = document.getElementById("webcam") as HTMLVideoElement;
  if (!videoEle) return;
  const webcam = await data.webcam(videoEle);

  async function addClass(classId) {
    console.log(classId, "1111111");
    // 从视频捕捉一张截图，作为分类
    const img = await webcam.capture();

    const activation = model?.infer(img, true);

    classifier.addExample(activation, classId);
    // 销毁张量，用以节省内存
    img.dispose();
  }

  document
    .getElementById("class-a")
    ?.addEventListener("click", () => addClass("a"));
  document
    .getElementById("class-b")
    ?.addEventListener("click", () => addClass("b"));
  document
    .getElementById("class-c")
    ?.addEventListener("click", () => addClass("c"));
  const videoResult = document.getElementById("video-result");
  // while (true) {
  //   if (classifier.getNumClasses() > 0) {
  //     const img = await webcam.capture();

  //     const activation = model.infer(img, "conv_preds");

  //     const result = await classifier.predictClass(activation);

  //     const classes = ["A", "B", "C"];
  //     console.log(result, "!!!!!!");
  //     // if (result && Array.isArray(result) && result.length) {
  //     //   let str = "";

  //     //   result.forEach((el) => {
  //     //     str += `识别结果：${el.className}，可能性为: ${
  //     //       el.probability * 100
  //     //     }%;\n;`;
  //     //   });

  //     //   videoResult.innerHTML = str;
  //     // }
  //   }
  // }
  // setInterval(async () => {
  //   const image = await webcam.capture();
  //   const result = await model.classify(image);

  //   if (!videoResult) return;
  //   if (result && Array.isArray(result) && result.length) {
  //     let str = "";

  //     result.forEach((el) => {
  //       str += `识别结果：${el.className}，可能性为: ${
  //         el.probability * 100
  //       }%;\n;`;
  //     });

  //     videoResult.innerHTML = str;
  //   }
  // }, 10000);
});
