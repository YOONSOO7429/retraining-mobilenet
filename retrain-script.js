/* 이 코드는 TensorFlow.js를 사용하여 웹캠으로 "바위, 가위, 보" 게임을 플레이하는 인공지능 모델을 훈련하고 예측하는 것입니다. */

let mobilenet;
let model;
const webcam = new Webcam(document.getElementById("wc"));
const dataset = new RPSDataset();
let rockSamples = 0;
let paperSamples = 0;
let scissorsSamples = 0;
let isPredicting;

// webcam image의 truncated mobilenet 출력을 training example로 추가하는 함수
function handleButton(elem) {
  // 버튼 클릭에 따라 가위, 바위, 보 sample의 수를 증가하고 화면에 표시
  switch (elem.id) {
    case "0":
      rockSamples++;
      document.getElementById("rockSamples").innerText =
        "바위 sample 수: " + rockSamples;
      break;
    case "1":
      paperSamples++;
      document.getElementById("paperSamples").innerText =
        "보 sample 수: " + paperSamples;
      break;
    case "2":
      scissorsSamples++;
      document.getElementById("scissorsSamples").innerText =
        "가위 sample 수: " + scissorsSamples;
      break;
  }
  // 선택된 label(가위, 바위, 보)을 정수로 변환, image를 캡처하고 example을 추가
  label = parseInt(elem.id);
  const img = webcam.capture();
  // 'dataset'에 예시를 추가, mobilenet 모델이 'img'를 입력 받아 예측한 결과와 그에 해당하는 'label'의 쌍으로 구성
  dataset.addExample(mobilenet.predict(img), label);
}

// model train
async function train() {
  // 'label' one-hot-encoding
  dataset.ys = null;
  dataset.encodeLabels(3);

  // 새로운 신경망 분류기 생성
  model = tf.sequential({
    layers: [
      tf.layers.flatten({
        inputShape: mobilenet.outputs[0].shape.slice(1),
      }),
      tf.layers.dense({
        units: 100,
        activation: "relu",
      }),
      tf.layers.dense({
        units: 3,
        activation: "softmax",
      }),
    ],
  });
  model.summary();

  // model compile 및 train
  const optimizer = tf.train.adam(0.0001);
  model.compile({ optimizer: optimizer, loss: "categoricalCrossentropy" });

  await model.fit(dataset.xs, dataset.ys, {
    epochs: 10,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        console.log("LOSS: ", logs.loss.toFixed(5));
      },
    },
  });
  console.log("train.. ends..");
}

// prediction loop
async function predict() {
  while (isPredicting) {
    // webcam image를 이용하여 prediction
    const predictedClass = tf.tidy(() => {
      const img = webcam.capture();
      const activation = mobilenet.predict(img);
      const predictions = model.predict(activation);
      /*예측 결과인 확률 벡터를 1차원 텐서로 변환(flatten)하고,
    가장 확률이 높은 클래스의 인덱스를 반환(argMax)-가위,바위,보 */
      return predictions.flatten().argMax();
    });

    // prediction 결과 display
    const classId = (await predictedClass.data())[0];
    let predictionText = "";
    switch (classId) {
      case 0:
        predictionText = "바위입니다.";
        break;
      case 1:
        predictionText = "보입니다.";
        break;
      case 2:
        predictionText = "가위입니다.";
        break;
    }

    // 예측된 텍스트를 화면에 출력
    document.getElementById("prediction").innerText = predictionText;

    predictedClass.dispose();
    // nestFrame waiting
    await tf.nextFrame();
  }
}

// 훈련을 시작
function doTraining() {
  console.log("train... begins");
  train();
}

// 예측을 시작
function startPredicting() {
  isPredicting = true;
  predict();
}

// 예측을 중단
function stopPredicting() {
  isPredicting = false;
}

// webcam 초기화, truncated mobilenet load
async function init() {
  // webcam 초기화,
  await webcam.setup();
  // truncated mobilenet 모델을 불러옵니다.
  mobilenet = await tf.loadLayersModel(
    "https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json"
  );
  // mobilenet 모델에서 'conv_pw_13_relu' 레이어를 가져옵니다.
  const layer = mobilenet.getLayer("conv_pw_13_relu");

  // truncated mobilenet model 생성
  mobilenet = tf.model({ inputs: mobilenet.inputs, outputs: layer.output });
}
// 초기화 함수를 호출
init();
