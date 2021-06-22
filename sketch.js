let brain;

let trained = false;

let classLabel = "training...";

let video;

let addDataButton;

const counters = {};

let canvas;

function setup() {
  canvas = createCanvas(400, 400);
  video = createCapture(VIDEO);
  video.size(64, 64);
  video.hide();

  addDataButton = select(".addData");
  addDataButton.mousePressed(() => {
    const data = select("#classSelector").value();
    const num = parseInt(select("#numSelector").value());
    addExample(data, num);
  });

  trainButton = createButton("Train");
  trainButton.mousePressed(train);

  // const labels = ['Chris', 'Bed'];

  // addButton1 = createButton(`Add ${labels[0]}`);
  // addButton1.mousePressed(() => {
  //   if (counters[labels[0]]) {
  //     if (counters[labels[0]] >= 100) {
  //       return;
  //     }
  //     counters[labels[0]]++;
  //   } else {
  //     counters[labels[0]] = 1;
  //   }

  //   addExample(labels[0]);
  // });

  // addButton2 = createButton(`Add ${labels[1]}`);
  // addButton2.mousePressed(() => {
  //   if (counters[labels[1]]) {
  //     if (counters[labels[1]] >= 100) {
  //       return;
  //     }
  //     counters[labels[1]]++;
  //   } else {
  //     counters[labels[1]] = 1;
  //   }

  //   addExample(labels[1]);
  // });

  const options = {
    inputs: [64, 64, 4],
    task: "imageClassification",
  };
  brain = new NeuralNetwork(options);
}

function train() {
  brain.normalizeData();
  brain.train({ epochs: 50 }).then(() => {
    console.log("done!");
    trained = true;
    classifyVideo();
  });
}

function classifyVideo() {
  const inputs = { image: video };
  const results = brain.query(inputs);
  const { label, confidence } = results[0];
  const percent = `${nf(confidence * 100, 2, 2)}%`;
  classLabel = `${label}:\n${percent}`;

  setTimeout(classifyVideo, 10);
}

async function addExample(label, dataAmount) {
  const fps = frameRate();
  for (let i = 0; i < dataAmount; i++) {
    if (counters[label]) {
      if (counters[label] >= dataAmount) return;
      counters[label]++;
    } else {
      counters[label] = 1;
    }
    const inputs = { image: video };
    const target = { label };
    console.log(`adding ${target.label}`);
    brain.addData(inputs, target);
    await delay(dataAmount / fps / dataAmount);
  }
}

function delay(sec) {
  return new Promise((res, rej) => {
    if (typeof sec !== "number") {
      rej(new Error("The seconds you passed in is not a nunber"));
    } else {
      setTimeout(res, sec * 1000);
    }
  });
}

function draw() {
  background(220);
  push();
  translate(width, 0);
  scale(-1, 1);
  image(video, 0, 0, width, height);
  pop();

  if (!trained) {
    Object.keys(counters).forEach((x, i) => {
      const { length } = Object.keys(counters);
      fill(255);
      textSize(16);
      textAlign(CENTER, CENTER);
      text(counters[x], (width / length) * (i + 1) - width / 2, height / 2);
    });
  }

  if (trained) {
    fill(255);
    textSize(64);
    textAlign(CENTER, CENTER);
    text(classLabel, width / 2, height / 2);
  }
}
