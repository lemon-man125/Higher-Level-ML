let brain;

const model = tf.sequential();

let video;

function setup() {
  createCanvas(400, 400);
  video = createCapture(VIDEO);
  video.size(64, 64);
  

  const options = {
    inputs: 2,
    outputs: 3,
    hidden: 16,
    task: 'classification'
  }
  brain = new NeuralNetwork(options);
  model.add(tf.layers.conv2d({
    filters: 64,
    inputShape: [64, 64, 4],
    kernelSize: [3, 3],
    activation: 'relu',
  }));
  model.add(tf.layers.maxPooling2d({
    poolSize: [2, 2]
  }))
  model.add(tf.layers.conv2d({
    filters: 64,
    inputShape: [64, 64, 4],
    kernelSize: [3, 3],
    activation: 'relu',
  }));
  model.add(tf.layers.maxPooling2d({
    poolSize: [2, 2]
  }))
  model.add(tf.layers.flatten({
    dataFormat: 'channelsLast'
  }));
  model.add(tf.layers.dropout({
    rate: 0.5
  }))
  model.add(tf.layers.dense({
    inputShape: [64, 64, 4],
    units: 512,
    activation: 'relu'
  }));
  model.add(tf.layers.dense({
    units: 512,
    activation: 'relu'
  }));
  model.add(tf.layers.dense({
    units: 3,
    activation: 'softmax'
  }));
}
function imgToPixelArray(img){
  // image image, bitmap, or canvas
  let imgWidth;
  let imgHeight;
  let inputImg;
 
  if (img instanceof HTMLImageElement ||
     img instanceof HTMLCanvasElement ||
     img instanceof HTMLVideoElement ||
     img instanceof ImageData) {
    inputImg = img;
  } else if (typeof img === 'object' &&
     (img.elt instanceof HTMLImageElement ||
       img.elt instanceof HTMLCanvasElement ||
       img.elt instanceof HTMLVideoElement ||
       img.elt instanceof ImageData)) {
 
    inputImg = img.elt; // Handle p5.js image
  } else if (typeof img === 'object' &&
     img.canvas instanceof HTMLCanvasElement) {
    inputImg = img.canvas; // Handle p5.js image
  } else {
    inputImg = img;
  }

 
  if (inputImg instanceof HTMLVideoElement) {
    // should be videoWidth, videoHeight?
    imgWidth = inputImg.width;
    imgHeight = inputImg.height;
  } else {
    imgWidth = inputImg.width;
    imgHeight = inputImg.height;
  }


  const canvas = document.createElement('canvas');
  canvas.width = imgWidth;
  canvas.height = imgHeight;


  const ctx = canvas.getContext('2d');
  ctx.drawImage(inputImg, 0, 0, imgWidth, imgHeight);

  const imgData = ctx.getImageData(0,0, imgWidth, imgHeight)
  return Array.from(imgData.data)
}

function keyPressed() {
  const img = imgToPixelArray(video);
}


function draw() {
  background(220);
}
