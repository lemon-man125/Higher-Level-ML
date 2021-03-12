let brain;


let video;

let canvas;

function setup() {
  canvas = createCanvas(400, 400);
  video = createCapture(VIDEO);
  video.size(28, 28);
  video.hide();
  

  const options = {
    inputs: [28, 28, 4],
    outputs: ['image'],
    hidden: 16,
    task: 'imageClassification'
  }
  brain = new NeuralNetwork(options);

}

function keyPressed() {
  brain.addData({ image: video }, { label: key });
}

function draw() {
  background(220);
  translate(width, 0);
  scale(-1, 1);
  image(video, 0, 0, width, height);
}
