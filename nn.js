/* jshint esversion: 8*/
//((value * (max - min)) + min)

function imgToPixelArray(img) {
  // image image, bitmap, or canvas
  let imgWidth;
  let imgHeight;
  let inputImg;

  if (
    img instanceof HTMLImageElement ||
    img instanceof HTMLCanvasElement ||
    img instanceof HTMLVideoElement ||
    img instanceof ImageData
  ) {
    inputImg = img;
  } else if (
    typeof img === "object" &&
    (img.elt instanceof HTMLImageElement ||
      img.elt instanceof HTMLCanvasElement ||
      img.elt instanceof HTMLVideoElement ||
      img.elt instanceof ImageData)
  ) {
    inputImg = img.elt; // Handle p5.js image
  } else if (
    typeof img === "object" &&
    img.canvas instanceof HTMLCanvasElement
  ) {
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

  const canvas = document.createElement("canvas");
  canvas.width = imgWidth;
  canvas.height = imgHeight;

  const ctx = canvas.getContext("2d");
  ctx.drawImage(inputImg, 0, 0, imgWidth, imgHeight);

  const imgData = ctx.getImageData(0, 0, imgWidth, imgHeight);

  return Array.from(imgData.data);
}

class NeuralNetwork {
  constructor(options) {
    if (options.inputs instanceof Array) {
      this.in = options.inputs.length;
      this.dataIn = options.inputs;
      this.dataOn = options.outputs ? options.outputs[0] : "label";
      this.on = 0;
      this.format = "named";
    } else {
      this.in = options.inputs;
      this.on = options.outputs;
      this.dataOn = "label";
      this.dataIn = [];
      for (let i = 0; i < this.in; i++) {
        this.dataIn.push(`x${i}`);
      }
      this.format = "numbered";
    }
    this.dataUrl = options.dataUrl ? options.dataUrl : "";
    this.hn = options.hidden ? options.hidden : 64;
    this.task = options.task;
    this.dataAdded = 0;
    this.learningRate = options.learningRate ? options.learningRate : 0.02;

    if (this.task === "imageClassification") {
      this.in = 1;
      this.dataOn = "label";
      this.imageWidth = this.dataIn[0];
      this.imageHeight = this.dataIn[1];
      this.imageChannels = this.dataIn[2];
      this.dataIn = ["image"];
    } else if (this.task === "transferLearning") {
      this.in = 1;
      this.dataOn = "label";
      this.imageWidth;
      this.imageHeight;
      this.imageChannels;
      this.dataIn = ["image"];
    }

    this.baseModelURL = "";

    this.model = this.createModel();
    this.trainData = [];

    this.metadata = {
      [this.dataOn]: {
        classes: [],
        keys: {},
      },
    };
    if (this.task !== "imageClassification") {
      for (let i = 0; i < this.in; i++) {
        this.metadata[this.dataIn[i]] = { min: 0, max: 0 };
      }
    }

    if (this.task === "imageClassification") {
      this.metadata.image = { min: 0, max: 255 };
    }

    if (this.task === "regression") {
      this.metadata[this.dataOn].options = { min: 0, max: 0 };
    }

    if (this.dataUrl !== "") {
      if (this.dataUrl.endsWith(".csv")) {
        (async () => {
          const response = await fetch(this.dataUrl);
          const text = await response.text();
          return text.split(/\n/).map((val) => val.split(/.*?\s*/));
        })().then((table) => {
          table.forEach((row, i) => {
            const inputs = {};
            this.dataIn.forEach((val, j) => {
              const index = table[0].indexOf(val);
              if (index > -1) {
                inputs[val] = row[index];
              }
            });
            const targets = {};
            const index = table[0].indexOf(this.dataOn);
            if (index > -1) {
              targets[this.dataOn] = row[index];
            }
          });
        });
      }
    }
  }

  createModel() {
    const model = tf.sequential();
    if (this.task === "classification") {
      const hidden = tf.layers.dense({
        inputShape: [this.in],
        units: this.hn,
        activation: "relu",
        useBias: true,
      });
      const output = tf.layers.dense({
        units: this.on > 0 ? this.on : 1,
        activation: "softmax",
        useBias: true,
      });
      model.add(hidden);
      model.add(output);

      if (this.task === "classification") {
        const optimizer = tf.train.sgd(this.learningRate);
        model.compile({
          optimizer,
          loss: "categoricalCrossentropy",
          metrics: ["accuracy"],
        });
      } else if (this.task === "regression") {
        model.add(
          tf.layers.dense({
            inputShape: [this.in],
            units: this.hn,
            activation: "relu",
            useBias: true,
          })
        );
        model.add(
          tf.layers.dense({
            units: 1,
            activation: "sigmoid",
            useBias: true,
          })
        );
        const optimizer = tf.train.adam(this.learningRate);
        model.compile({
          optimizer,
          loss: "meanSquaredError",
          metrics: ["accuracy"],
        });
      }
    } else if (this.task === "imageClassification") {
      model.add(
        tf.layers.conv2d({
          filters: 8,
          inputShape: [this.imageWidth, this.imageHeight, this.imageChannels],
          kernelSize: 5,
          kernelInitializer: "varianceScaling",
          activation: "relu",
        })
      );
      model.add(
        tf.layers.maxPooling2d({
          poolSize: [2, 2],
          strides: [2, 2],
        })
      );
      model.add(
        tf.layers.conv2d({
          filters: 16,
          kernelSize: 5,
          kernelInitializer: "varianceScaling",
          activation: "relu",
        })
      );
      model.add(
        tf.layers.maxPooling2d({
          poolSize: [2, 2],
          strides: [2, 2],
        })
      );
      model.add(tf.layers.flatten());
      model.add(
        tf.layers.dense({
          units: this.on > 0 ? this.on : 1,
          kernelInitializer: "varianceScaling",
          activation: "softmax",
        })
      );

      const optimizer = tf.train.sgd(this.learningRate);
      model.compile({
        optimizer,
        loss: "categoricalCrossentropy",
        metrics: ["accuracy"],
      });
    } else if (this.task === "transferLearning") {
      if (this.baseModelURL !== "") {
        const baseModel = tf.loadLayersModel(this.baseModelURL);

        console.log(baseModel.layers);

        baseModel.layers.pop();

        baseModel.layers.forEach((x, i) => {
          x.trainable = false;
        });

        baseModel.add(
          tf.layers.dense({
            units: this.on > 0 ? this.on : 1,
            kernelInitializer: "varianceScaling",
            activation: "softmax",
          })
        );
        return baseModel;
      }
    }
    return model;
  }

  load(modelUrl) {
    if (this.task === "transferLearning") {
      if (!/[\\/]model\.json$/.test(modelUrl)) {
        throw "The url you provided is not a proper tensorflow model!";
      }
      this.baseModelURL = modelUrl;
    }
  }

  query(in_arr) {
    return tf.tidy(() => {
      let inputs = [];
      if (this.task !== "imageClassification") {
        for (let i = 0; i < this.dataIn.length; i++) {
          let normalized;
          if (this.metadata[this.dataIn[i]]) {
            if (this.format === "named") {
              normalized = map(
                in_arr[this.dataIn[i]],
                this.metadata[this.dataIn[i]].min,
                this.metadata[this.dataIn[i]].max,
                0,
                1
              );
            } else if (this.format === "numbered") {
              const label = Object.keys(in_arr)[i];
              normalized = map(
                in_arr[label],
                this.metadata[this.dataIn[i]].min,
                this.metadata[this.dataIn[i]].max,
                0,
                1
              );
            }
          }
          inputs.push(normalized);
        }
      } else {
        const arr = [];
        if (in_arr instanceof Uint8Array) {
          in_arr.forEach((val) => arr.push(val));
        }
        const pixels =
          in_arr instanceof Uint8Array ? arr : imgToPixelArray(in_arr.image);
        inputs = pixels.map((val) => map(val, 0, 255, 0, 1));
      }

      const xs =
        this.task === "imageClassification"
          ? tf.tensor4d(inputs, [
              1,
              this.imageWidth,
              this.imageHeight,
              this.imageChannels,
            ])
          : tf.tensor2d([inputs]);
      // xs.print();
      // console.log(xs.shape);
      const ys = this.model.predict(xs).dataSync();
      //print(ys);
      if (this.dataIn && this.dataOn) {
        const outputs = [];
        for (let i = 0; i < this.metadata[this.dataOn].classes.length; i++) {
          outputs.push({
            label: this.metadata[this.dataOn].classes[i],
            confidence: ys[i],
          });
        }

        outputs.sort((a, b) => a.confidence - b.confidence);
        outputs.reverse();
        return outputs;
      } else {
        return ys;
      }
    });
  }

  async predict(in_obj) {
    return tf.tidy(() => {
      if (in_obj instanceof Object) {
        let inputs = [];
        for (let i = 0; i < this.dataIn.length; i++) {
          let normalized;
          if (this.metadata[this.dataIn[i]]) {
            normalized = map(
              in_obj[this.dataIn[i]],
              this.metadata[this.dataIn[i]].min,
              this.metadata[this.dataIn[i]].max,
              0,
              1
            );
          }
          inputs.push(normalized);
        }
        const xs = tf.tensor2d([inputs]);
        const ys = this.model.predict(xs).dataSync();
        print(ys);
        const { max, min } = this.metadata[this.dataOn].options;
        return;
        const val = map(unFormattedValue, 0, 1, min, max);

        return {
          value: val,
          label: this.dataOn,
        };
      }
    });
  }

  query2d(in_arr) {
    return tf.tidy(() => {
      const xs = tf.tensor2d(in_arr);
      const ys = this.model.predict(xs).dataSync();
      return ys;
    });
  }
  addData(in_arr, tar_arr) {
    if (Object.keys(tar_arr)[0] !== this.dataOn) {
      throw new Error(
        "Your labeled target does not equal the labeled target set by you."
      );
    }

    if (
      !(in_arr instanceof Uint8Array) &&
      Object.keys(in_arr).length !== this.in
    ) {
      throw new Error(
        "The amount of inputs you sent in do not match to the amount of inputs set in start."
      );
    }
    if (this.format == "named") {
      const inputs = {};
      const targets = {};

      if (this.task === "imageClassification") {
        if (in_arr.image || in_arr instanceof Uint8Array) {
          const arr = in_arr instanceof Uint8Array ? [] : null;
          if (in_arr instanceof Uint8Array) {
            in_arr.forEach((x, i) => {
              arr.push(x);
            });
          }
          const pixels = in_arr.image ? imgToPixelArray(in_arr.image) : arr;
          inputs.image = pixels;
          targets[this.dataOn] = tar_arr[this.dataOn];
          if (
            this.metadata[this.dataOn].classes.includes(tar_arr[this.dataOn])
          ) {
            const index = this.metadata[this.dataOn].classes.indexOf(
              tar_arr[this.dataOn]
            );
            this.metadata[this.dataOn].keys[tar_arr[this.dataOn]][index] = 1;
          } else {
            this.on++;
            this.metadata[this.dataOn].classes.push(tar_arr[this.dataOn]);
            this.metadata[this.dataOn].keys[tar_arr[this.dataOn]] = [];
            for (let i = 0; i < this.on - 1; i++) {
              this.metadata[this.dataOn].keys[tar_arr[this.dataOn]].push(0);
            }
            const keys = Object.keys(this.metadata[this.dataOn].keys);
            for (const name of keys) {
              this.metadata[this.dataOn].keys[name].push(0);
            }

            const index = this.metadata[this.dataOn].classes.indexOf(
              tar_arr[this.dataOn]
            );
            this.metadata[this.dataOn].keys[tar_arr[this.dataOn]][index] = 1;
            targets[this.dataOn] = tar_arr[this.dataOn];
          }
          this.trainData.push({ inputs, targets });
        }
      } else {
        for (let i = 0; i < this.in; i++) {
          //print(in_arr[this.dataIn[i]]);
          inputs[this.dataIn[i]] = in_arr[this.dataIn[i]];
          if (this.metadata[this.dataIn[i]]) {
            const item = this.metadata[this.dataIn[i]];
            if (this.dataAdded === 0) {
              item.min = in_arr[this.dataIn[i]];
            } else if (item.min > in_arr[this.dataIn[i]]) {
              item.min = in_arr[this.dataIn[i]];
            }

            if (this.task === "regression") {
              const num = tar_arr[this.dataOn];
              if (this.dataAdded === 0) {
                this.metadata[this.dataOn].options.min = num;
              } else if (num > this.metadata[this.dataOn].options.max) {
                this.metadata[this.dataOn].options.max = num;
              } else if (num < this.metadata[this.dataOn].options.min) {
                this.metadata[this.dataOn].options.min = num;
              }
            }

            if (in_arr[this.dataIn[i]] > item.max) {
              item.max = in_arr[this.dataIn[i]];
            }
          }
        }

        if (this.task !== "regression") {
          if (
            this.metadata[this.dataOn].classes.includes(tar_arr[this.dataOn])
          ) {
            const index = this.metadata[this.dataOn].classes.indexOf(
              tar_arr[this.dataOn]
            );
            this.metadata[this.dataOn].keys[tar_arr[this.dataOn]][index] = 1;
            targets[this.dataOn] = tar_arr[this.dataOn];
          } else {
            this.on++;
            this.metadata[this.dataOn].classes.push(tar_arr[this.dataOn]);
            this.metadata[this.dataOn].keys[tar_arr[this.dataOn]] = [];
            for (let i = 0; i < this.on - 1; i++) {
              this.metadata[this.dataOn].keys[tar_arr[this.dataOn]].push(0);
            }
            const keys = Object.keys(this.metadata[this.dataOn].keys);
            for (const name of keys) {
              this.metadata[this.dataOn].keys[name].push(0);
            }

            const index = this.metadata[this.dataOn].classes.indexOf(
              tar_arr[this.dataOn]
            );
            this.metadata[this.dataOn].keys[tar_arr[this.dataOn]][index] = 1;
            targets[this.dataOn] = tar_arr[this.dataOn];
          }
        }

        targets[this.dataOn] = tar_arr[this.dataOn];

        this.trainData.push({ inputs, targets });
      }
    } else if (this.format == "numbered") {
      const inputs = {};
      const targets = {};

      for (let i = 0; i < this.in; i++) {
        //print(in_arr[this.dataIn[i]]);
        const label = Object.keys(in_arr)[i];
        if (this.metadata[this.dataIn[i]]) {
          const item = this.metadata[this.dataIn[i]];
          if (this.dataAdded === 0) {
            item.min = in_arr[label];
          } else if (item.min > in_arr[label]) {
            item.min = in_arr[label];
          }

          if (this.task === "regression") {
            const num = tar_arr[this.dataOn];
            if (this.dataAdded === 0) {
              this.metadata[this.dataOn].options.min = num;
            } else if (num > this.metadata[this.dataOn].options.max) {
              this.metadata[this.dataOn].options.max = num;
            } else if (num < this.metadata[this.dataOn].options.min) {
              this.metadata[this.dataOn].options.min = num;
            }
          }

          if (in_arr[label] > item.max) {
            item.max = in_arr[label];
          }
        }
      }

      if (this.metadata[this.dataOn].classes.includes(tar_arr[this.dataOn])) {
        const index = this.metadata[this.dataOn].classes.indexOf(
          tar_arr[this.dataOn]
        );
        this.metadata[this.dataOn].keys[tar_arr[this.dataOn]][index] = 1;
        targets[this.dataOn] = tar_arr[this.dataOn];
      } else {
        if (this.metadata[this.dataOn].classes.length === this.on) {
          throw new Error(
            "You have provided more outputs than specified in your nn cofigurations."
          );
        }
        this.metadata[this.dataOn].classes.push(tar_arr[this.dataOn]);
        this.metadata[this.dataOn].keys[tar_arr[this.dataOn]] = [];
        for (let i = 0; i < this.on; i++) {
          this.metadata[this.dataOn].keys[tar_arr[this.dataOn]].push(0);
        }
        const index = this.metadata[this.dataOn].classes.indexOf(
          tar_arr[this.dataOn]
        );
        this.metadata[this.dataOn].keys[tar_arr[this.dataOn]][index] = 1;
        targets[this.dataOn] = tar_arr[this.dataOn];
      }

      Object.keys(in_arr).forEach((x, i) => {
        if (this.dataIn.includes(`x${i}`)) {
          inputs[`x${i}`] = in_arr[x];
        }
      });

      this.trainData.push({
        inputs,
        targets,
      });
    }
    this.dataAdded++;
    this.model = this.createModel();
  }

  normalizeData() {
    for (const data of this.trainData) {
      if (this.dataOn && this.dataIn) {
        for (let i = 0; i < this.dataIn.length; i++) {
          if (this.task !== "imageClassification") {
            data.inputs[this.dataIn[i]] = map(
              data.inputs[this.dataIn[i]],
              this.metadata[this.dataIn[i]].min,
              this.metadata[this.dataIn[i]].max,
              0,
              1
            );
          } else {
            data.inputs[this.dataIn[i]] = data.inputs[
              this.dataIn[i]
            ].map((val) => map(val, 0, 255, 0, 1));
          }
        }
        if (this.task == "regression") {
          data.targets[this.dataOn] = map(
            data.targets[this.dataOn],
            this.metadata[this.dataOn].options.min,
            this.metadata[this.dataOn].options.max,
            0,
            1
          );
        }
      }
    }
  }

  async loadData(data) {
    const response = await fetch(data);
    const json = await response.json();
    json.data.forEach((x, i) => {
      const { inputs, targets } = x;
      this.addData(inputs, targets);
    });
  }

  saveData(fileName = "model-data") {
    const trainData = {
      data: this.trainData,
    };
    saveJSON(trainData, fileName);
  }

  async train(options, call) {
    let xs = [];
    let ys = [];
    for (const data of this.trainData) {
      if (this.dataIn && this.dataOn) {
        const inputs = [];
        for (let i = 0; i < this.dataIn.length; i++) {
          if (this.task !== "imageClassification") {
            inputs.push(data.inputs[this.dataIn[i]]);
          } else {
            xs.push(data.inputs[this.dataIn[i]]);
          }
        }

        if (this.task !== "regression") {
          ys.push(this.metadata[this.dataOn].keys[data.targets[this.dataOn]]);
        } else {
          ys.push([data.targets[this.dataOn]]);
        }
        if (this.task !== "imageClassification") {
          xs.push(inputs);
        }
      } else {
        xs.push(data.inputs);
        ys.push(data.targets);
      }
    }

    if (this.task !== "imageClassification") {
      xs = tf.tensor2d(xs);
      ys = tf.tensor2d(ys);
    } else {
      xs = tf.tensor(xs, [
        xs.length,
        this.imageWidth,
        this.imageHeight,
        this.imageChannels,
      ]);
      ys = tf.tensor2d(ys);
    }

    const surface = tfvis
      .visor()
      .surface({ name: "DebugScreen", tab: "Debug", style: { height: 300 } });

    const history = [];

    const result = await this.model.fit(xs, ys, {
      batchSize: options.batchSize ? options.batchSize : 32,
      epochs: options.epochs ? options.epochs : 2,
      shuffle: true,
      callbacks: {
        onEpochEnd: (epoch, log) => {
          history.push(log);
          tfvis.show.history(surface, history, ["loss", "acc"]);
        },
      },
    });

    xs.dispose();
    ys.dispose();
    return result;
  }
}
