/* jshint esversion: 8*/
//((value * (max - min)) + min)

class NeuralNetwork {
  constructor(options) {
    if (options.inputs instanceof Array && options.outputs instanceof Array) {
      this.in = options.inputs.length;
      this.hn = options.hidden;
      this.on = 0;
      this.dataIn = options.inputs;
      this.dataOn = options.outputs;
      this.task = options.task;
      this.dataAdded = 0;
      this.metadata = {
        [this.dataOn[0]]: {
          classes: [],
          keys: {}
        }
      }

      if (this.task === 'regression') {
        this.metadata[this.dataOn[0]].options = {min: 0, max: 0};
      }

      for (let i = 0; i < this.in; i++) {
        this.metadata[this.dataIn[i]] = {min: 0, max: 0};
      }
    } else {
      this.in = options.inputs;
      this.hn = options.hidden;
      this.on = options.outputs;
    }
    this.learningRate = 0.6;
    this.model = this.createModel();
    this.trainData = [];
  }
  
  createModel() {
    const model = tf.sequential();
    const hidden = tf.layers.dense({
      inputShape: [this.in],
      units: this.hn,
      activation: 'relu',
      useBias: true
    });
    const output = tf.layers.dense({
      units: this.on > 0 ? this.on : 1,
      activation: 'softmax',
      useBias: true
    });
    model.add(hidden);
    model.add(output);

    if (this.task === 'classification') {
      const optimizer = tf.train.sgd(this.learningRate);
      model.compile({
        optimizer,
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy'],
      })
    } else if (this.task === 'regression') {
      const optimizer = tf.train.adam(this.learningRate);
      model.compile({
        optimizer,
        loss: 'meanSquaredError',
        metrics: ['accuracy'],
      })
    }
    return model;
  }
  query(in_arr) {
    return tf.tidy(() => {
      let inputs = [];
      if (in_arr instanceof Array) {
        inputs = in_arr;
      } else {
        for (let i = 0; i < this.dataIn.length; i++) {
          let normalized;
          if (this.metadata[this.dataIn[i]]) {
            normalized = map(in_arr[this.dataIn[i]], this.metadata[this.dataIn[i]].min, this.metadata[this.dataIn[i]].max, 0, 1);
          }
          inputs.push(normalized); 
        }
      }
      const xs = tf.tensor2d([inputs]);
      const ys = this.model.predict(xs).dataSync();
       //print(ys);
      if (this.dataIn && this.dataOn) {
        const outputs = [];
        for (let i = 0; i < this.metadata[this.dataOn[0]].classes.length; i++) {
          outputs.push({
            label: this.metadata[this.dataOn[0]].classes[i],
            confidence: ys[i]
          })
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
    if (in_obj instanceof Object) {
      let inputs = [];
        for (let i = 0; i < this.dataIn.length; i++) {
          let normalized;
          if (this.metadata[this.dataIn[i]]) {
            normalized = map(in_obj[this.dataIn[i]], this.metadata[this.dataIn[i]].min, this.metadata[this.dataIn[i]].max, 0, 1);
          }
          inputs.push(normalized); 
      }
      const xs = tf.tensor2d([inputs]);
      const ys = await this.model.predict(xs).array();
      print(ys);
      const { max, min } = this.metadata[this.dataOn[0]].options;
      const unFormattedValue = ys[0][0];
      const val = map(unFormattedValue, 0, 1, min, max);

      return {
        value: val,
        label: this.dataOn[0],
      }
    }
  }

  query2d(in_arr) {
    return tf.tidy(() => {
      const xs = tf.tensor2d(in_arr);
      const ys = this.model.predict(xs).dataSync();
      return ys;
    });
  }
  addData(in_arr, tar_arr) {
    if (this.dataIn && this.dataOn) {
      const inputs = {};
      const targets = {};
      for (let i = 0; i < this.in; i++) {
        //print(in_arr[this.dataIn[i]]);
        inputs[this.dataIn[i]] = in_arr[this.dataIn[i]]
        if (this.metadata[this.dataIn[i]]) {
          const item = this.metadata[this.dataIn[i]];
          if (this.dataAdded === 0) {
            item.min = in_arr[this.dataIn[i]];
          } else if (item.min > in_arr[this.dataIn[i]]) {
            item.min = in_arr[this.dataIn[i]];
          }

          if (this.task === 'regression') {
            const num = tar_arr[this.dataOn[0]];
            if (this.dataAdded === 0) {
              this.metadata[this.dataOn[0]].options.min = num;
            } else if (num > this.metadata[this.dataOn[0]].options.max) {
              this.metadata[this.dataOn[0]].options.max = num;
            } else if (num < this.metadata[this.dataOn[0]].options.min) {
              this.metadata[this.dataOn[0]].options.min = num;
            }
          }

          if (in_arr[this.dataIn[i]] > item.max) {
            item.max = in_arr[this.dataIn[i]]
          }
        }
        this.dataAdded++;
      }

      
      if (this.metadata[this.dataOn[0]].classes.includes(tar_arr[this.dataOn[0]])) {
        const index = this.metadata[this.dataOn[0]].classes.indexOf(tar_arr[this.dataOn[0]]);
        this.metadata[this.dataOn[0]].keys[tar_arr[this.dataOn[0]]][index] = 1;
        targets[this.dataOn[0]] =  tar_arr[this.dataOn[0]];
      } else {
        this.on++;
        this.metadata[this.dataOn[0]].classes.push(tar_arr[this.dataOn[0]]);
        this.metadata[this.dataOn[0]].keys[tar_arr[this.dataOn[0]]] = [];
        for (let i = 0; i < this.on-1; i++) {
          this.metadata[this.dataOn[0]].keys[tar_arr[this.dataOn[0]]].push(0);
        }
        const keys = Object.keys(this.metadata[this.dataOn[0]].keys);
        for (const name of keys) {
          this.metadata[this.dataOn[0]].keys[name].push(0);
        }

        const index = this.metadata[this.dataOn[0]].classes.indexOf(tar_arr[this.dataOn[0]]);
        this.metadata[this.dataOn[0]].keys[tar_arr[this.dataOn[0]]][index] = 1;
        targets[this.dataOn[0]] = tar_arr[this.dataOn[0]];
      }

      
      this.trainData.push({ inputs, targets });
    } else {
      this.trainData.push({
        inputs: in_arr,
        targets: tar_arr
      })
    }
    this.model = this.createModel();
  }

  normalizeData() {
    for (const data of this.trainData) {
      if (this.dataOn && this.dataIn) {
        for (let i = 0; i < this.dataIn.length; i++) {
          data.inputs[this.dataIn[i]] = map(data.inputs[this.dataIn[i]], this.metadata[this.dataIn[i]].min, this.metadata[this.dataIn[i]].max, 0, 1);
        }
      }
    }
  }
  
  async loadData(data) {
    const response = await fetch(data);
    const json = await response.json();
    this.trainData = json.data;
    this.metadata = json.meta;
    for (let i = 0; i < this.metadata[this.dataOn[0]].classes.length; i++) {
      this.on++;
    }
  }
  
  saveData() {
    const trainData = {
      data: this.trainData,
      meta: this.metadata,
    }
    saveJSON(trainData, 'model-data');
  }
  
  async train(options, call) {
    let xs = [];
    let ys = [];
    for (const data of this.trainData) {
      if (this.dataIn && this.dataOn) {
        const inputs = [];
        for (let i = 0; i < this.dataIn.length; i++) {
          inputs.push(data.inputs[this.dataIn[i]]);
        }

        ys.push(this.metadata[this.dataOn[0]].keys[data.targets[this.dataOn[0]]]);      
        xs.push(inputs);
      } else {
        xs.push(data.inputs);
        ys.push(data.targets);
      }
    }


    xs = tf.tensor2d(xs);
    ys = tf.tensor2d(ys);
    
    const surface = tfvis.visor().surface({name: 'DebugScreen', tab: 'Debug', style:{height: 300}})

    const history = [];

    const result = await this.model.fit(xs, ys, {
      batchSize: options.batchSize ? options.batchSize : 32,
      epochs: options.epochs ? options.epochs : 2,
      shuffle: true,
      callbacks: {
        onEpochEnd: (epoch, log) => {
          history.push(log);
          tfvis.show.history(surface, history, ['loss', 'acc']);
        }
      }
    });

    xs.dispose();
    ys.dispose();
    return result
  }
}