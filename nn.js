/* jshint esversion: 8*/
//((value * (max - min)) + min)

class NeuralNetwork {
  constructor(options) {
    if (options.inputs instanceof Array && options.outputs instanceof Array) {
      this.in = options.inputs.length;
      this.hn = options.hidden;
      this.dataIn = options.inputs;
      this.dataOn = options.outputs[0];
      this.on = 0;
      this.format = 'named';
    } else {
      this.in = options.inputs;
      this.hn = options.hidden;
      this.on = options.outputs;
      this.dataOn = 'label';
      this.dataIn = [];
      for (let i = 0; i < this.in; i++) {
        this.dataIn.push(`x${i}`);
      }
      this.format = 'numbered';
    }
    this.task = options.task;
    this.dataAdded = 0;
    this.learningRate = 0.6;
    this.model = this.createModel();
    this.trainData = [];
    this.metadata = {
      [this.dataOn]: {
        classes: [],
        keys: {}
      }
    }
    for (let i = 0; i < this.in; i++) {
      this.metadata[this.dataIn[i]] = {min: 0, max: 0};
    }

    if (this.task === 'regression') {
      this.metadata[this.dataOn].options = {min: 0, max: 0};
    }
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
            if (this.format === 'named') {
              normalized = map(in_arr[this.dataIn[i]], this.metadata[this.dataIn[i]].min, this.metadata[this.dataIn[i]].max, 0, 1);
            } else if (this.format === 'numbered') {
              const label = Object.keys(in_arr)[i];
              normalized = map(in_arr[label], this.metadata[this.dataIn[i]].min, this.metadata[this.dataIn[i]].max, 0, 1);
            }
          }
          inputs.push(normalized); 
        }
      }
      const xs = tf.tensor2d([inputs]);
      const ys = this.model.predict(xs).dataSync();
       //print(ys);
      if (this.dataIn && this.dataOn) {
        const outputs = [];
        for (let i = 0; i < this.metadata[this.dataOn].classes.length; i++) {
          outputs.push({
            label: this.metadata[this.dataOn].classes[i],
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
      const { max, min } = this.metadata[this.dataOn].options;
      const unFormattedValue = ys[0][0];
      const val = map(unFormattedValue, 0, 1, min, max);

      return {
        value: val,
        label: this.dataOn,
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
    if (Object.keys(tar_arr)[0] !== this.dataOn) {
      throw new Error('Your labeled target does not equal the labeled target set by you.');
    }
    
    if (Object.keys(in_arr).length !== this.in) {
      throw new Error('The amount of inputs you sent in do not match to the amount of inputs set in start.');
    } 
    if (this.format == 'named') {
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
            item.max = in_arr[this.dataIn[i]]
          }
        }
      }

      
      if (this.metadata[this.dataOn].classes.includes(tar_arr[this.dataOn])) {
        const index = this.metadata[this.dataOn].classes.indexOf(tar_arr[this.dataOn]);
        this.metadata[this.dataOn].keys[tar_arr[this.dataOn]][index] = 1;
        targets[this.dataOn] =  tar_arr[this.dataOn];
      } else {
        this.on++;
        this.metadata[this.dataOn].classes.push(tar_arr[this.dataOn]);
        this.metadata[this.dataOn].keys[tar_arr[this.dataOn]] = [];
        for (let i = 0; i < this.on-1; i++) {
          this.metadata[this.dataOn].keys[tar_arr[this.dataOn]].push(0);
        }
        const keys = Object.keys(this.metadata[this.dataOn].keys);
        for (const name of keys) {
          this.metadata[this.dataOn].keys[name].push(0);
        }

        const index = this.metadata[this.dataOn].classes.indexOf(tar_arr[this.dataOn]);
        this.metadata[this.dataOn].keys[tar_arr[this.dataOn]][index] = 1;
        targets[this.dataOn] = tar_arr[this.dataOn];
      }

      
      this.trainData.push({ inputs, targets });
    } else if (this.format == 'numbered') {

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

          if (this.task === 'regression') {
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
            item.max = in_arr[label]
          }
        }
      }

      if (this.metadata[this.dataOn].classes.includes(tar_arr[this.dataOn])) {
        const index = this.metadata[this.dataOn].classes.indexOf(tar_arr[this.dataOn]);
        this.metadata[this.dataOn].keys[tar_arr[this.dataOn]][index] = 1;
        targets[this.dataOn] = tar_arr[this.dataOn];
      } else {
        if (this.metadata[this.dataOn].classes.length === this.on) {
          throw new Error('You have provided more outputs than specified in your nn cofigurations.');        
        }
        this.metadata[this.dataOn].classes.push(tar_arr[this.dataOn]);
        this.metadata[this.dataOn].keys[tar_arr[this.dataOn]] = [];
        for (let i = 0; i < this.on; i++) {
          this.metadata[this.dataOn].keys[tar_arr[this.dataOn]].push(0);
        }
        const index = this.metadata[this.dataOn].classes.indexOf(tar_arr[this.dataOn]);
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
        targets
      })
    }
    this.dataAdded++;
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
    for (let i = 0; i < this.metadata[this.dataOn].classes.length; i++) {
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

        ys.push(this.metadata[this.dataOn].keys[data.targets[this.dataOn]]);      
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
