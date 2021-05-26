function sigmoid(t) {
  return 1 / (1 + Math.pow(Math.E, -t));
}

function derivadaSigmoid(t) {
  return t * (1 - t);
}

function setup() {
  createCanvas(500, 500);
  background(0);

  dataset = {
    inputs: [[1, 1],
    [1, 0],
    [0, 1],
    [0, 0]],
    outputs: [[0],
    [1],
    [1],
    [0]]
  }

  
  entrada = [[1,1]];
  saida = [[5]];

  rede = new RedeNeural(2, 3, 1);
}
var train = true;
function draw() {
  if (train) {
    for (var i = 0; i < 10000; i++) {
      var index = floor(random(4));
      rede.train(dataset.inputs[index], dataset.outputs[index]);
      //rede.train(entrada[index], saida[index]);
    }
    if (rede.predict([0, 0])[0] < 0.04 && rede.predict([1, 0])[0] > 0.98) {
      train = false;
      console.log("terminou");
    }
  }
}

class RedeNeural {

  constructor(i_nodes, h_nodes, o_nodes) {
    this.i_nodes = i_nodes;
    this.h_nodes = h_nodes;
    this.o_nodes = o_nodes;

    this.bias_ih = new Matriz(this.h_nodes, 1);
    this.bias_ih.randomize();
    this.bias_ho = new Matriz(this.o_nodes, 1);
    this.bias_ho.randomize();

    this.weigths_ih = new Matriz(this.h_nodes, this.i_nodes);
    this.weigths_ih.randomize()

    this.weigths_ho = new Matriz(this.o_nodes, this.h_nodes)
    this.weigths_ho.randomize()

    this.learning_rate = 0.1;
}

train(arr,target) {
    // INPUT -> HIDDEN
    let input = Matriz.arrayToMatriz(arr);
    let hidden = Matriz.multiply(this.weigths_ih, input);
    hidden = Matriz.add(hidden, this.bias_ih);

    hidden.map(sigmoid)

    // HIDDEN -> OUTPUT
    // d(Sigmoid) = Output * (1- Output)
    let output = Matriz.multiply(this.weigths_ho, hidden);
    output = Matriz.add(output, this.bias_ho);
    output.map(sigmoid);

    // BACKPROPAGATION

    // OUTPUT -> HIDDEN
    let expected = Matriz.arrayToMatriz(target);
    let output_error = Matriz.subtract(expected,output);
    let d_output = Matriz.map(output,derivadaSigmoid);
    let hidden_T = Matriz.transpose(hidden);

    let gradient = Matriz.hadamard(d_output,output_error);
    gradient = Matriz.escalar_multiply(gradient,this.learning_rate);
    
    // Adjust Bias O->H
    this.bias_ho = Matriz.add(this.bias_ho, gradient);
    // Adjust Weigths O->H
    let weigths_ho_deltas = Matriz.multiply(gradient,hidden_T);
    this.weigths_ho = Matriz.add(this.weigths_ho,weigths_ho_deltas);

    // HIDDEN -> INPUT
    let weigths_ho_T = Matriz.transpose(this.weigths_ho);
    let hidden_error = Matriz.multiply(weigths_ho_T,output_error);
    let d_hidden = Matriz.map(hidden,derivadaSigmoid);
    let input_T = Matriz.transpose(input);

    let gradient_H = Matriz.hadamard(d_hidden,hidden_error);
    gradient_H = Matriz.escalar_multiply(gradient_H, this.learning_rate);

    // Adjust Bias O->H
    this.bias_ih = Matriz.add(this.bias_ih, gradient_H);
    // Adjust Weigths H->I
    let weigths_ih_deltas = Matriz.multiply(gradient_H, input_T);
    this.weigths_ih = Matriz.add(this.weigths_ih, weigths_ih_deltas);
}

predict(arr){
    // INPUT -> HIDDEN
    let input = Matriz.arrayToMatriz(arr);

    let hidden = Matriz.multiply(this.weigths_ih, input);
    hidden = Matriz.add(hidden, this.bias_ih);

    hidden.map(sigmoid)

    // HIDDEN -> OUTPUT
    let output = Matriz.multiply(this.weigths_ho, hidden);
    output = Matriz.add(output, this.bias_ho);
    output.map(sigmoid);
    output = Matriz.MatrizToArray(output);

    return output;
}

}

class Matriz {
  constructor(rows, cols) {
    this.rows = rows;
    this.cols = cols;

    this.data = [];

    for (let i = 0; i < rows; i++) {
        let arr = []
        for (let j = 0; j < cols; j++) {
            arr.push(0)
        }
        this.data.push(arr);
    }
}

// Funções Diversas

static arrayToMatriz(arr) {
    let matriz = new Matriz(arr.length, 1);
    matriz.map((elm, i, j) => {
        return arr[i];
    })
    return matriz;
}

static MatrizToArray(obj) {
    let arr = []
    obj.map((elm, i, j) => {
        arr.push(elm);
    })
    return arr;
}


print() {
    console.table(this.data);
}

randomize() {
    this.map((elm, i, j) => {
        return Math.random() * 2 - 1;
    });
}

static map(A, func) {
    let matriz = new Matriz(A.rows, A.cols);

    matriz.data = A.data.map((arr, i) => {
        return arr.map((num, j) => {
            return func(num, i, j);
        })
    })

    return matriz;
}

map(func) {

    this.data = this.data.map((arr, i) => {
        return arr.map((num, j) => {
            return func(num, i, j);
        })
    })

    return this;
}

static transpose(A){
    var matriz = new Matriz(A.cols, A.rows);
    matriz.map((num,i,j) => {
        return A.data[j][i];
    });
    return matriz;
}

// Operações Estáticas Matriz x Escalar 

static escalar_multiply(A, escalar) {
    var matriz = new Matriz(A.rows, A.cols);

    matriz.map((num, i, j) => {
        return A.data[i][j] * escalar;
    });

    return matriz;
}

// Operações Estáticas Matriz x Matriz

static hadamard(A, B) {
    var matriz = new Matriz(A.rows, A.cols);

    matriz.map((num, i, j) => {
        return A.data[i][j] * B.data[i][j]
    });

    return matriz;
}

static add(A, B) {
    var matriz = new Matriz(A.rows, A.cols);

    matriz.map((num, i, j) => {
        return A.data[i][j] + B.data[i][j]
    });

    return matriz;
}

static subtract(A, B) {
    var matriz = new Matriz(A.rows, A.cols);

    matriz.map((num, i, j) => {
        return A.data[i][j] - B.data[i][j]
    });

    return matriz;
}

static multiply(A, B) {
    var matriz = new Matriz(A.rows, B.cols);

    matriz.map((num, i, j) => {
        let sum = 0
        for (let k = 0; k < A.cols; k++) {
            let elm1 = A.data[i][k];
            let elm2 = B.data[k][j];
            sum += elm1 * elm2;
        }
        
        return sum;
    })

    return matriz;
}

}
