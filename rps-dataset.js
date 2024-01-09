class RPSDataset {
  constructor() {
    this.labels = [];
  }

  /* parameter 설명: 
    example: truncated mobilenet의 output (tensor)
    label: 0, 1, 2 (integer)
  */
  addExample(example, label) {
    if (this.xs == null) {
      // 첫번째 example인 경우
      this.xs = example;
      this.labels.push(label);
    } else {
      const oldX = this.xs;
      this.xs = oldX.concat(example, 0); // 두번째 example부터는 이전 example들과 연결
      this.labels.push(label);
      oldX.dispose();
    }
  }

  // label들을 one-hot-encoding
  encodeLabels(numClasses) {
    for (var i = 0; i < this.labels.length; i++) {
      if (this.ys == null) {
        // 첫번째 example인 경우
        this.ys = tf.tidy(() => {
          return tf.oneHot(tf.tensor1d([this.labels[i]]).toInt(), numClasses);
        });
      } else {
        const y = tf.tidy(() => {
          return tf.oneHot(
            // 두번째 example부터는 이전 example들과 연결
            tf.tensor1d([this.labels[i]]).toInt(),
            numClasses
          );
        });
        const oldY = this.ys;
        this.ys = oldY.concat(y, 0);
        oldY.dispose();
        y.dispose();
      }
    }
  }
}
