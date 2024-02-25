import React, { useState } from 'react';
import './App.css';

function App() {
  const [learningRate, setLearningRate] = useState(0.01);
  const [batchSize, setBatchSize] = useState(64);
  const [epochs, setEpochs] = useState(10);
  const [dropoutRate, setDropoutRate] = useState(0.5);

  const handleSubmit = async (e) => {
    e.preventDefault();
    const response = await fetch('http://localhost:5000/train', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        learning_rate: learningRate,
        batch_size: batchSize,
        epochs: epochs,
        dropout_rate: dropoutRate,
      }),
    });
    const data = await response.json();
    console.log(data);
  };

  return (
    <div className="App">
      <header className="App-header">
        <form onSubmit={handleSubmit}>
          <div>
            <label>
              Learning Rate:
              <input
                type="number"
                value={learningRate}
                onChange={(e) => setLearningRate(e.target.value)}
              />
            </label>
          </div>
          <div>
            <label>
              Batch Size:
              <input
                type="number"
                value={batchSize}
                onChange={(e) => setBatchSize(e.target.value)}
              />
            </label>
          </div>
          <div>
            <label>
              Epochs:
              <input
                type="number"
                value={epochs}
                onChange={(e) => setEpochs(e.target.value)}
              />
            </label>
          </div>
          <div>
            <label>
              Dropout Rate:
              <input
                type="number"
                step="0.01"
                value={dropoutRate}
                onChange={(e) => setDropoutRate(e.target.value)}
              />
            </label>
          </div>
          <button type="submit">Start Training</button>
        </form>
      </header>
    </div>
  );
}

export default App;

