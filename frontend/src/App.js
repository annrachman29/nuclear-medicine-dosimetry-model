import React, { useState } from "react";
import axios from "axios";
import { Container, Form, Button, Table, Alert } from "react-bootstrap";
import 'bootstrap/dist/css/bootstrap.min.css';

// Import KaTeX
import 'katex/dist/katex.min.css';
import { BlockMath } from 'react-katex';

function App() {
  const [rows, setRows] = useState([]);
  const [time, setTime] = useState("");
  const [activity, setActivity] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  // Tambah row baru
  const addRow = () => {
    if (time === "" || activity === "") return;
    setRows([...rows, { Time: parseFloat(time), "%ID/gr": parseFloat(activity) }]);
    setTime("");
    setActivity("");
  };

  // Hapus semua row & reset result
  const clearRows = () => {
    setRows([]);
    setResult(null);
    setError("");
  };

  // Prediksi model dari backend
  const predict = async () => {
    if (rows.length === 0) {
      setError("Add at least one row before predicting!");
      return;
    }
    setError("");
    try {
      const res = await axios.post("http://127.0.0.1:8000/predict_model", { data: rows });
      if (res.data) {
        setResult({
          best_model: res.data.best_model || "Unknown",
          params: res.data.params || {},
          plot: res.data.plot || null,
          formula: res.data.formula || "" // Formula dari backend
        });
      } else {
        setError("No data returned from backend.");
      }
    } catch (err) {
      console.error(err);
      setError("API Error. Make sure backend is running.");
      setResult(null);
    }
  };

  return (
    <Container className="mt-4">
      <h2>Nuclear Medicine Dosimetry Modeller</h2>

      {error && <Alert variant="danger">{error}</Alert>}

      {/* Input data */}
      <Form className="mb-3 d-flex flex-wrap gap-2">
        <Form.Control
          type="number"
          placeholder="Time (h)"
          value={time}
          onChange={(e) => setTime(e.target.value)}
        />
        <Form.Control
          type="number"
          placeholder="%ID/gr"
          value={activity}
          onChange={(e) => setActivity(e.target.value)}
        />
        <Button onClick={addRow}>Add Row</Button>
        <Button variant="danger" onClick={clearRows}>Clear</Button>
      </Form>

      {/* Tabel data */}
      {rows.length > 0 && (
        <Table striped bordered hover responsive>
          <thead>
            <tr>
              <th>#</th>
              <th>Time (h)</th>
              <th>%ID/gr</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row, idx) => (
              <tr key={idx}>
                <td>{idx + 1}</td>
                <td>{row.Time}</td>
                <td>{row["%ID/gr"]}</td>
              </tr>
            ))}
          </tbody>
        </Table>
      )}

      <Button onClick={predict} className="mt-2">Predict Best Model</Button>

      {/* Hasil prediksi */}
      {result && (
        <div className="mt-3">
          <h4>Result:</h4>
          <p><strong>Best Model:</strong> {result.best_model}</p>

          {/* Render formula dengan KaTeX */}
          {result.formula && (
            <div className="mt-3">
              <h5>Formula:</h5>
              <BlockMath math={result.formula} />
            </div>
          )}

          {result.params && Object.keys(result.params).length > 0 ? 
          (
            <ul>
              {Object.entries(result.params).map(([key, val]) => (
                <li key={key}>{key}: {parseFloat(val).toFixed(6)}</li>
              ))}
            </ul>
          ) : (
            <p>No parameters returned.</p>
          )}

          {result.plot && (
            <div className="mt-3">
              <h5>Curve Fit Plot:</h5>
              <img
                src={`data:image/png;base64,${result.plot}`}
                alt="Curve Fit"
                style={{ maxWidth: "100%", height: "auto", border: "1px solid #ccc" }}
              />
            </div>
          )}
        </div>
      )}
    </Container>
  );
}

export default App;
